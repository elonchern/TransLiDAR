import torch
import numpy as np
import os
import yaml
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from pathlib import Path
from nuscenes.utils import splits
from torchvision import transforms as T
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))


class TransLiDAR(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        # self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset

        if imageset == 'train':
            split = semkittiyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += semkittiyaml['split']['valid']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'array']), num_vote)
            calib_path = os.path.join(data_path, str(i_folder).zfill(2), "calib.txt")
            calib = self.read_calib(calib_path)
            proj_matrix = np.matmul(calib["P2"], calib["Tr"])
            self.proj_matrix[i_folder] = proj_matrix

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 3))
        origin_len = len(raw_data)
        points = raw_data[:, :3]

        # if self.imageset == 'test':
        #     annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        #     instance_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        # else:
        #     annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
        #                                  dtype=np.uint32).reshape((-1, 1))
        #     instance_label = annotated_data >> 16
        #     annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
        #     annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data) # 函数向量化

        #     if self.config['dataset_params']['ignore_label'] != 0:
        #         annotated_data -= 1
        #         annotated_data[annotated_data == -1] = self.config['dataset_params']['ignore_label']

        image_file = self.im_idx[index].replace('array', 'image_2').replace('.bin', '.jpg')
        image = Image.open(image_file)
        proj_matrix = self.proj_matrix[int(self.im_idx[index][-19:-17])]

        data_dict = {}
        data_dict['xyz'] = points
        # data_dict['labels'] = annotated_data.astype(np.uint8)
        # data_dict['instance_label'] = instance_label
        # data_dict['signal'] = raw_data[:, 3:4]
        data_dict['origin_len'] = origin_len
        data_dict['img'] = image
        data_dict['proj_matrix'] = proj_matrix

        return data_dict, self.im_idx[index]
    
    
    
class point_image_dataset_semkitti(data.Dataset):
    def __init__(self, in_dataset, config):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
        self.debug = config['debug']

        

    def __len__(self):
        'Denotes the total number of samples'
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        # labels = data['labels']
        # instance_label = data['instance_label']
        # sig = data['signal']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        # ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        point_num = len(xyz)


        # load 2D data
        image = data['img']
        proj_matrix = data['proj_matrix']

        # project points into image
        keep_idx = xyz[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([xyz[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ points_hcoords.T).T
        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image.size)
        keep_idx[keep_idx] = keep_idx_img_pts

        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)
        points_img = img_points[keep_idx_img_pts]

        # img_label = labels[keep_idx]

        point2img_index = np.arange(len(xyz))[keep_idx]
        # feat = np.concatenate((xyz, sig), axis=1)
        feat = xyz

        img_indices = points_img.astype(np.int64)

        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.


        # ------------ 可视化点投影到图像上 ----------- #
        
        # proj_label = np.zeros((image.shape[0], image.shape[1],1), dtype=np.float32)
        # proj_label[img_indices[:,0],img_indices[:,1]] = labels[point2img_index]
        # proj_instance_label = np.zeros((image.shape[0], image.shape[1],1), dtype=np.float32)
        # proj_instance_label[img_indices[:,0],img_indices[:,1]] = instance_label[point2img_index]
        
        # -------------------------------------------- #           

        data_dict = {}
        data_dict['point_feat'] = feat
        # data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        # data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root

        data_dict['img'] = image
        data_dict['img_indices'] = img_indices
        # data_dict['img_label'] = img_label
        data_dict['point2img_index'] = point2img_index
        # data_dict['proj_label'] = proj_label
        # data_dict['proj_instance_label'] = proj_instance_label
        return data_dict
    
    
def collate_fn_default(data):
    point_num = [d['point_num'] for d in data]
    batch_size = len(point_num)
    # ref_labels = data[0]['ref_label']
    origin_len = data[0]['origin_len']
    ref_indices = [torch.from_numpy(d['ref_index']) for d in data]
    point2img_index = [torch.from_numpy(d['point2img_index']).long() for d in data]
    path = [d['root'] for d in data]

    img = [torch.from_numpy(d['img']) for d in data]
    img_indices = [d['img_indices'] for d in data]
    # img_label = [torch.from_numpy(d['img_label']) for d in data]

    b_idx = []
    for i in range(batch_size):
        b_idx.append(torch.ones(point_num[i]) * i)
    points = [torch.from_numpy(d['point_feat']) for d in data]
    ref_xyz = [torch.from_numpy(d['ref_xyz']) for d in data]
    # labels = [torch.from_numpy(d['point_label']) for d in data]
    # proj_label = [torch.from_numpy(d['proj_label']) for d in data]
    # proj_instance_label = [torch.from_numpy(d['proj_instance_label']) for d in data]

    return {
        'points': torch.cat(points).float(),
        'ref_xyz': torch.cat(ref_xyz).float(),
        'batch_idx': torch.cat(b_idx).long(),
        'batch_size': batch_size,
        # 'labels': torch.cat(labels).long().squeeze(1),
        # 'raw_labels': torch.from_numpy(ref_labels).long(),
        'origin_len': origin_len,
        'indices': torch.cat(ref_indices).long(),
        'point2img_index': point2img_index,
        'img': torch.stack(img, 0).permute(0, 3, 1, 2),
        'img_indices': img_indices,
        # 'img_label': torch.cat(img_label, 0).squeeze(1).long(),
        'path': path,
        # 'proj_label': torch.stack(proj_label,0).long(),
        # 'proj_instance_label': torch.stack(proj_instance_label,0).long(),
    }

