import numpy as np
import os
import argparse
import cv2
from glob import glob
import pathlib
import random

import shutil


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_data_train', type=int, default=30)
    parser.add_argument('--num_data_val', type=int, default=10)
    parser.add_argument("--input_path", type=str , default="/data/xzy/elon/TransLiDAR_bin/")
    parser.add_argument("--output_path_name_train", type = str, default = "train/lidar")
    parser.add_argument("--output_path_name_val", type = str, default = "val/lidar")
    # parser.add_argument("--output_path_label_train", type = str, default = "train/label_array")
    # parser.add_argument("--output_path_label_val", type = str, default = "val/label_array")
    parser.add_argument("--create_val", action='store_true', default=True)
   
    return parser.parse_args()


def create_range_map(points_array, label_lidar, image_rows_full, image_cols, ang_start_y, ang_res_y, ang_res_x, max_range, min_range):
    range_image = np.zeros((image_rows_full, image_cols, 1), dtype=np.float32)
    label_map = np.zeros((image_rows_full, image_cols, 1), dtype=np.float32)
    x = points_array[:,0]
    y = points_array[:,1]
    z = points_array[:,2]
    label = label_lidar
    # find row id

    vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
    relative_vertical_angle = vertical_angle + ang_start_y
    rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
    # Inverse sign of y for kitti data
    horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi

    colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2

    shift_ids = np.where(colId>=image_cols)
    colId[shift_ids] = colId[shift_ids] - image_cols
    colId = colId.astype(np.int64)
    # filter range
    thisRange = np.sqrt(x * x + y * y + z * z)
    thisRange[thisRange > max_range] = 0
    thisRange[thisRange < min_range] = 0

    # filter Internsity
    label[thisRange > max_range] = 0
    label[thisRange < min_range] = 0


    valid_scan = (rowId >= 0) & (rowId < image_rows_full) & (colId >= 0) & (colId < image_cols)

    rowId_valid = rowId[valid_scan]
    colId_valid = colId[valid_scan]
    thisRange_valid = thisRange[valid_scan]
    label_valid = label[valid_scan]

    range_image[rowId_valid, colId_valid, :] = thisRange_valid.reshape(-1, 1)
    label_map[rowId_valid, colId_valid, :] = label_valid.reshape(-1, 1)

    lidar_data_projected = np.concatenate((range_image, label_map), axis = -1)

    return lidar_data_projected


def load_from_bin(bin_path):
    lidar_data = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)
    # ignore reflectivity info
    return lidar_data

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def main(args):
    num_data_train = args.num_data_train
    num_data_val = args.num_data_val
    dir_name = os.path.dirname(args.input_path)
    output_dir_name_train = os.path.join(dir_name, args.output_path_name_train)
    # output_dir_label_train = os.path.join(dir_name, args.output_path_label_train)
    pathlib.Path(output_dir_name_train).mkdir(parents=True, exist_ok=True)
    # pathlib.Path(output_dir_label_train).mkdir(parents=True, exist_ok=True)
    if args.create_val:
        output_dir_name_val = os.path.join(dir_name, args.output_path_name_val)
        # output_dir_label_val = os.path.join(dir_name, args.output_path_label_val)
        pathlib.Path(output_dir_name_val).mkdir(parents=True, exist_ok=True)
        # pathlib.Path(output_dir_label_val).mkdir(parents=True, exist_ok=True)

    train_split_path = "/userHome/xzy/Projects/elon/TransLiDAR_TULIP/kitti_utils/train_files.txt"
    val_split_path = "/userHome/xzy/Projects/elon/TransLiDAR_TULIP/kitti_utils/val_files.txt"

    train_split = np.array(readlines(train_split_path), dtype = str)
    val_split = np.array(readlines(val_split_path), dtype = str)

    train_data = []
    val_data = []

    # If the required data number is lower than the total number of scan, then sample the scan
    
    
    for train_folder in train_split:
        files = glob(os.path.join(dir_name, train_folder, "lidar/*.bin"))
        files.sort() 
        sample_train_data = np.array(files)
        # sample_train_data = np.array(glob(os.path.join(dir_name, train_folder, "array/*.bin")).sort())
        train_data.extend(sample_train_data)
     
        
    # assert len(train_data) == num_data_train, "The number of training data is not correct"  


    
    for val_folder in val_split:
        files = glob(os.path.join(dir_name, val_folder, "lidar/*.bin"))
        files.sort() 
        sample_val_data = np.array(files)
        # sample_val_data = np.array(glob(os.path.join(dir_name, val_folder, "array/*.bin")))
        val_data.extend(sample_val_data)


    # assert len(val_data) == num_data_val, "The number of validation data is not correct"


    image_rows = 64
    image_cols = 256 # 256
    ang_start_y = 12.55 # 12.55
    ang_res_y = 35.05 / (image_rows -1) # 35.05
    ang_res_x = 81.7 / image_cols # 81.7
    max_range = 90
    min_range = 0
    
    
    # Move the data to the output directory
    for i, train_data_path in enumerate(train_data):

        lidar_data = load_from_bin(train_data_path)
        label_lidar_path = train_data_path.replace('/lidar/', '/labels_lidar/').replace('.bin', '.npy')
        label_lidar = np.load(label_lidar_path)
        # lidar_data = lidar_data[lidar_data[:,2]>-0.4]
        range_intensity_map = create_range_map(lidar_data,label_lidar, image_rows_full = image_rows, image_cols = image_cols, ang_start_y = ang_start_y, ang_res_y = ang_res_y, ang_res_x = ang_res_x, max_range = max_range, min_range = min_range)

        np.save(os.path.join(output_dir_name_train,'{:08d}.npy'.format(i)), range_intensity_map.astype(np.float32))
        

    if args.create_val:
        for j, val_data_path in enumerate(val_data):
            lidar_data = load_from_bin(val_data_path)
            label_lidar_path = val_data_path.replace('/lidar/', '/labels_lidar/').replace('.bin', '.npy')
            label_lidar = np.load(label_lidar_path)
            # lidar_data = lidar_data[lidar_data[:,2]>-0.4]
            range_intensity_map = create_range_map(lidar_data, label_lidar, image_rows_full = image_rows, image_cols = image_cols, ang_start_y = ang_start_y, ang_res_y = ang_res_y, ang_res_x = ang_res_x, max_range = max_range, min_range = min_range)
            np.save(os.path.join(output_dir_name_val,'{:08d}.npy'.format(j)), range_intensity_map.astype(np.float32))
            

if __name__ == "__main__":
    args = read_args()
    main(args)
    