import os
import open3d as o3d
import struct
import numpy as np
# 指定.pcd文件所在的文件夹路径
pcd_folder = '/data/xzy/elon/TransLiDAR/sequences/00/lidar/' # '/mnt/usb/data_cqupt_completion/completion_data/area_array_lidar/data13/points_area/'

# 确保输出的.bin文件夹存在
output_folder = '/data/xzy/elon/TransLiDAR_bin/sequences/00/lidar/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def extract_number_from_filename(filename):
    # 假设文件名为类似 '1.pcd', '2.pcd', '10.pcd' 格式，提取数字部分
    return int(filename.split('.')[0])


# 遍历文件夹中的所有.pcd文件
for i, filename in enumerate(sorted(os.listdir(pcd_folder), key=extract_number_from_filename)):
    if filename.endswith(".pcd"):
        pcd_file_path = os.path.join(pcd_folder, filename)
        
        # 读取.pcd文件
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        
        # 将点云数据转换为NumPy数组
        points = o3d.utility.Vector3dVector(pcd.points) # np.array(pcd.points)
        
        # 构造输出的.bin文件名，采用6位数字命名
        bin_file_name = f'{i:06d}.bin'
        bin_file_path = os.path.join(output_folder, bin_file_name)
        
        # 保存点云数据为二进制格式的.bin文件
        with open(bin_file_path, 'wb') as bin_file:
            for point in points:
                bin_file.write(struct.pack('fff', point[0], point[1], point[2]))
