# TransLiDAR dataset

TransLiDAR is a cross-sensor point cloud translation dataset designed to bridge the modality gaps between different types of LiDAR sensors commonly used in autonomous vehicles. These vehicles are typically equipped with a combination of primary and auxiliary LiDAR units that vary significantly in structural design, resolution, and scanning mechanisms—leading to challenges in cross-sensor adaptation and joint learning. To address this issue, TransLiDAR provides paired point clouds captured from both mechanical and hybrid solid-state LiDAR sensors within the same scenes. This enables effective learning-based translation and alignment between heterogeneous LiDAR modalities, facilitating research in cross-sensor point cloud understanding and domain adaptation.

数据集的相关信息如下所示：

## 图示信息

- **Dataset directory structure** 
- ![figure_1：Dataset directory structur](figures/fig1.jpg)
  
- **图2：** 地理配准位姿叠加在 CQUPT 地图上  
  *Figure 2: Georegistered poses overlaid on CQUPTMap*

- **图3：** 采集序列和轨迹的可视化概览  
  *Figure 3: Qualitative overview of sequences and trajectories*

## 表格信息

**表1：数据集统计信息**  
*Table 1: Dataset Statistics*

| Sequence | Frames (LiDAR) | Frames (Area Array) | Data Volume |
|----------|----------------|---------------------|-------------|
| 00       | 3047           | 3047                | 1.21 GB     |
| 01       | 1503           | 1503                | 0.55 GB     |
| 02       | 64             | 64                  | 0.08 GB     |
| 03       | 1572           | 1572                | 0.57 GB     |
| 04       | 600            | 600                 | 0.22 GB     |
| 05       | 600            | 600                 | 0.21 GB     |
| 06       | 561            | 561                 | 0.21 GB     |
| 07       | 770            | 770                 | 0.29 GB     |
| 08       | 3200           | 3200                | 1.17 GB     |
| 09       | 37             | 37                  | 0.01 GB     |
| 10       | 910            | 910                 | 1.15 GB     |
| 11       | 2770           | 2770                | 1.02 GB     |

