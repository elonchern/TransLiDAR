# TransLiDAR: A dataset and benchmark for Cross-Sensor Point Cloud Translation
This is an official implementation of the paper **TransLiDAR: A dataset and benchmark for Cross-Sensor Point Cloud Translation**.
## Installation
Our work is implemented with the following environmental setups:
* Python == 3.8
* PyTorch == 1.12.0
* CUDA == 11.3

You can use conda to create the correct environment:
```
conda create -n myenv python=3.8
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

Then, install the dependencies in the environment:
```
pip install -r requirements.txt
pip install git+'https://github.com/otaheri/chamfer_distance'  # need access to gpu for compilation
```
You can refer to more details about chamfer distance package from https://github.com/otaheri/chamfer_distance

## TransLiDAR Dataset 
The TransLiDAR dataset is collected using an autonomous vehicle platform equipped with two types of LiDAR sensors: an Ouster 64-line mechanical LiDAR mounted on the roof and a Livox hybrid solid-state LiDAR mounted at the front of the vehicle, as shown in Figure 1. Data collection takes place on a university campus, where 12 routes are designed to comprehensively cover the entire area, as illustrated in Figure 2. In total, the dataset contains 15,641 pairs of point cloud data. Each pair includes a frame of synchronized point clouds from both the mechanical LiDAR and the hybrid solid-state LiDAR. Detailed information is provided in Table 1.

<img src="figures/fig1.jpg" alt="figure_1：Dataset directory structure" width="400" height="300"/>

You can download the dataset via Baidu Netdisk using the following link: https://pan.baidu.com/s/1wX8j819NX-fGRABLRTuPPA?pwd=z14b **Extraction code: z14b** 
The transLiDA dataset should be structured in this way:
```
TransLiDAR
│
├── sequences                          # The data includes 12 routes.
│   │
│   ├── 00                              # The data of the first route.
│   │   ├── lidar                       # The data of the rotating mechanical LiDAR.
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   └── ...
│   │   ├── hybrid                      # The data of the Hybrid semi-solid-state LiDAR.
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   └── ...
│   │   ├── image                       # Image data captured by the camera.
│   │   │   ├── 000000.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ...
│   │   └── Calib.txt                   # The intrinsic and extrinsic matrices between the hybrid and the mechanical LiDAR.
│   │
│   ├── 01                              # The data of the second route.
│   ├── ...                             # Additional routes (02 to 10)
│   └── 11                              # The data of the last route.
```

After downloading the raw dataset, create train and test split for LiDAR upsampling:
```
bash bash_scripts/create_durlar_dataset.sh
bash bash_scripts/create_kitti_dataset.sh
```
The new dataset should be structured in this way:
```

dataset
│
└─── TransLiDA
   │
   └───train
   │   │   00000001.npy
   │   │   00000002.npy
   │   │   ...
   └───val
       │   00000001.npy
       │   00000002.npy
       │   ...
```

## Training
We provide some bash files for running the experiment quickly with default settings. 
```
bash bash_scripts/tulip_upsampling_kitti.sh (KITTI)
bash bash_scripts/tulip_upsampling_carla.sh (CARLA)
bash bash_scripts/tulip_upsampling_durlar.sh (DurLAR)
```

## Evaluation
After the model training is completed, you can find the saved model weights under the `experiment/` directory. By default, the model weights are saved every 100 epochs. Then, run the evaluation code below:
```
bash bash_scripts/tulip_evaluation_kitti.sh (KITTI)
bash bash_scripts/tulip_evaluation_carla.sh (CARLA)
bash bash_scripts/tulip_evaluation_durlar.sh (DurLAR)
```

## Citation
```
@inproceedings{yang2024tulip,
  title={TULIP: Transformer for Upsampling of LiDAR Point Clouds},
  author={Yang, Bin and Pfreundschuh, Patrick and Siegwart, Roland and Hutter, Marco and Moghadam, Peyman and Patil, Vaishakh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15354--15364},
  year={2024}
}
```


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

