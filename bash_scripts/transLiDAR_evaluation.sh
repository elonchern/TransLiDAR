#!/bin/bash

args=(
    --eval
    # --mc_drop
    --save_pcd
    --noise_threshold 0.03
    --model_select tulip_base
    --pixel_shuffle
    --circular_padding
    --patch_unmerging
    --log_transform
    # Dataset
    --dataset_select kitti
    --data_path_low_res /data/elon/TransLiDAR_bin/
    --data_path_high_res /data/elon/TransLiDAR_bin/
    # --save_pcd
    # WandB Parameters
    --run_name tulip_base
    --entity myentity
    # --wandb_disabled
    --project_name kitti_evaluation
    --output_dir /home/elon/Workshops/TransLiDAR_V1/experiment/kitti/tulip_base/checkpoint-599.pth
    --img_size_low_res 64 256
    --img_size_high_res 64 256
    --window_size 2 8
    --patch_size 4 4
    --in_chans 1
    )

torchrun --nproc_per_node=1 --master_port=29501 tulip/main_lidar_upsampling.py "${args[@]}"