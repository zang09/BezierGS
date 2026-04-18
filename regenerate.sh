#!/bin/bash

scene_list=(017 026 090 145 147 158 181)

## Use env: bezier ##
for scene in "${scene_list[@]}"; do
    echo "Processing scene $scene regenerating camera projection..."

    python script/waymo/regenerate_camera_projection.py \
      --data_root /home/haebeom/data/BezierGS/Waymo/${scene_num} \
      --output_dir /home/haebeom/data/BezierGS/Waymo/${scene_num} \
      --extrinsics_dir extrinsics_fixed \
      --camera-list 0 1 2 3 4 \
      --write-compatible-pointcloud \
      --compatible-output-name pointcloud_fixed.npz
done
