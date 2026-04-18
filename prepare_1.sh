#!/bin/bash

scene_list=(017 026 090 145 147 158 181)

## Use env: waymo ##
python script/waymo/waymo_converter.py \
    --root_dir /home/haebeom/data/Waymo_v2/raw \
    --save_dir /home/haebeom/data/BezierGS/Waymo \
    --split_file script/waymo/waymo_splits/tlc_calib_dynamic.txt \
    --segment_file script/waymo/waymo_splits/segment_list_train.txt

for scene in "${scene_list[@]}"; do
    echo "Processing scene $scene lidar depth..."
    python script/waymo/generate_lidar_depth.py --datadir /home/haebeom/data/BezierGS/Waymo/"$scene"
done