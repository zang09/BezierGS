#!/bin/bash

scene_list=(017 026 090 145 147 158 181)

## Use env: grounded_sam ##
for scene in "${scene_list[@]}"; do
    # echo "Processing scene $scene sky mask..."
    # python script/waymo/generate_sky_mask.py --datadir /home/haebeom/data/BezierGS/Waymo/"$scene" --sam_checkpoint Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth
    
    echo "Processing scene $scene generating instance segmentation..."
    bash waymo_segmentation.sh $scene 5 6 7
done