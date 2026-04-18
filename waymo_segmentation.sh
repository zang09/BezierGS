#!/usr/bin/env bash
set -euo pipefail

# This script processes images with three text prompts:
# 1. car
# 2. bus. truck
# 3. pedestrian

# Usage:
#   bash waymo_segmentation.sh SCENE_ID
#   bash waymo_segmentation.sh SCENE_ID 0,1,2
#   bash waymo_segmentation.sh SCENE_ID 0 1 2

SCENE_ID=${1:?Usage: bash waymo_segmentation.sh SCENE_ID [GPU0,GPU1,GPU2 | GPU0 GPU1 GPU2]}
shift

if [ $# -eq 0 ]; then
    CAM_DEVICES=(5 6 7)
elif [ $# -eq 1 ]; then
    IFS=',' read -r -a CAM_DEVICES <<< "$1"
elif [ $# -eq 3 ]; then
    CAM_DEVICES=("$1" "$2" "$3")
else
    echo "Usage: bash waymo_segmentation.sh SCENE_ID [GPU0,GPU1,GPU2 | GPU0 GPU1 GPU2]" >&2
    exit 1
fi

if [ "${#CAM_DEVICES[@]}" -ne 3 ]; then
    echo "Expected exactly 3 GPU device ids, got: ${CAM_DEVICES[*]}" >&2
    exit 1
fi

echo "Scene ID: $SCENE_ID"
echo "Camera devices: cam0=${CAM_DEVICES[0]}, cam1=${CAM_DEVICES[1]}, cam2=${CAM_DEVICES[2]}"

# Define the text prompts to use.
TEXT_PROMPTS=("car" "bus. truck" "pedestrian")

# Process each text prompt.
for text_prompt in "${TEXT_PROMPTS[@]}"; do
    echo "Processing text prompt: $text_prompt"
    
    # First device processes camera 0 images.
    CUDA_VISIBLE_DEVICES="${CAM_DEVICES[0]}" python grounded_sam_demo.py \
        --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth \
        --sam_checkpoint Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth \
        --image_dir /home/haebeom/data/BezierGS/Waymo/$SCENE_ID/images \
        --output_dir /home/haebeom/data/BezierGS/Waymo/$SCENE_ID/seg_npy \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "$text_prompt" \
        --device "cuda" \
        --file_pattern "*_0.png" &

    # Second device processes camera 1 images.
    CUDA_VISIBLE_DEVICES="${CAM_DEVICES[1]}" python grounded_sam_demo.py \
        --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth \
        --sam_checkpoint Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth \
        --image_dir /home/haebeom/data/BezierGS/Waymo/$SCENE_ID/images \
        --output_dir /home/haebeom/data/BezierGS/Waymo/$SCENE_ID/seg_npy \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "$text_prompt" \
        --device "cuda" \
        --file_pattern "*_1.png" &

    # Third device processes camera 2 images.
    CUDA_VISIBLE_DEVICES="${CAM_DEVICES[2]}" python grounded_sam_demo.py \
        --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounded_checkpoint Grounded-Segment-Anything/weights/groundingdino_swint_ogc.pth \
        --sam_checkpoint Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth \
        --image_dir /home/haebeom/data/BezierGS/Waymo/$SCENE_ID/images \
        --output_dir /home/haebeom/data/BezierGS/Waymo/$SCENE_ID/seg_npy \
        --box_threshold 0.3 \
        --text_threshold 0.25 \
        --text_prompt "$text_prompt" \
        --device "cuda" \
        --file_pattern "*_2.png" &

    # Wait for all background processes for the current text prompt to finish.
    wait
    
    echo "Finished processing text prompt: $text_prompt"
done

echo "All processing complete"
