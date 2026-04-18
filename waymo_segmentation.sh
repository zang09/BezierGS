# This script processes images with three text prompts:
# 1. car
# 2. bus. truck
# 3. pedestrian

# Define the text prompts to use.
SCENE_ID=$1
TEXT_PROMPTS=("car" "bus. truck" "pedestrian")

# Process each text prompt.
for text_prompt in "${TEXT_PROMPTS[@]}"; do
    echo "Processing text prompt: $text_prompt"
    
    # GPU 0 processes camera 0 images.
    CUDA_VISIBLE_DEVICES=5 python grounded_sam_demo.py \
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

    # GPU 1 processes camera 1 images.
    CUDA_VISIBLE_DEVICES=6 python grounded_sam_demo.py \
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

    # GPU 2 processes camera 2 images.
    CUDA_VISIBLE_DEVICES=7 python grounded_sam_demo.py \
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
