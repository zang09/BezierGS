scene_num=${1:-017}

python evaluate.py \
    --config configs/waymo/${scene_num}.yaml \
    source_path=/home/haebeom/data/BezierGS/Waymo/${scene_num} \
    model_path=outputs/waymo_nvs/${scene_num} \
    checkpoint=outputs/waymo_nvs/${scene_num}/chkpnt30000.pth