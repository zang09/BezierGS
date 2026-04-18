#!/usr/bin/env bash
set -euo pipefail

scene_num=017
use_fixed=false
use_float=false
train_args=()

usage() {
    echo "Usage: $0 [scene_num] [--fixed|--float] [train.py overrides...]"
}

if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    scene_num=$1
    shift
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --fixed)
            use_fixed=true
            train_args+=("$1")
            ;;
        --float|--floater)
            use_float=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            train_args+=("$1")
            ;;
    esac
    shift
done

config_name=${scene_num}
model_name=waymo/${scene_num}

if [ "$use_float" = true ]; then
    config_name=${config_name}_floater
    model_name=${model_name}/floater
fi

if [ "$use_fixed" = true ]; then
    model_name=${model_name}_fixed
fi

config_path=configs/waymo/${config_name}.yaml

time=$(date "+%Y-%m-%d_%H:%M:%S")
# model_path=outputs/${model_name}/${time}
model_path=/home/haebeom/dev/GS/BezierGS/outputs/waymo/017/floater_fixed/2026-04-18_08:29:35

# python train.py \
#     --config ${config_path} \
#     source_path=/home/haebeom/data/BezierGS/Waymo/${scene_num} \
#     model_path=${model_path} \
#     "${train_args[@]}"

render_args=()
if [ "$use_fixed" = true ]; then
    render_args+=(fixed=true)
fi

python3 render.py \
    --config ${config_path} \
    source_path=/home/haebeom/data/BezierGS/Waymo/${scene_num} \
    model_path=${model_path} \
    checkpoint=${model_path}/chkpnt30000.pth \
    "${render_args[@]}" \
    render_split=test \
    render_camera_id=0 \
    render_interp_steps=10 \
    render_fps=30
