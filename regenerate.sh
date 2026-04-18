scene_num=${1:-017}

python3 script/waymo/regenerate_camera_projection.py \
  --data_root /home/haebeom/data/BezierGS/Waymo/${scene_num} \
  --output_dir /home/haebeom/data/BezierGS/Waymo/${scene_num} \
  --extrinsics_dir extrinsics_fixed \
  --camera-list 0 1 2 3 4 \
  --write-compatible-pointcloud \
  --compatible-output-name pointcloud_fixed.npz