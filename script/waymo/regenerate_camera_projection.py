import argparse
import json
import os
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm


UNKNOWN_CAMERA = -1


def load_intrinsics(data_root, cam_id):
    intrinsic = np.loadtxt(os.path.join(data_root, "intrinsics", f"{cam_id}.txt")).reshape(-1)
    fx, fy, cx, cy = intrinsic[:4]
    distortion = intrinsic[4:9] if intrinsic.shape[0] >= 9 else np.zeros(5, dtype=np.float64)
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return k, distortion.astype(np.float64)


def image_size(data_root, cam_id, frame_ids):
    image_dir = os.path.join(data_root, "images")
    for frame_id in frame_ids:
        image_path = os.path.join(image_dir, f"{frame_id:06d}_{cam_id}.png")
        if os.path.exists(image_path):
            with Image.open(image_path) as image:
                return image.size
    raise FileNotFoundError(f"No image found for camera {cam_id} in {image_dir}")


def distort_points(x, y, distortion):
    k1, k2, p1, p2, k3 = distortion
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_distorted, y_distorted


def project_points(points_vehicle, ego_to_cam, k, distortion, width, height, use_distortion, rounding, min_depth):
    points_homo = np.concatenate(
        [points_vehicle, np.ones_like(points_vehicle[:, :1])],
        axis=1,
    )
    points_camera = points_homo @ ego_to_cam.T
    xyz = points_camera[:, :3]
    z = xyz[:, 2]
    valid = z > min_depth

    u = np.zeros(points_vehicle.shape[0], dtype=np.float64)
    v = np.zeros(points_vehicle.shape[0], dtype=np.float64)
    if np.any(valid):
        x = xyz[valid, 0] / z[valid]
        y = xyz[valid, 1] / z[valid]
        if use_distortion:
            x, y = distort_points(x, y, distortion)
        u[valid] = k[0, 0] * x + k[0, 2]
        v[valid] = k[1, 1] * y + k[1, 2]

    if rounding == "floor":
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
    else:
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)

    valid = valid & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    return valid, ui, vi


def regenerate_projection(args):
    pointcloud_path = args.pointcloud_path or os.path.join(args.data_root, "pointcloud.npz")
    pointcloud_npz = np.load(pointcloud_path, allow_pickle=True)
    pointcloud = pointcloud_npz["pointcloud"].item()

    frame_ids = sorted(pointcloud.keys())
    if args.start_frame is not None:
        frame_ids = [frame_id for frame_id in frame_ids if frame_id >= args.start_frame]
    if args.end_frame is not None:
        frame_ids = [frame_id for frame_id in frame_ids if frame_id <= args.end_frame]
    if not frame_ids:
        raise ValueError("No frames selected.")

    camera_list = args.camera_list
    intrinsics = {}
    distortions = {}
    ego_to_cams = {}
    sizes = {}
    for cam_id in camera_list:
        intrinsics[cam_id], distortions[cam_id] = load_intrinsics(args.data_root, cam_id)
        cam_to_ego = np.loadtxt(os.path.join(args.data_root, args.extrinsics_dir, f"{cam_id}.txt"))
        ego_to_cams[cam_id] = np.linalg.inv(cam_to_ego)
        sizes[cam_id] = image_size(args.data_root, cam_id, frame_ids)

    camera_projection = {}
    visibility_counts = {str(cam_id): 0 for cam_id in camera_list}
    no_projection_count = 0

    for frame_id in tqdm(frame_ids, desc="Regenerating camera projection"):
        points_vehicle = pointcloud[frame_id]
        projection = np.zeros((points_vehicle.shape[0], 6), dtype=np.int16)
        projection[:, 0] = UNKNOWN_CAMERA
        projection[:, 3] = UNKNOWN_CAMERA
        filled = np.zeros(points_vehicle.shape[0], dtype=np.int8)

        for cam_id in camera_list:
            width, height = sizes[cam_id]
            valid, ui, vi = project_points(
                points_vehicle=points_vehicle,
                ego_to_cam=ego_to_cams[cam_id],
                k=intrinsics[cam_id],
                distortion=distortions[cam_id],
                width=width,
                height=height,
                use_distortion=not args.no_distortion,
                rounding=args.rounding,
                min_depth=args.min_depth,
            )
            visibility_counts[str(cam_id)] += int(valid.sum())

            first_slot = valid & (filled == 0)
            projection[first_slot, 0] = cam_id
            projection[first_slot, 1] = ui[first_slot]
            projection[first_slot, 2] = vi[first_slot]
            filled[first_slot] = 1

            second_slot = valid & (filled == 1)
            projection[second_slot, 3] = cam_id
            projection[second_slot, 4] = ui[second_slot]
            projection[second_slot, 5] = vi[second_slot]
            filled[second_slot] = 2

        no_projection_count += int((filled == 0).sum())
        camera_projection[frame_id] = projection

    os.makedirs(args.output_dir, exist_ok=True)
    projection_path = os.path.join(args.output_dir, args.output_name)
    np.savez_compressed(projection_path, camera_projection=camera_projection)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": os.path.abspath(args.data_root),
        "pointcloud_path": os.path.abspath(pointcloud_path),
        "extrinsics_dir": args.extrinsics_dir,
        "camera_list": camera_list,
        "start_frame": frame_ids[0],
        "end_frame": frame_ids[-1],
        "frame_count": len(frame_ids),
        "use_distortion": not args.no_distortion,
        "rounding": args.rounding,
        "min_depth": args.min_depth,
        "visibility_counts": visibility_counts,
        "no_projection_count": no_projection_count,
        "projection_path": os.path.abspath(projection_path),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if args.write_compatible_pointcloud:
        compatible_path = os.path.join(args.output_dir, args.compatible_output_name)
        np.savez_compressed(
            compatible_path,
            pointcloud=pointcloud,
            camera_projection=camera_projection,
        )
        metadata["compatible_pointcloud_path"] = os.path.abspath(compatible_path)
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return projection_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate Waymo LiDAR camera_projection from current intrinsics/extrinsics."
    )
    parser.add_argument("--data_root", required=True, help="Converted Waymo scene directory.")
    parser.add_argument("--output_dir", required=True, help="Directory for regenerated projection files.")
    parser.add_argument("--pointcloud_path", default=None, help="Optional pointcloud.npz path. Defaults to data_root/pointcloud.npz.")
    parser.add_argument("--extrinsics_dir", default="extrinsics_fixed", help="Extrinsics directory inside data_root.")
    parser.add_argument("--camera-list", dest="camera_list", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--start-frame", dest="start_frame", type=int, default=None)
    parser.add_argument("--end-frame", dest="end_frame", type=int, default=None, help="Inclusive end frame.")
    parser.add_argument("--output-name", default="camera_projection.npz")
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--rounding", choices=["round", "floor"], default="round")
    parser.add_argument("--no-distortion", action="store_true", help="Project with pinhole intrinsics only.")
    parser.add_argument(
        "--write-compatible-pointcloud",
        action="store_true",
        help="Also write output_dir/pointcloud.npz with original pointcloud and regenerated projection.",
    )
    parser.add_argument(
        "--compatible-output-name",
        default="pointcloud.npz",
        help="Filename used with --write-compatible-pointcloud.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    output_path = regenerate_projection(parse_args())
    print(f"Saved regenerated camera projection to {output_path}")
