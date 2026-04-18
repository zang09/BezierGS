#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import glob
import os
import subprocess
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from types import SimpleNamespace
from tqdm import tqdm
from torchvision.utils import save_image
from omegaconf import OmegaConf
from gaussian_renderer import render
from scene import Scene, GaussianModel, EnvLight, ColorCorrection, PoseCorrection
from utils.general_utils import seed_everything


def _normalize_quaternion(q):
    return q / q.norm().clamp_min(1e-8)


def _matrix_to_quaternion(R):
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = torch.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = torch.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = torch.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return _normalize_quaternion(torch.stack([qw, qx, qy, qz]))


def _quaternion_to_matrix(q):
    q = _normalize_quaternion(q)
    w, x, y, z = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return torch.stack([
        torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)]),
        torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)]),
        torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)]),
    ])


def _slerp(q0, q1, t):
    q0 = _normalize_quaternion(q0)
    q1 = _normalize_quaternion(q1)
    dot = torch.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = torch.clamp(dot, -1.0, 1.0)
    if dot > 0.9995:
        return _normalize_quaternion((1.0 - t) * q0 + t * q1)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp_min(1e-8)
    theta = theta_0 * t

    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    return _normalize_quaternion(s0 * q0 + s1 * q1)


def _build_render_camera(ref_cam, c2w, timestamp):
    world_view_transform = torch.linalg.inv(c2w).transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(ref_cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    render_cam = SimpleNamespace()
    render_cam.FoVx = ref_cam.FoVx
    render_cam.FoVy = ref_cam.FoVy
    render_cam.image_height = ref_cam.image_height
    render_cam.image_width = ref_cam.image_width
    render_cam.projection_matrix = ref_cam.projection_matrix
    render_cam.world_view_transform = world_view_transform
    render_cam.full_proj_transform = full_proj_transform
    render_cam.camera_center = camera_center
    render_cam.c2w = c2w
    render_cam.timestamp = timestamp
    # Keep camera metadata so correction modules can index embeddings.
    render_cam.colmap_id = int(ref_cam.colmap_id)
    render_cam.uid = int(getattr(ref_cam, "uid", 0))
    render_cam.id = int(getattr(ref_cam, "id", render_cam.uid))
    render_cam.image_name = getattr(ref_cam, "image_name", "")
    render_cam.cx = ref_cam.cx
    render_cam.cy = ref_cam.cy
    render_cam.fx = ref_cam.fx
    render_cam.fy = ref_cam.fy
    render_cam.grid = ref_cam.grid

    def get_world_directions(train=False):
        u, v = render_cam.grid.unbind(-1)
        if train:
            directions = torch.stack(
                [
                    (u - render_cam.cx + torch.rand_like(u)) / render_cam.fx,
                    (v - render_cam.cy + torch.rand_like(v)) / render_cam.fy,
                    torch.ones_like(u),
                ],
                dim=0,
            )
        else:
            directions = torch.stack(
                [
                    (u - render_cam.cx + 0.5) / render_cam.fx,
                    (v - render_cam.cy + 0.5) / render_cam.fy,
                    torch.ones_like(u),
                ],
                dim=0,
            )
        directions = F.normalize(directions, dim=0)
        return (render_cam.c2w[:3, :3] @ directions.reshape(3, -1)).reshape(
            3, render_cam.image_height, render_cam.image_width
        )

    render_cam.get_world_directions = get_world_directions
    return render_cam


def _interpolate_camera(cam0, cam1, t):
    c2w0 = cam0.c2w.detach()
    c2w1 = cam1.c2w.detach()

    r0 = c2w0[:3, :3]
    r1 = c2w1[:3, :3]
    t0 = c2w0[:3, 3]
    t1 = c2w1[:3, 3]

    q0 = _matrix_to_quaternion(r0)
    q1 = _matrix_to_quaternion(r1)
    q = _slerp(q0, q1, t)
    r = _quaternion_to_matrix(q)
    trans = (1.0 - t) * t0 + t * t1
    ts = (1.0 - t) * cam0.timestamp + t * cam1.timestamp

    c2w = torch.eye(4, device=c2w0.device, dtype=c2w0.dtype)
    c2w[:3, :3] = r
    c2w[:3, 3] = trans
    return _build_render_camera(cam0, c2w, ts)


def _select_cameras(scene, split, scale, camera_id=None, camera_id_mod=10):
    if split == "test":
        cameras = list(scene.getTestCameras(scale=scale))
    elif split == "train":
        cameras = list(scene.getTrainCameras())
    else:
        raise ValueError(f"Unsupported split: {split}")

    cameras = sorted(cameras, key=lambda x: x.colmap_id)
    if camera_id is not None:
        cameras = [cam for cam in cameras if (cam.colmap_id % camera_id_mod) == camera_id]
    return cameras


@torch.no_grad()
def render_interpolated_video(
    iteration,
    scene: Scene,
    renderFunc,
    renderArgs,
    split="test",
    camera_id=None,
    camera_id_mod=10,
    interp_steps=8,
    fps=24,
    env_map=None,
    color_correction=None,
    pose_correction=None,
):
    scale = scene.resolution_scales[0]
    cameras = _select_cameras(scene, split, scale, camera_id=camera_id, camera_id_mod=camera_id_mod)
    if len(cameras) < 2:
        raise RuntimeError(f"Need at least 2 cameras to interpolate, got {len(cameras)} for split={split}.")

    camera_suffix = f"cam{camera_id}" if camera_id is not None else "allcams"
    outdir = os.path.join(scene.model_path, "render", f"{split}_{camera_suffix}_{iteration}")
    os.makedirs(outdir, exist_ok=True)
    video_path = os.path.join(outdir, "interpolated.mp4")

    frame_index = 0
    total_frames = (len(cameras) - 1) * interp_steps + 1
    progress = tqdm(total=total_frames, desc="Interpolated rendering")

    for i in range(len(cameras) - 1):
        cam0, cam1 = cameras[i], cameras[i + 1]
        for step in range(interp_steps):
            t = step / float(interp_steps)
            interp_cam = _interpolate_camera(cam0, cam1, t)
            v, _ = scene.gaussians.get_inst_velocity(interp_cam.timestamp)
            render_pkg = renderFunc(
                interp_cam,
                scene.gaussians,
                *renderArgs,
                env_map=env_map,
                color_correction=color_correction,
                pose_correction=pose_correction,
                other=[v],
            )
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            save_image(image, os.path.join(outdir, f"{frame_index:06d}.png"))
            frame_index += 1
            progress.update(1)

    last_cam = cameras[-1]
    v, _ = scene.gaussians.get_inst_velocity(last_cam.timestamp)
    render_pkg = renderFunc(
        last_cam,
        scene.gaussians,
        *renderArgs,
        env_map=env_map,
        color_correction=color_correction,
        pose_correction=pose_correction,
        other=[v],
    )
    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
    save_image(image, os.path.join(outdir, f"{frame_index:06d}.png"))
    progress.update(1)
    progress.close()

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(outdir, "%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        video_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video saved: {video_path}")
        frame_paths = glob.glob(os.path.join(outdir, "*.png"))
        for frame_path in frame_paths:
            os.remove(frame_path)
        print(f"Deleted {len(frame_paths)} frame images from {outdir}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ffmpeg failed or is unavailable. Frames are saved as PNG in:")
        print(outdir)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default = "configs/base.yaml")
    args, _ = parser.parse_known_args()

    default_render_conf = OmegaConf.create(
        {
            "render_split": "test",
            "render_camera_id": None,
            "render_camera_id_mod": 10,
            "render_interp_steps": 10,
            "render_fps": 30,
        }
    )

    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli()
    args = OmegaConf.merge(default_render_conf, base_conf, second_conf, cli_conf)
    args.resolution_scales = args.resolution_scales[:1]
    print(args)

    seed_everything(args.seed)

    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)

    if args.env_map_res > 0:
        env_map = EnvLight(resolution=args.env_map_res).cuda()
        env_map.training_setup(args)
    else:
        env_map = None

    if args.use_color_correction:
        color_correction = ColorCorrection(args)
        color_correction.training_setup(args)
    else:
        color_correction = None

    if args.use_pose_correction:
        pose_correction = PoseCorrection(args)
        pose_correction.training_setup(args)
    else:
        pose_correction = None

    checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    assert len(checkpoints) > 0, "No checkpoints found."
    checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, args)

    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(checkpoint),
                                    os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
        (light_params, _) = torch.load(env_checkpoint)
        env_map.restore(light_params)
    if color_correction is not None:
        color_correction_checkpoint = os.path.join(
            os.path.dirname(checkpoint),
            os.path.basename(checkpoint).replace("chkpnt", "color_correction_chkpnt"),
        )
        (color_correction_params, _) = torch.load(color_correction_checkpoint)
        color_correction.restore(color_correction_params)
    if pose_correction is not None:
        pose_correction_checkpoint = os.path.join(
            os.path.dirname(checkpoint),
            os.path.basename(checkpoint).replace("chkpnt", "pose_correction_chkpnt"),
        )
        (pose_correction_params, _) = torch.load(pose_correction_checkpoint)
        pose_correction.restore(pose_correction_params)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    render_interpolated_video(
        first_iter,
        scene,
        render,
        (args, background),
        split=args.render_split,
        camera_id=args.render_camera_id,
        camera_id_mod=args.render_camera_id_mod,
        interp_steps=args.render_interp_steps,
        fps=args.render_fps,
        env_map=env_map,
        color_correction=color_correction,
        pose_correction=pose_correction,
    )
    print("Interpolated rendering complete.")
