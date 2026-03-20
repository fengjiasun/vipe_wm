import argparse
import os
from pathlib import Path

import importlib
import imageio
import numpy as np
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize camera trajectory as a video.")
    parser.add_argument("pose_npy", type=Path, help="Path to pose.npy (N,4,4) c2w matrices.")
    parser.add_argument(
        "--intrinsics-npy",
        type=Path,
        default=None,
        help="Optional intrinsics.npy (N,4) or (4,) for frustum drawing.",
    )
    parser.add_argument("--image-width", type=int, default=None, help="Image width for frustum projection.")
    parser.add_argument("--image-height", type=int, default=None, help="Image height for frustum projection.")
    parser.add_argument("--output", "-o", type=Path, default=Path("trajectory.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--plane", choices=["xy", "xz", "yz", "xyz"], default="xz")
    parser.add_argument(
        "--view",
        choices=["default", "front", "back", "left", "right", "top", "bottom", "iso"],
        default="default",
        help="Preset view for 3D (only for --plane xyz).",
    )
    parser.add_argument("--view-elev", type=float, default=None, help="Custom elev for 3D view.")
    parser.add_argument("--view-azim", type=float, default=None, help="Custom azim for 3D view.")
    parser.add_argument(
        "--view-from-frame",
        type=int,
        default=None,
        help="Use this frame's forward direction as view (3D only).",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for visualization.")
    parser.add_argument("--tail", type=int, default=0, help="Tail length (0 means show full history).")
    parser.add_argument("--arrow-scale", type=float, default=0.05, help="Arrow length as fraction of scene size.")
    parser.add_argument("--frustum-step", type=int, default=10, help="Draw frustum every N frames.")
    parser.add_argument(
        "--frustum-depth",
        type=float,
        default=0.15,
        help="Frustum depth as fraction of scene size.",
    )
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--figsize", type=float, nargs=2, default=(6.4, 4.8))
    parser.add_argument(
        "--backend",
        choices=["auto", "ffmpeg", "opencv"],
        default="auto",
        help="Video writer backend. 'auto' tries opencv on Windows, else ffmpeg.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N frames (0 disables).",
    )
    return parser.parse_args()


def _project(vec: np.ndarray, plane: str) -> np.ndarray:
    if plane == "xy":
        return vec[[0, 1]]
    if plane == "xz":
        return vec[[0, 2]]
    if plane == "yz":
        return vec[[1, 2]]
    return vec


def _axis_labels(plane: str) -> tuple[str, str]:
    if plane == "xy":
        return "X", "Y"
    if plane == "xz":
        return "X", "Z"
    if plane == "yz":
        return "Y", "Z"
    return "X", "Y"


def _get_bounds(points_2d: np.ndarray, padding: float = 0.05) -> tuple[float, float, float, float]:
    x_min, y_min = points_2d.min(axis=0)
    x_max, y_max = points_2d.max(axis=0)
    dx = x_max - x_min
    dy = y_max - y_min
    pad_x = dx * padding if dx > 0 else 1.0
    pad_y = dy * padding if dy > 0 else 1.0
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def _load_intrinsics(intrinsics_path: Path, num_frames: int) -> np.ndarray:
    intr = np.load(intrinsics_path)
    if intr.ndim == 1:
        intr = intr[None, :]
    if intr.shape[0] == 1 and num_frames > 1:
        intr = np.repeat(intr, num_frames, axis=0)
    return intr


def _draw_frustum(
    ax,
    pose: np.ndarray,
    intr: np.ndarray,
    image_size: tuple[int, int],
    depth: float,
    color: str = "k",
    alpha: float = 0.6,
) -> None:
    fx, fy, cx, cy = intr[:4]
    w, h = image_size
    corners_uv = np.array(
        [
            [0.0, 0.0],
            [w, 0.0],
            [w, h],
            [0.0, h],
        ],
        dtype=np.float64,
    )
    corners_cam = np.zeros((4, 3), dtype=np.float64)
    corners_cam[:, 2] = depth
    corners_cam[:, 0] = (corners_uv[:, 0] - cx) / fx * depth
    corners_cam[:, 1] = (corners_uv[:, 1] - cy) / fy * depth

    R = pose[:3, :3]
    t = pose[:3, 3]
    origin = t
    corners_world = (R @ corners_cam.T).T + t[None, :]

    # Draw edges
    for i in range(4):
        j = (i + 1) % 4
        ax.plot(
            [corners_world[i, 0], corners_world[j, 0]],
            [corners_world[i, 1], corners_world[j, 1]],
            [corners_world[i, 2], corners_world[j, 2]],
            color=color,
            alpha=alpha,
            linewidth=0.8,
        )
        ax.plot(
            [origin[0], corners_world[i, 0]],
            [origin[1], corners_world[i, 1]],
            [origin[2], corners_world[i, 2]],
            color=color,
            alpha=alpha,
            linewidth=0.8,
        )


def _preset_view(view: str) -> tuple[float, float]:
    presets = {
        "default": (30.0, -60.0),
        "front": (0.0, 90.0),
        "back": (0.0, -90.0),
        "left": (0.0, 0.0),
        "right": (0.0, 180.0),
        "top": (90.0, -90.0),
        "bottom": (-90.0, -90.0),
        "iso": (30.0, 45.0),
    }
    return presets.get(view, presets["default"])


def _view_from_forward(forward: np.ndarray) -> tuple[float, float]:
    f = _normalize(forward)
    xy_len = np.linalg.norm(f[:2])
    elev = np.degrees(np.arctan2(f[2], xy_len))
    azim = np.degrees(np.arctan2(f[1], f[0]))
    return elev, azim


def _opencv_available() -> bool:
    return importlib.util.find_spec("cv2") is not None


def _open_writer(args: argparse.Namespace, fig):
    backend = args.backend
    if backend == "auto":
        if os.name == "nt" and _opencv_available():
            backend = "opencv"
        else:
            backend = "ffmpeg"

    if backend == "opencv":
        import cv2  # type: ignore

        width, height = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV VideoWriter failed to open output.")
        return writer, True

    # ffmpeg backend
    try:
        writer = imageio.get_writer(
            str(args.output),
            fps=args.fps,
            format="ffmpeg",
            codec="libx264",
        )
        return writer, False
    except Exception as exc:
        raise RuntimeError(
            "Failed to open ffmpeg writer. Install imageio-ffmpeg or use --backend opencv."
        ) from exc


def _write_frame(writer, writer_is_cv2: bool, img: np.ndarray) -> None:
    if writer_is_cv2:
        writer.write(img[:, :, ::-1])
    else:
        writer.append_data(img)


def _close_writer(writer, writer_is_cv2: bool) -> None:
    if writer_is_cv2:
        writer.release()
    else:
        writer.close()


def main() -> None:
    args = _parse_args()
    poses = np.load(args.pose_npy)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("pose.npy must be of shape (N,4,4).")

    poses = poses[:: args.stride]
    positions = poses[:, :3, 3]
    intrinsics = None
    if args.intrinsics_npy is not None:
        if args.plane != "xyz":
            raise ValueError("Frustum drawing requires --plane xyz.")
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height are required for frustum drawing.")
        intrinsics = _load_intrinsics(args.intrinsics_npy, len(poses))

    if args.plane == "xyz":
        # 3D view
        fig = plt.figure(figsize=tuple(args.figsize))
        ax = fig.add_subplot(111, projection="3d")
        all_pos = positions
        x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
        y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
        z_min, z_max = all_pos[:, 2].min(), all_pos[:, 2].max()
        padding = 0.05
        dx = x_max - x_min
        dy = y_max - y_min
        dz = z_max - z_min
        ax.set_xlim(x_min - dx * padding, x_max + dx * padding)
        ax.set_ylim(y_min - dy * padding, y_max + dy * padding)
        ax.set_zlim(z_min - dz * padding, z_max + dz * padding)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        elev, azim = _preset_view(args.view)
        if args.view_from_frame is not None:
            frame_idx = int(np.clip(args.view_from_frame, 0, len(poses) - 1))
            elev, azim = _view_from_forward(poses[frame_idx, :3, 2])
        if args.view_elev is not None:
            elev = args.view_elev
        if args.view_azim is not None:
            azim = args.view_azim
        ax.view_init(elev=elev, azim=azim)
        scene_size = max(dx, dy, dz, 1.0)
    else:
        # 2D plane view
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
        proj_positions = np.stack([_project(p, args.plane) for p in positions], axis=0)
        x_min, x_max, y_min, y_max = _get_bounds(proj_positions)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        x_label, y_label = _axis_labels(args.plane)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        scene_size = max(x_max - x_min, y_max - y_min, 1.0)

    writer, writer_is_cv2 = _open_writer(args, fig)

    for i in range(len(poses)):
        ax.cla()
        if args.plane == "xyz":
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(x_min - dx * 0.05, x_max + dx * 0.05)
            ax.set_ylim(y_min - dy * 0.05, y_max + dy * 0.05)
            ax.set_zlim(z_min - dz * 0.05, z_max + dz * 0.05)
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            x_label, y_label = _axis_labels(args.plane)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        start = 0
        if args.tail and args.tail > 0:
            start = max(0, i - args.tail + 1)

        if args.plane == "xyz":
            traj = positions[start : i + 1]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1.5)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c="r", s=15)
            forward = poses[i, :3, 2]
            fwd = _normalize(forward) * args.arrow_scale * scene_size
            ax.quiver(
                traj[-1, 0],
                traj[-1, 1],
                traj[-1, 2],
                fwd[0],
                fwd[1],
                fwd[2],
                color="r",
                length=1.0,
                normalize=False,
            )
            if intrinsics is not None and args.frustum_step > 0 and i % args.frustum_step == 0:
                frustum_depth = args.frustum_depth * scene_size
                intr = intrinsics[min(i, intrinsics.shape[0] - 1)]
                _draw_frustum(
                    ax,
                    poses[i],
                    intr,
                    (args.image_width, args.image_height),
                    frustum_depth,
                )
        else:
            traj = np.stack([_project(p, args.plane) for p in positions[start : i + 1]], axis=0)
            ax.plot(traj[:, 0], traj[:, 1], lw=1.5)
            ax.scatter(traj[-1, 0], traj[-1, 1], c="r", s=15)
            forward = poses[i, :3, 2]
            fwd = _normalize(_project(forward, args.plane)) * args.arrow_scale * scene_size
            ax.arrow(
                traj[-1, 0],
                traj[-1, 1],
                fwd[0],
                fwd[1],
                head_width=0.02 * scene_size,
                head_length=0.03 * scene_size,
                color="r",
                length_includes_head=True,
            )

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        _write_frame(writer, writer_is_cv2, img)
        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            print(f"[trajectory] {i + 1}/{len(poses)} frames")

    _close_writer(writer, writer_is_cv2)
    plt.close(fig)


if __name__ == "__main__":
    main()
