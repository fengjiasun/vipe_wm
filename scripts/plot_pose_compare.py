import argparse
from pathlib import Path

import numpy as np


def _load_pose(path: Path) -> np.ndarray:
    poses = np.load(path)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Invalid pose shape {poses.shape} in {path}")
    return poses.astype(np.float64, copy=False)


def _umeyama_alignment(x: np.ndarray, y: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    n = x.shape[0]
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    x0 = x - mean_x
    y0 = y - mean_y
    cov = (y0.T @ x0) / max(n, 1)
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    if with_scale:
        var_y = (y0**2).sum() / max(n, 1)
        scale = float(s.sum() / max(var_y, 1e-12))
    else:
        scale = 1.0
    t = mean_x - scale * (r @ mean_y)
    return scale, r, t


def _apply_transform(poses: np.ndarray, scale: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = poses.copy()
    out[:, :3, 3] = (scale * (r @ out[:, :3, 3].T)).T + t[None, :]
    out[:, :3, :3] = np.einsum("ij,njk->nik", r, out[:, :3, :3])
    return out


def _find_pose_path(output_dir: Path, video: str) -> Path:
    pose_dir = output_dir / "pose"
    if video.endswith(".npy"):
        name = video
    elif video.endswith(".mp4"):
        name = f"{video}.npy"
    else:
        name = f"{video}.npy"
    matches = list(pose_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"No pose file '{name}' under {pose_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple pose files found for '{name}': {matches[:5]}")
    return matches[0]


def _plot_xy_xz_yz(t_a: np.ndarray, t_b: np.ndarray, label_a: str, label_b: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    planes = [("X", "Y", (0, 1)), ("X", "Z", (0, 2)), ("Y", "Z", (1, 2))]
    for ax, (lx, ly, (i, j)) in zip(axes, planes):
        ax.plot(t_a[:, i], t_a[:, j], label=label_a, linewidth=1.5)
        ax.plot(t_b[:, i], t_b[:, j], label=label_b, linewidth=1.5)
        ax.scatter(t_a[0, i], t_a[0, j], s=20, marker="o")
        ax.scatter(t_b[0, i], t_b[0, j], s=20, marker="o")
        ax.scatter(t_a[-1, i], t_a[-1, j], s=20, marker="x")
        ax.scatter(t_b[-1, i], t_b[-1, j], s=20, marker="x")
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_3d(
    t_a: np.ndarray,
    t_b: np.ndarray,
    label_a: str,
    label_b: str,
    out: Path,
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(t_a[:, 0], t_a[:, 1], t_a[:, 2], label=label_a, linewidth=1.5)
    ax.plot(t_b[:, 0], t_b[:, 1], t_b[:, 2], label=label_b, linewidth=1.5)
    ax.scatter(t_a[0, 0], t_a[0, 1], t_a[0, 2], s=20, marker="o")
    ax.scatter(t_b[0, 0], t_b[0, 1], t_b[0, 2], s=20, marker="o")
    ax.scatter(t_a[-1, 0], t_a[-1, 1], t_a[-1, 2], s=20, marker="x")
    ax.scatter(t_b[-1, 0], t_b[-1, 1], t_b[-1, 2], s=20, marker="x")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    mins = np.minimum(t_a.min(axis=0), t_b.min(axis=0))
    maxs = np.maximum(t_a.max(axis=0), t_b.max(axis=0))
    span = np.maximum(maxs - mins, 1e-6)
    center = (maxs + mins) / 2.0
    ax.set_box_aspect(span.tolist())
    ax.set_xlim(center[0] - span[0] / 2, center[0] + span[0] / 2)
    ax.set_ylim(center[1] - span[1] / 2, center[1] + span[1] / 2)
    ax.set_zlim(center[2] - span[2] / 2, center[2] + span[2] / 2)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

def _rotation_angle(r: np.ndarray) -> float:
    trace = np.trace(r)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def _plot_error_curves(
    poses_a: np.ndarray,
    poses_b: np.ndarray,
    label_a: str,
    label_b: str,
    out: Path,
) -> None:
    import matplotlib.pyplot as plt

    t_a = poses_a[:, :3, 3]
    t_b = poses_b[:, :3, 3]
    trans_err = np.linalg.norm(t_a - t_b, axis=1)

    r_a = poses_a[:, :3, :3]
    r_b = poses_b[:, :3, :3]
    rot_err = []
    for i in range(r_a.shape[0]):
        r_err = r_b[i].T @ r_a[i]
        rot_err.append(_rotation_angle(r_err))
    rot_err = np.array(rot_err, dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(trans_err, linewidth=1.5)
    axes[0].set_ylabel("translation error")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(rot_err, linewidth=1.5, color="tab:orange")
    axes[1].set_ylabel("rotation error (rad)")
    axes[1].set_xlabel("frame index")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"{label_a} vs {label_b} per-frame errors")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _mat_to_quat(r: np.ndarray) -> np.ndarray:
    m00, m01, m02 = r[0, 0], r[0, 1], r[0, 2]
    m10, m11, m12 = r[1, 0], r[1, 1], r[1, 2]
    m20, m21, m22 = r[2, 0], r[2, 1], r[2, 2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    cos_theta = float(np.dot(q0, q1))
    if cos_theta < 0.0:
        q1 = -q1
        cos_theta = -cos_theta
    if cos_theta > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * q0 + b * q1


def _resample_poses(poses: np.ndarray, n_target: int) -> np.ndarray:
    if n_target <= 1 or poses.shape[0] == n_target:
        return poses
    n = poses.shape[0]
    t_src = np.linspace(0.0, 1.0, n)
    t_dst = np.linspace(0.0, 1.0, n_target)
    out = np.zeros((n_target, 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1.0

    trans = poses[:, :3, 3]
    rots = poses[:, :3, :3]
    quats = np.array([_mat_to_quat(r) for r in rots], dtype=np.float64)

    for i, td in enumerate(t_dst):
        idx = np.searchsorted(t_src, td, side="right") - 1
        idx = int(np.clip(idx, 0, n - 2))
        t0, t1 = t_src[idx], t_src[idx + 1]
        alpha = 0.0 if t1 == t0 else (td - t0) / (t1 - t0)
        out[i, :3, 3] = (1.0 - alpha) * trans[idx] + alpha * trans[idx + 1]
        q = _slerp(quats[idx], quats[idx + 1], float(alpha))
        out[i, :3, :3] = _quat_to_mat(q)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot two pose trajectories together.")
    parser.add_argument("--pose-a", type=Path, help="Pose .npy path for run A.")
    parser.add_argument("--pose-b", type=Path, help="Pose .npy path for run B.")
    parser.add_argument("--output-a", type=Path, help="Output dir A (contains pose/).")
    parser.add_argument("--output-b", type=Path, help="Output dir B (contains pose/).")
    parser.add_argument("--video", help="Video filename to locate pose (e.g. xxx.mp4 or xxx.mp4.npy).")
    parser.add_argument("--align", choices=["none", "se3", "sim3"], default="se3", help="Alignment mode.")
    parser.add_argument("--length", choices=["min", "strict"], default="min", help="Handle length mismatch.")
    parser.add_argument("--interp", action="store_true", help="Resample both trajectories to same length.")
    parser.add_argument("--interp-length", type=int, default=0, help="Target length for interpolation.")
    parser.add_argument("--label-a", default="A", help="Legend label for A.")
    parser.add_argument("--label-b", default="B", help="Legend label for B.")
    parser.add_argument("--out", type=Path, default=Path("pose_compare.png"), help="Output image path.")
    parser.add_argument("--out-3d", type=Path, help="Optional output path for 3D plot.")
    parser.add_argument("--error-out", type=Path, help="Optional output path for per-frame error plot.")
    args = parser.parse_args()

    if args.pose_a and args.pose_b:
        pose_a_path = args.pose_a
        pose_b_path = args.pose_b
    elif args.output_a and args.output_b and args.video:
        pose_a_path = _find_pose_path(args.output_a, args.video)
        pose_b_path = _find_pose_path(args.output_b, args.video)
    else:
        raise ValueError("Provide either --pose-a/--pose-b or --output-a/--output-b with --video.")

    poses_a = _load_pose(pose_a_path)
    poses_b = _load_pose(pose_b_path)
    n_a, n_b = poses_a.shape[0], poses_b.shape[0]
    if args.interp:
        n_target = args.interp_length if args.interp_length > 0 else max(n_a, n_b)
        poses_a = _resample_poses(poses_a, n_target)
        poses_b = _resample_poses(poses_b, n_target)
    else:
        if args.length == "strict" and n_a != n_b:
            raise ValueError(f"Length mismatch: {n_a} vs {n_b}")
        n = min(n_a, n_b)
        poses_a = poses_a[:n]
        poses_b = poses_b[:n]

    if args.align != "none":
        t_a = poses_a[:, :3, 3]
        t_b = poses_b[:, :3, 3]
        scale, r, t = _umeyama_alignment(t_a, t_b, with_scale=(args.align == "sim3"))
        poses_b = _apply_transform(poses_b, scale, r, t)

    t_a = poses_a[:, :3, 3]
    t_b = poses_b[:, :3, 3]
    _plot_xy_xz_yz(t_a, t_b, args.label_a, args.label_b, args.out)
    if args.out_3d is not None:
        _plot_3d(t_a, t_b, args.label_a, args.label_b, args.out_3d)
    if args.error_out is not None:
        _plot_error_curves(poses_a, poses_b, args.label_a, args.label_b, args.error_out)
    print(f"Saved: {args.out}")
    if args.out_3d is not None:
        print(f"Saved: {args.out_3d}")


if __name__ == "__main__":
    main()
