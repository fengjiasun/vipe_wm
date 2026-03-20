import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np


def _collect_npy_files(base_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in base_dir.rglob("*.npy"):
        if path.is_file():
            rel = str(path.relative_to(base_dir))
            files[rel] = path
    return files


def _load_pose(path: Path) -> np.ndarray:
    poses = np.load(path)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Invalid pose shape {poses.shape} in {path}")
    return poses.astype(np.float64, copy=False)


def _normalize_intrinsics(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[None, :]
    else:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float64, copy=False)


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


def _rotation_angle(r: np.ndarray) -> float:
    trace = np.trace(r)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def _pose_stats(poses_a: np.ndarray, poses_b: np.ndarray) -> dict[str, float]:
    t_a = poses_a[:, :3, 3]
    t_b = poses_b[:, :3, 3]
    err = t_a - t_b
    dist = np.linalg.norm(err, axis=1)
    return {
        "trans_rmse": float(np.sqrt(np.mean(dist**2))),
        "trans_mean": float(np.mean(dist)),
        "trans_median": float(np.median(dist)),
        "trans_p95": float(np.percentile(dist, 95)),
        "trans_max": float(np.max(dist)),
    }


def _rotation_stats(poses_a: np.ndarray, poses_b: np.ndarray) -> dict[str, float]:
    r_a = poses_a[:, :3, :3]
    r_b = poses_b[:, :3, :3]
    angles = []
    for i in range(r_a.shape[0]):
        r_err = r_b[i].T @ r_a[i]
        angles.append(_rotation_angle(r_err))
    angles = np.array(angles, dtype=np.float64)
    return {
        "rot_mean_rad": float(np.mean(angles)),
        "rot_median_rad": float(np.median(angles)),
        "rot_p95_rad": float(np.percentile(angles, 95)),
        "rot_max_rad": float(np.max(angles)),
    }


def _compare_pose_pair(
    path_a: Path,
    path_b: Path,
    align: str,
    length_mode: str,
) -> dict[str, float]:
    poses_a = _load_pose(path_a)
    poses_b = _load_pose(path_b)
    n_a = poses_a.shape[0]
    n_b = poses_b.shape[0]
    if length_mode == "strict" and n_a != n_b:
        raise ValueError(f"Length mismatch: {path_a} ({n_a}) vs {path_b} ({n_b})")
    n = min(n_a, n_b)
    poses_a = poses_a[:n]
    poses_b = poses_b[:n]
    t_a = poses_a[:, :3, 3]
    t_b = poses_b[:, :3, 3]

    if align != "none":
        with_scale = align == "sim3"
        scale, r, t = _umeyama_alignment(t_a, t_b, with_scale)
        t_b = (scale * (r @ t_b.T)).T + t[None, :]
        r_b = poses_b[:, :3, :3]
        r_b = np.einsum("ij,njk->nik", r, r_b)
        poses_b = poses_b.copy()
        poses_b[:, :3, :3] = r_b
        poses_b[:, :3, 3] = t_b

    stats = _pose_stats(poses_a, poses_b)
    stats.update(_rotation_stats(poses_a, poses_b))
    stats["frames_compared"] = float(n)
    return stats


def _compare_intr_pair(
    path_a: Path,
    path_b: Path,
    length_mode: str,
) -> dict[str, float]:
    intr_a = _normalize_intrinsics(np.load(path_a))
    intr_b = _normalize_intrinsics(np.load(path_b))
    if intr_a.shape[1] != intr_b.shape[1]:
        raise ValueError(f"Dim mismatch: {path_a} ({intr_a.shape[1]}) vs {path_b} ({intr_b.shape[1]})")

    if intr_a.shape[0] == 1 and intr_b.shape[0] > 1:
        intr_a = np.repeat(intr_a, intr_b.shape[0], axis=0)
    if intr_b.shape[0] == 1 and intr_a.shape[0] > 1:
        intr_b = np.repeat(intr_b, intr_a.shape[0], axis=0)

    n_a = intr_a.shape[0]
    n_b = intr_b.shape[0]
    if length_mode == "strict" and n_a != n_b:
        raise ValueError(f"Length mismatch: {path_a} ({n_a}) vs {path_b} ({n_b})")
    n = min(n_a, n_b)
    intr_a = intr_a[:n]
    intr_b = intr_b[:n]

    diff = intr_a - intr_b
    absdiff = np.abs(diff)
    per_param_mean = absdiff.mean(axis=0)
    per_param_max = absdiff.max(axis=0)
    return {
        "frames_compared": float(n),
        "mean_abs": float(absdiff.mean()),
        "max_abs": float(absdiff.max()),
        "per_param_mean_abs": per_param_mean.tolist(),
        "per_param_max_abs": per_param_max.tolist(),
    }


def _summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _parse_time_log(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    timing_re = re.compile(r"timing: wait_io=([0-9.]+)s pipeline=([0-9.]+)s save=([0-9.]+)s")
    overall_re = re.compile(r"Overall \d+/\d+ \| Elapsed (\d+):(\d+):(\d+)")
    wait_list: list[float] = []
    pipe_list: list[float] = []
    save_list: list[float] = []
    overall = 0.0

    for m in timing_re.finditer(text):
        wait_list.append(float(m.group(1)))
        pipe_list.append(float(m.group(2)))
        save_list.append(float(m.group(3)))
    for m in overall_re.finditer(text):
        h, mi, s = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        overall = max(overall, h * 3600 + mi * 60 + s)

    stats = {
        "timing_count": float(len(pipe_list)),
        "wait_mean": _summary_stats(wait_list)["mean"],
        "pipeline_mean": _summary_stats(pipe_list)["mean"],
        "save_mean": _summary_stats(save_list)["mean"],
        "overall_elapsed": float(overall),
    }
    return stats


def _format_path(base: Path | None, rel: str) -> str:
    if base is None:
        return rel
    return str((base / rel).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pose/intrinsics outputs between two runs.")
    parser.add_argument("--output-a", "-a", type=Path, required=True, help="Output dir A.")
    parser.add_argument("--output-b", "-b", type=Path, required=True, help="Output dir B.")
    parser.add_argument("--pose-subdir", default="pose", help="Pose subdir name (default: pose).")
    parser.add_argument("--intrinsics-subdir", default="intrinsics", help="Intrinsics subdir name.")
    parser.add_argument("--align", choices=["none", "se3", "sim3"], default="se3", help="Pose alignment mode.")
    parser.add_argument("--length", choices=["min", "strict"], default="min", help="How to handle length mismatch.")
    parser.add_argument("--pose-only", action="store_true", help="Only compare poses.")
    parser.add_argument("--intr-only", action="store_true", help="Only compare intrinsics.")
    parser.add_argument("--per-video", action="store_true", help="Print per-video stats.")
    parser.add_argument("--topk", type=int, default=10, help="Top-K videos to show by translation RMSE.")
    parser.add_argument("--time-log-a", type=Path, help="Optional log file for run A.")
    parser.add_argument("--time-log-b", type=Path, help="Optional log file for run B.")
    args = parser.parse_args()

    pose_dir_a = args.output_a / args.pose_subdir
    pose_dir_b = args.output_b / args.pose_subdir
    intr_dir_a = args.output_a / args.intrinsics_subdir
    intr_dir_b = args.output_b / args.intrinsics_subdir

    if not args.intr_only:
        pose_a = _collect_npy_files(pose_dir_a)
        pose_b = _collect_npy_files(pose_dir_b)
        common_pose = sorted(set(pose_a) & set(pose_b))
        missing_pose = sorted(set(pose_a) ^ set(pose_b))
        pose_stats = []

        for rel in common_pose:
            try:
                stats = _compare_pose_pair(pose_a[rel], pose_b[rel], args.align, args.length)
                stats["rel"] = rel
                pose_stats.append(stats)
            except Exception as exc:
                print(f"[WARN] pose compare failed for {rel}: {exc}")

        print("Pose comparison:")
        print(f"- common={len(common_pose)} missing={len(missing_pose)}")
        if missing_pose:
            print(f"- missing examples: {missing_pose[:5]}")
        if pose_stats:
            rmse_list = [s["trans_rmse"] for s in pose_stats]
            rot_list = [s["rot_mean_rad"] for s in pose_stats]
            print(
                f"- trans_rmse mean={_summary_stats(rmse_list)['mean']:.6g} "
                f"p95={_summary_stats(rmse_list)['p95']:.6g} "
                f"max={_summary_stats(rmse_list)['max']:.6g}"
            )
            print(
                f"- rot_mean_rad mean={_summary_stats(rot_list)['mean']:.6g} "
                f"p95={_summary_stats(rot_list)['p95']:.6g} "
                f"max={_summary_stats(rot_list)['max']:.6g}"
            )
            if args.per_video:
                for s in sorted(pose_stats, key=lambda x: x["trans_rmse"], reverse=True)[: args.topk]:
                    print(
                        f"- {s['rel']} | frames={int(s['frames_compared'])} "
                        f"trans_rmse={s['trans_rmse']:.6g} trans_p95={s['trans_p95']:.6g} "
                        f"rot_mean_rad={s['rot_mean_rad']:.6g} rot_p95_rad={s['rot_p95_rad']:.6g}"
                    )

    if not args.pose_only:
        intr_a = _collect_npy_files(intr_dir_a)
        intr_b = _collect_npy_files(intr_dir_b)
        common_intr = sorted(set(intr_a) & set(intr_b))
        missing_intr = sorted(set(intr_a) ^ set(intr_b))
        intr_stats = []

        for rel in common_intr:
            try:
                stats = _compare_intr_pair(intr_a[rel], intr_b[rel], args.length)
                stats["rel"] = rel
                intr_stats.append(stats)
            except Exception as exc:
                print(f"[WARN] intrinsics compare failed for {rel}: {exc}")

        print("\nIntrinsics comparison:")
        print(f"- common={len(common_intr)} missing={len(missing_intr)}")
        if missing_intr:
            print(f"- missing examples: {missing_intr[:5]}")
        if intr_stats:
            mean_abs_list = [s["mean_abs"] for s in intr_stats]
            max_abs_list = [s["max_abs"] for s in intr_stats]
            print(
                f"- mean_abs mean={_summary_stats(mean_abs_list)['mean']:.6g} "
                f"p95={_summary_stats(mean_abs_list)['p95']:.6g} "
                f"max={_summary_stats(mean_abs_list)['max']:.6g}"
            )
            print(
                f"- max_abs mean={_summary_stats(max_abs_list)['mean']:.6g} "
                f"p95={_summary_stats(max_abs_list)['p95']:.6g} "
                f"max={_summary_stats(max_abs_list)['max']:.6g}"
            )
            if args.per_video:
                for s in sorted(intr_stats, key=lambda x: x["max_abs"], reverse=True)[: args.topk]:
                    print(
                        f"- {s['rel']} | frames={int(s['frames_compared'])} "
                        f"mean_abs={s['mean_abs']:.6g} max_abs={s['max_abs']:.6g} "
                        f"per_param_mean_abs={np.round(s['per_param_mean_abs'], 6).tolist()}"
                    )

    if args.time_log_a and args.time_log_b:
        time_a = _parse_time_log(args.time_log_a)
        time_b = _parse_time_log(args.time_log_b)
        print("\nTiming comparison:")
        print(
            f"- A timing_count={int(time_a['timing_count'])} "
            f"wait_mean={time_a['wait_mean']:.6g} pipeline_mean={time_a['pipeline_mean']:.6g} "
            f"save_mean={time_a['save_mean']:.6g} overall_elapsed={time_a['overall_elapsed']:.6g}"
        )
        print(
            f"- B timing_count={int(time_b['timing_count'])} "
            f"wait_mean={time_b['wait_mean']:.6g} pipeline_mean={time_b['pipeline_mean']:.6g} "
            f"save_mean={time_b['save_mean']:.6g} overall_elapsed={time_b['overall_elapsed']:.6g}"
        )


if __name__ == "__main__":
    main()
