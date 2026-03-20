import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _iter_npy_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".npy":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.npy")))
    return files


def _resolve_targets(args: argparse.Namespace) -> tuple[list[Path], Path | None]:
    base_dir: Path | None = None
    targets: list[Path] = []
    if args.output_dir:
        base_dir = Path(args.output_dir) / args.mode
        targets.append(base_dir)
    elif args.path:
        path = Path(args.path)
        targets.append(path)
        if path.is_dir():
            base_dir = path
    else:
        raise ValueError("Please provide --output-dir or a path to .npy or folder.")
    return _iter_npy_files(targets), base_dir


def _normalize_intrinsics(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[None, :]
    else:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float64, copy=False)


def _summarize_per_video(
    path: Path,
    tol: float,
    collect_unique: bool = False,
    unique_max: int = 5,
    unique_decimals: int = 6,
) -> dict:
    data = np.load(path)
    frames = _normalize_intrinsics(data)
    n_frames, dim = frames.shape
    ref = frames[0]
    diff = np.abs(frames - ref[None, :])
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    std_mean = float(frames.std(axis=0).mean()) if n_frames > 1 else 0.0
    std_max = float(frames.std(axis=0).max()) if n_frames > 1 else 0.0
    result = {
        "path": str(path),
        "frames": int(n_frames),
        "dim": int(dim),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "std_mean": std_mean,
        "std_max": std_max,
        "is_constant": max_abs <= tol,
        "mean_vec": frames.mean(axis=0).tolist(),
    }
    if collect_unique:
        rounded = np.round(frames, decimals=unique_decimals)
        uniq, counts = np.unique(rounded, axis=0, return_counts=True)
        order = np.argsort(-counts)
        if unique_max > 0:
            order = order[:unique_max]
        unique_values = [
            {"value": uniq[int(i)].tolist(), "count": int(counts[int(i)])} for i in order
        ]
        result["unique_count"] = int(uniq.shape[0])
        result["unique_values"] = unique_values
        result["unique_decimals"] = int(unique_decimals)
    return result


def _print_top_variation(records: list[dict], limit: int, base_dir: Path | None) -> None:
    print("\nTop per-frame variation (max_abs_diff):")
    for rec in sorted(records, key=lambda r: r["max_abs_diff"], reverse=True)[:limit]:
        rel = rec["path"]
        if base_dir is not None:
            try:
                rel = str(Path(rel).relative_to(base_dir))
            except ValueError:
                pass
        print(
            f"- {rel} | frames={rec['frames']} dim={rec['dim']} "
            f"max_abs_diff={rec['max_abs_diff']:.6g} mean_abs_diff={rec['mean_abs_diff']:.6g} "
            f"std_mean={rec['std_mean']:.6g}"
        )


def _print_cross_video_stats(records: list[dict], base_dir: Path | None, limit: int) -> None:
    by_dim: dict[int, list[dict]] = {}
    for rec in records:
        by_dim.setdefault(rec["dim"], []).append(rec)

    print("\nAcross-video stats (by dim):")
    for dim, recs in sorted(by_dim.items(), key=lambda x: x[0]):
        means = np.array([r["mean_vec"] for r in recs], dtype=np.float64)
        if means.size == 0:
            continue
        per_param_min = means.min(axis=0)
        per_param_max = means.max(axis=0)
        per_param_std = means.std(axis=0)
        print(
            f"- dim={dim} videos={len(recs)} | "
            f"param_min={per_param_min.round(6).tolist()} | "
            f"param_max={per_param_max.round(6).tolist()} | "
            f"param_std={per_param_std.round(6).tolist()}"
        )

        center = means.mean(axis=0)
        dists = np.linalg.norm(means - center[None, :], axis=1)
        order = np.argsort(-dists)[:limit]
        print("  most different from global mean:")
        for idx in order:
            rec = recs[int(idx)]
            rel = rec["path"]
            if base_dir is not None:
                try:
                    rel = str(Path(rel).relative_to(base_dir))
                except ValueError:
                    pass
            print(f"  - {rel} | dist={dists[int(idx)]:.6g}")

def _print_unique_values(records: list[dict], base_dir: Path | None) -> None:
    print("\nUnique intrinsics per file:")
    for rec in records:
        if "unique_count" not in rec:
            continue
        rel = rec["path"]
        if base_dir is not None:
            try:
                rel = str(Path(rel).relative_to(base_dir))
            except ValueError:
                pass
        values = rec.get("unique_values", [])
        values_text = "; ".join(
            f"{item['value']} x{item['count']}" for item in values
        ) or "n/a"
        decimals = rec.get("unique_decimals", None)
        decimals_text = f", round={decimals}d" if decimals is not None else ""
        print(
            f"- {rel} | frames={rec['frames']} dim={rec['dim']} "
            f"unique={rec['unique_count']}{decimals_text} | {values_text}"
        )


def _pose_step_metrics(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("pose.npy must have shape (N, 4, 4)")
    t = poses[:, :3, 3]
    t_diff = np.linalg.norm(np.diff(t, axis=0), axis=1)
    r = poses[:, :3, :3]
    r_rel = np.matmul(np.transpose(r[:-1], (0, 2, 1)), r[1:])
    trace = np.einsum("nii->n", r_rel)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    ang = np.arccos(cos_theta)
    return t_diff, ang


def _summarize_pose(path: Path, jump_sigma: float) -> dict:
    poses = np.load(path)
    n_frames = int(poses.shape[0])
    t_diff, ang = _pose_step_metrics(poses)
    t_mean = float(t_diff.mean()) if t_diff.size else 0.0
    t_std = float(t_diff.std()) if t_diff.size else 0.0
    a_mean = float(ang.mean()) if ang.size else 0.0
    a_std = float(ang.std()) if ang.size else 0.0
    t_thr = t_mean + jump_sigma * t_std if t_diff.size else 0.0
    a_thr = a_mean + jump_sigma * a_std if ang.size else 0.0
    t_jumps = np.where(t_diff > t_thr)[0].tolist()
    a_jumps = np.where(ang > a_thr)[0].tolist()
    return {
        "path": str(path),
        "frames": n_frames,
        "trans_step_mean": t_mean,
        "trans_step_std": t_std,
        "rot_step_mean_rad": a_mean,
        "rot_step_std_rad": a_std,
        "trans_jump_threshold": float(t_thr),
        "rot_jump_threshold_rad": float(a_thr),
        "trans_jump_indices": t_jumps,
        "rot_jump_indices": a_jumps,
    }


def _print_pose_summary(records: list[dict], base_dir: Path | None, limit: int) -> None:
    print("\nPose step stats:")
    for rec in records:
        rel = rec["path"]
        if base_dir is not None:
            try:
                rel = str(Path(rel).relative_to(base_dir))
            except ValueError:
                pass
        t_jump = len(rec["trans_jump_indices"])
        r_jump = len(rec["rot_jump_indices"])
        t_idx = rec["trans_jump_indices"][:limit]
        r_idx = rec["rot_jump_indices"][:limit]
        print(
            f"- {rel} | frames={rec['frames']} "
            f"t_step_mean={rec['trans_step_mean']:.6g} std={rec['trans_step_std']:.6g} "
            f"r_step_mean={rec['rot_step_mean_rad']:.6g}rad std={rec['rot_step_std_rad']:.6g}rad "
            f"jumps: trans={t_jump} rot={r_jump} "
            f"idx(trans)={t_idx} idx(rot)={r_idx}"
        )


def _print_pairwise_similarity(
    records: list[dict],
    base_dir: Path | None,
    per_video_topk: int,
    global_topk: int,
    max_files: int,
) -> None:
    if per_video_topk <= 0 and global_topk <= 0:
        return
    by_dim: dict[int, list[dict]] = {}
    for rec in records:
        by_dim.setdefault(rec["dim"], []).append(rec)

    print("\nPairwise similarity (L2 distance on mean intrinsics):")
    for dim, recs in sorted(by_dim.items(), key=lambda x: x[0]):
        n = len(recs)
        if max_files > 0 and n > max_files:
            print(
                f"- dim={dim} videos={n} | skip pairwise (>{max_files}); "
                "raise --pairwise-max to enable"
            )
            continue

        means = np.array([r["mean_vec"] for r in recs], dtype=np.float64)
        norms = (means**2).sum(axis=1, keepdims=True)
        dist2 = norms + norms.T - 2.0 * (means @ means.T)
        dist2 = np.maximum(dist2, 0.0)
        dist = np.sqrt(dist2)
        np.fill_diagonal(dist, np.inf)

        print(f"- dim={dim} videos={n}")
        if global_topk > 0:
            triu = np.triu_indices(n, k=1)
            flat = dist[triu]
            k = min(global_topk, flat.size)
            if k > 0:
                idx = np.argpartition(flat, k - 1)[:k]
                best = idx[np.argsort(flat[idx])]
                print("  most similar pairs:")
                for j in best:
                    i0 = triu[0][j]
                    i1 = triu[1][j]
                    rec0 = recs[int(i0)]
                    rec1 = recs[int(i1)]
                    p0 = rec0["path"]
                    p1 = rec1["path"]
                    if base_dir is not None:
                        try:
                            p0 = str(Path(p0).relative_to(base_dir))
                            p1 = str(Path(p1).relative_to(base_dir))
                        except ValueError:
                            pass
                    print(f"  - {p0} <-> {p1} | dist={dist[int(i0), int(i1)]:.6g}")

        if per_video_topk > 0:
            k = min(per_video_topk, n - 1)
            if k <= 0:
                continue
            print("  nearest neighbors per video:")
            for i in range(n):
                rec = recs[i]
                rel = rec["path"]
                if base_dir is not None:
                    try:
                        rel = str(Path(rel).relative_to(base_dir))
                    except ValueError:
                        pass
                nn = np.argpartition(dist[i], k)[:k]
                nn = nn[np.argsort(dist[i][nn])]
                pairs = ", ".join(
                    f"{Path(recs[int(j)]['path']).name} ({dist[i, int(j)]:.6g})" for j in nn
                )
                print(f"  - {rel} -> {pairs}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze intrinsics/pose .npy files: per-frame consistency and pose jumps."
    )
    parser.add_argument("path", nargs="?", help="Path to intrinsics/pose .npy or folder.")
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        help="ViPE output dir (will scan output_dir/{intrinsics|pose}).",
    )
    parser.add_argument(
        "--mode",
        choices=["intrinsics", "pose"],
        default="intrinsics",
        help="Analyze intrinsics or pose npy files.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for per-frame equality (max abs diff <= tol).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Top-K entries to show for variation/differences.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optional path to save JSON report.",
    )
    parser.add_argument(
        "--show-unique",
        action="store_true",
        help="Show per-file unique intrinsics values (rounded).",
    )
    parser.add_argument(
        "--unique-max",
        type=int,
        default=5,
        help="Max unique values to show per file (default: 5).",
    )
    parser.add_argument(
        "--unique-decimals",
        type=int,
        default=6,
        help="Round decimals before computing uniqueness (default: 6).",
    )
    parser.add_argument(
        "--unique-only",
        action="store_true",
        help="Only show unique values (skip distance-based summaries).",
    )
    parser.add_argument(
        "--pairwise-topk",
        type=int,
        default=0,
        help="Show K nearest neighbors for each video (0 disables).",
    )
    parser.add_argument(
        "--pairwise-global-topk",
        type=int,
        default=0,
        help="Show K most similar pairs globally (0 disables).",
    )
    parser.add_argument(
        "--pairwise-max",
        type=int,
        default=300,
        help="Max videos per dim to compute pairwise distances (0 disables limit).",
    )
    parser.add_argument(
        "--jump-sigma",
        type=float,
        default=3.0,
        help="Pose jump threshold = mean + sigma * std (default: 3.0).",
    )
    parser.add_argument(
        "--jump-limit",
        type=int,
        default=10,
        help="Max jump indices to print per file.",
    )
    args = parser.parse_args()

    files, base_dir = _resolve_targets(args)
    if not files:
        raise FileNotFoundError("No .npy files found.")

    records: list[dict] = []
    if args.mode == "pose":
        for path in files:
            try:
                records.append(_summarize_pose(path, args.jump_sigma))
            except Exception as exc:
                print(f"[WARN] Failed to read {path}: {exc}")
    else:
        collect_unique = args.show_unique or args.unique_only
        for path in files:
            try:
                records.append(
                    _summarize_per_video(
                        path,
                        args.tol,
                        collect_unique=collect_unique,
                        unique_max=args.unique_max,
                        unique_decimals=args.unique_decimals,
                    )
                )
            except Exception as exc:
                print(f"[WARN] Failed to read {path}: {exc}")

    if not records:
        raise RuntimeError("No valid intrinsics files loaded.")

    if args.mode == "pose":
        print("Pose jump detection:")
        print(f"- files={len(records)} | jump_sigma={args.jump_sigma}")
        _print_pose_summary(records, base_dir, args.jump_limit)
    else:
        n_constant = sum(1 for r in records if r["is_constant"])
        print("Per-frame consistency:")
        print(f"- files={len(records)} | constant={n_constant} | non-constant={len(records) - n_constant}")
        print(f"- tol={args.tol}")

        if collect_unique:
            _print_unique_values(records, base_dir)
        if not args.unique_only:
            _print_top_variation(records, args.limit, base_dir)
            _print_cross_video_stats(records, base_dir, args.limit)
            _print_pairwise_similarity(
                records,
                base_dir,
                args.pairwise_topk,
                args.pairwise_global_topk,
                args.pairwise_max,
            )

    if args.report_json:
        payload = {"summary": {"total": len(records), "mode": args.mode}, "records": records}
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {args.report_json}")


if __name__ == "__main__":
    main()
