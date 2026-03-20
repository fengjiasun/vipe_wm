import argparse
import json
import logging
import multiprocessing as mp
import os
import queue as queue_mod
import threading
import time
from typing import Any, Optional
from pathlib import Path

import numpy as np
import vipe.utils.logging as vipe_logging


FALLBACK_FIELDS = ("video", "video_path", "path", "filepath", "file")


def _iter_video_paths(jsonl_path: Path, field: str | None) -> list[Path]:
    paths: list[Path] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            value: str | None = None
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning("Invalid JSON at line %d, skip.", line_idx)
                    continue

                if field:
                    value = obj.get(field)
                    if value is None:
                        logging.warning("Missing field '%s' at line %d, skip.", field, line_idx)
                        continue
                else:
                    for k in FALLBACK_FIELDS:
                        if k in obj:
                            value = obj[k]
                            break
                    if value is None:
                        logging.warning("No video field found at line %d, skip.", line_idx)
                        continue
            else:
                value = line

            if isinstance(value, list):
                for v in value:
                    if isinstance(v, str):
                        paths.append(_resolve_path(jsonl_path, v))
                continue

            if not isinstance(value, str):
                logging.warning("Invalid path type at line %d, skip.", line_idx)
                continue

            paths.append(_resolve_path(jsonl_path, value))
    return paths


def _resolve_path(jsonl_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = jsonl_path.parent / path
    return path


def _artifact_name(path: Path, use_full_path: bool) -> str:
    if not use_full_path:
        return path.stem
    safe = str(path).replace(":", "").replace("\\", "__").replace("/", "__")
    return safe


def _pose_intrinsics_paths(output_dir: Path, video_path: Path) -> tuple[Path, Path]:
    # Use the folder above "clips" as subfolder name when possible.
    if video_path.parent.name == "clips" and video_path.parent.parent.name:
        group = video_path.parent.parent.name
    else:
        group = video_path.parent.name or "unknown"
    filename = video_path.name
    pose_path = output_dir / "pose" / group / f"{filename}.npy"
    intrinsics_path = output_dir / "intrinsics" / group / f"{filename}.npy"
    return pose_path, intrinsics_path


def _filter_videos(video_paths: list[Path], args: argparse.Namespace, logger: logging.Logger) -> list[Path]:
    """Pre-filter: remove non-existent and already-completed videos before distribution."""
    filtered: list[Path] = []
    skipped_missing = 0
    skipped_done = 0
    for vp in video_paths:
        if not vp.exists():
            skipped_missing += 1
            continue
        if args.resume:
            pose_path, intrinsics_path = _pose_intrinsics_paths(args.output, vp)
            if pose_path.exists() and intrinsics_path.exists():
                skipped_done += 1
                continue
        filtered.append(vp)
    if skipped_missing:
        logger.info("Pre-filter: skipped %d missing videos.", skipped_missing)
    if skipped_done:
        logger.info("Pre-filter: skipped %d already completed videos (resume).", skipped_done)
    logger.info("Videos to process after filtering: %d / %d", len(filtered), len(video_paths))
    return filtered


def _iter_tasks(task_source):
    """Yield paths from a list or multiprocessing Queue."""
    if isinstance(task_source, list):
        for item in task_source:
            yield item
        return
    while True:
        item = task_source.get()
        if item is None:
            break
        yield item


def _save_pose_npy(pose_path: Path, trajectory) -> None:
    pose_data = trajectory.matrix().cpu().numpy()
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pose_path, pose_data)


def _save_intrinsics_npy(intrinsics_path: Path, intrinsics, n_frames: int) -> None:
    intr_data = intrinsics.cpu().numpy()
    if intr_data.ndim == 1:
        intr_data = intr_data[None, :]
    if intr_data.shape[0] == 1 and n_frames > 1:
        intr_data = np.repeat(intr_data, n_frames, axis=0)
    intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(intrinsics_path, intr_data)

def _build_overrides(args: argparse.Namespace) -> list[str]:
    if args.visualize and not args.save_artifacts:
        logging.warning("--visualize requires --save-artifacts; visualization will be skipped.")

    overrides = [
        f"pipeline={args.pipeline}",
        f"pipeline.output.path={args.output}",
    ]
    if args.visualize:
        overrides.append("pipeline.output.save_viz=true")
        overrides.append("pipeline.slam.visualize=true")
    else:
        overrides.append("pipeline.output.save_viz=false")

    if args.pose_only_fast:
        overrides.append("pipeline.init.instance=null")
        overrides.append("pipeline.post.depth_align_model=null")

    if args.save_artifacts:
        overrides.append("pipeline.output.save_artifacts=true")
    else:
        overrides.append("pipeline.output.save_artifacts=false")

    if not args.verbose:
        overrides.append("+pipeline.slam.solver_verbose=false")

    return overrides


def _prefetch_worker(
    task_iter,
    make_stream_fn,
    out_queue: queue_mod.Queue,
    skip_fn,
    verbose: bool,
    batch_logger,
    worker_id: int,
) -> None:
    """Background thread: pre-decode videos so GPU never waits for I/O."""
    for video_path in task_iter:
        if not video_path.exists():
            if verbose:
                batch_logger.warning("[W%d][prefetch] Missing: %s", worker_id, video_path)
            out_queue.put((video_path, None, True))
            continue
        if skip_fn(video_path):
            out_queue.put((video_path, None, True))
            continue
        try:
            stream = make_stream_fn(video_path)
            out_queue.put((video_path, stream, False))
        except Exception:
            out_queue.put((video_path, None, True))
    out_queue.put(None)


EMPTY_CACHE_INTERVAL = 20


def _process_videos(
    worker_id: int,
    gpu_id: int,
    task_source,
    total_count: int,
    args: argparse.Namespace,
    overrides: list[str],
    progress: Optional[Any] = None,
    progress_lock: Optional[Any] = None,
) -> None:
    import torch
    import hydra
    from vipe import get_config_path, make_pipeline
    from vipe.streams.base import ProcessedVideoStream
    from vipe.streams.raw_mp4_stream import RawMp4Stream
    from vipe.utils.logging import configure_logging

    torch.cuda.set_device(gpu_id)
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if not args.verbose:
        vipe_logging.disable_progress_bar = True
        logging.getLogger("vipe").setLevel(logging.WARNING)
    configure_logging()
    batch_logger = _get_batch_logger()

    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        cfg = hydra.compose("default", overrides=overrides)

    pipeline = make_pipeline(cfg.pipeline)
    if not args.save_artifacts:
        pipeline.return_payload = True

    def _make_stream(video_path: Path):
        stream_name = _artifact_name(video_path, args.name_from_path)
        return ProcessedVideoStream(RawMp4Stream(video_path, name=stream_name), []).cache(
            desc="Reading video stream"
        )

    def _should_skip(video_path: Path) -> bool:
        if args.resume:
            pose_path, intrinsics_path = _pose_intrinsics_paths(args.output, video_path)
            if pose_path.exists() and intrinsics_path.exists():
                return True
        return False

    prefetch_queue_size = max(1, int(args.prefetch_queue_size))
    prefetch_queue: queue_mod.Queue = queue_mod.Queue(maxsize=prefetch_queue_size)
    prefetch_thread = threading.Thread(
        target=_prefetch_worker,
        args=(
            _iter_tasks(task_source),
            _make_stream,
            prefetch_queue,
            _should_skip,
            args.verbose,
            batch_logger,
            worker_id,
        ),
        daemon=True,
    )
    prefetch_thread.start()

    total = total_count if total_count > 0 else 0
    start_time = time.time()
    last_log_time = start_time
    local_done = 0
    error_log_path = _error_log_path(args.output, worker_id)
    idx = 0

    t_wait_acc = 0.0
    t_infer_acc = 0.0
    t_save_acc = 0.0
    profile_count = 0

    while True:
        t_wait_start = time.time()
        item = prefetch_queue.get()
        t_wait = time.time() - t_wait_start
        if item is None:
            break
        video_path, video_stream, skipped = item
        idx += 1
        local_done += 1

        if skipped:
            if progress is not None and progress_lock is not None:
                with progress_lock:
                    progress.value += 1
            continue

        try:
            t_infer_start = time.time()

            if args.save_artifacts:
                pipeline.run(video_stream)
            else:
                output = pipeline.run(video_stream)
                if output.payload is None:
                    if args.verbose:
                        batch_logger.warning("[W%d] No payload returned for %s", worker_id, video_path)
                else:
                    t_save_start = time.time()
                    t_infer_elapsed = t_save_start - t_infer_start

                    pose_path, intrinsics_path = _pose_intrinsics_paths(args.output, video_path)
                    _save_pose_npy(pose_path, output.payload.trajectory)
                    _save_intrinsics_npy(
                        intrinsics_path,
                        output.payload.intrinsics[0],
                        output.payload.trajectory.matrix().shape[0],
                    )
                    t_save_elapsed = time.time() - t_save_start

                    t_wait_acc += t_wait
                    t_infer_acc += t_infer_elapsed
                    t_save_acc += t_save_elapsed
                    profile_count += 1

                    if profile_count <= 5 or profile_count % 20 == 0:
                        batch_logger.info(
                            "[W%d] #%d timing: wait_io=%.2fs pipeline=%.2fs save=%.2fs | "
                            "avg(last %d): wait_io=%.2fs pipeline=%.2fs save=%.2fs | %s",
                            worker_id, profile_count,
                            t_wait, t_infer_elapsed, t_save_elapsed,
                            profile_count,
                            t_wait_acc / profile_count,
                            t_infer_acc / profile_count,
                            t_save_acc / profile_count,
                            video_path.name,
                        )
        except Exception as exc:
            if args.verbose:
                batch_logger.exception("[W%d] Failed on %s", worker_id, video_path)
            _write_error(error_log_path, video_path, exc)
        finally:
            if local_done % EMPTY_CACHE_INTERVAL == 0:
                torch.cuda.empty_cache()
            if progress is not None and progress_lock is not None:
                with progress_lock:
                    progress.value += 1
            if progress is None:
                _log_local_progress(batch_logger, worker_id, local_done, total, start_time, last_log_time)
                last_log_time = time.time()

    prefetch_thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ViPE pose inference from a jsonl list.")
    parser.add_argument("jsonl", type=Path, help="Path to jsonl file containing video paths.")
    parser.add_argument("--output", "-o", type=Path, default=Path.cwd() / "vipe_results")
    parser.add_argument("--pipeline", "-p", default="default")
    parser.add_argument("--field", help="JSON field name containing the video path.")
    parser.add_argument("--visualize", action="store_true", help="Save visualization video.")
    parser.add_argument("--save-artifacts", action="store_true", help="Save full artifacts (rgb/depth/etc).")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode; re-process even if outputs exist.",
    )
    parser.add_argument("--name-from-path", action="store_true", help="Use full path to create unique names.")
    parser.add_argument(
        "--pose-only-fast",
        action="store_true",
        help="Disable instance segmentation and post depth alignment for faster runs (pose/intrinsics still saved).",
    )
    parser.add_argument(
        "--gpus",
        default="auto",
        help="GPU ids to use, e.g. '0,1,2,3'. Use 'auto' to use all visible GPUs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (default: one per GPU). "
        "Can exceed GPU count to overlap IO with compute.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Workers per GPU when --workers is not set (default: 1).",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable cudnn benchmark mode for potentially faster convolutions.",
    )
    parser.add_argument(
        "--prefetch-queue-size",
        type=int,
        default=1,
        help="Prefetch queue size per worker (default: 1).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-video logs and internal ViPE logs.",
    )
    args = parser.parse_args()
    if not hasattr(args, "resume"):
        args.resume = True

    if not args.verbose:
        vipe_logging.disable_progress_bar = True
        logging.getLogger("vipe").setLevel(logging.WARNING)
    logger = _get_batch_logger()
    jsonl_path = args.jsonl
    if not jsonl_path.exists():
        raise FileNotFoundError(f"jsonl not found: {jsonl_path}")

    raw_video_paths = _iter_video_paths(jsonl_path, args.field)
    if not raw_video_paths:
        logger.warning("No valid video paths found in %s", jsonl_path)
        return
    _warn_duplicate_names(raw_video_paths, args.output, logger)

    # Pre-filter before distributing — avoids idle workers
    video_paths = _filter_videos(raw_video_paths, args, logger)
    if not video_paths:
        logger.info("All videos already processed or missing. Nothing to do.")
        return

    overrides = _build_overrides(args)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if args.gpus == "auto":
        env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_gpus:
            gpu_ids = [idx for idx, _ in enumerate(env_gpus.split(",")) if _.strip() != ""]
        else:
            import torch
            n_gpus = torch.cuda.device_count()
            gpu_ids = list(range(n_gpus))
    else:
        gpu_list = [x.strip() for x in args.gpus.split(",") if x.strip() != ""]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
        gpu_ids = list(range(len(gpu_list)))
    if not gpu_ids:
        raise RuntimeError("No GPUs available. Please set --gpus or check CUDA.")

    if args.workers > 0:
        num_workers = args.workers
    else:
        per_gpu = max(1, args.workers_per_gpu)
        num_workers = len(gpu_ids) * per_gpu
    # Allow more workers than GPUs to overlap IO/preprocessing with GPU compute
    # Each worker uses gpu_ids[worker_id % len(gpu_ids)]
    if num_workers <= 1:
        _process_videos(0, gpu_ids[0], video_paths, len(video_paths), args, overrides, None, None)
        logger.info("Done.")
        return

    # Dynamic task queue: workers pull tasks on-demand instead of static round-robin
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    for vp in video_paths:
        task_queue.put(vp)
    for _ in range(num_workers):
        task_queue.put(None)

    progress = ctx.Value("i", 0)
    progress_lock = ctx.Lock()
    total = len(video_paths)
    workers: list[mp.Process] = []
    for worker_id in range(num_workers):
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]
        proc = ctx.Process(
            target=_process_videos,
            args=(worker_id, gpu_id, task_queue, total, args, overrides, progress, progress_lock),
            daemon=False,
        )
        proc.start()
        workers.append(proc)

    start_time = time.time()
    last_report = start_time
    while any(proc.is_alive() for proc in workers):
        time.sleep(5)
        now = time.time()
        if now - last_report >= 10:
            with progress_lock:
                done = progress.value
            elapsed = now - start_time
            eta = (elapsed / done) * (total - done) if done > 0 else 0.0
            logger.info(
                "Overall %d/%d | Elapsed %s | ETA %s",
                done,
                total,
                _format_time(elapsed),
                _format_time(eta),
            )
            last_report = now

    for proc in workers:
        proc.join()

    logger.info("Done.")


def _format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _get_batch_logger() -> logging.Logger:
    logger = logging.getLogger("vipe.batch")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _error_log_path(output_dir: Path, worker_id: int) -> Path:
    error_dir = output_dir / "vipe_errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    return error_dir / f"infer_errors_worker{worker_id}.jsonl"


def _write_error(error_log_path: Path, video_path: Path, exc: Exception) -> None:
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "video": str(video_path),
        "error_type": type(exc).__name__,
        "error": repr(exc),
    }
    with error_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _warn_duplicate_names(video_paths: list[Path], output_dir: Path, logger: logging.Logger) -> None:
    name_map: dict[str, int] = {}
    for path in video_paths:
        pose_path, _ = _pose_intrinsics_paths(output_dir, path)
        key = str(pose_path)
        name_map[key] = name_map.get(key, 0) + 1
    dup_paths = [name for name, count in name_map.items() if count > 1]
    if not dup_paths:
        return
    sample = ", ".join(dup_paths[:3])
    logger.warning(
        "Detected %d duplicate output paths (e.g. %s). Check your input list for duplicates.",
        len(dup_paths),
        sample,
    )


def _log_local_progress(
    logger: logging.Logger,
    worker_id: int,
    done: int,
    total: int,
    start_time: float,
    last_log_time: float,
) -> None:
    now = time.time()
    if done == 1 or done % 10 == 0 or now - last_log_time >= 30:
        elapsed = now - start_time
        eta = (elapsed / done) * (total - done) if done > 0 else 0.0
        logger.info(
            "[W%d] Progress %d/%d | Elapsed %s | ETA %s",
            worker_id,
            done,
            total,
            _format_time(elapsed),
            _format_time(eta),
        )


if __name__ == "__main__":
    main()

