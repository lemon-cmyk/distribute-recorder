from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq


CAMERA_STATE_INDICES = {
    "front": tuple(range(7)) + tuple(range(8, 15)),
    "left_wrist": tuple(range(7)),
    "right_wrist": tuple(range(8, 15)),
}


def _debug(message: str) -> None:
    print(f"[图像状态偏差] {message}", file=sys.stderr, flush=True)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float64, copy=False)
    kernel = np.full((window,), 1.0 / window, dtype=np.float64)
    return np.convolve(values, kernel, mode="same")


def state_motion_signal(states: np.ndarray, indices: Iterable[int], smooth_window: int) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    selected = values[:, tuple(indices)]
    deltas = np.diff(selected, axis=0)
    scales = np.quantile(np.abs(deltas), 0.95, axis=0)
    active = scales > 1e-5
    if not np.any(active):
        return np.zeros((len(values) - 1,), dtype=np.float64)
    normalized = deltas[:, active] / scales[active]
    motion = np.sqrt(np.mean(np.square(normalized), axis=1))
    upper = np.quantile(motion, 0.995)
    if upper > 0:
        motion = np.minimum(motion, upper)
    return _moving_average(motion, smooth_window)


def _aligned_signals(
    state_motion: np.ndarray,
    image_motion: np.ndarray,
    image_lag_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align signals where positive lag means the image is older than the row state."""
    size = min(len(state_motion), len(image_motion))
    state = state_motion[:size]
    image = image_motion[:size]
    if image_lag_frames > 0:
        return state[:-image_lag_frames], image[image_lag_frames:]
    if image_lag_frames < 0:
        return state[-image_lag_frames:], image[:image_lag_frames]
    return state, image


def estimate_image_lag(
    state_motion: np.ndarray,
    image_motion: np.ndarray,
    max_lag_frames: int,
) -> dict:
    correlations = {}
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        state, image = _aligned_signals(state_motion, image_motion, lag)
        if len(state) < 20 or np.std(state) < 1e-9 or np.std(image) < 1e-9:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(state, image)[0, 1])
        correlations[lag] = correlation

    finite = [(lag, value) for lag, value in correlations.items() if np.isfinite(value)]
    if not finite:
        raise ValueError("motion signals do not contain enough variation")
    best_lag, best_correlation = max(finite, key=lambda item: item[1])
    distinct = [value for lag, value in finite if abs(lag - best_lag) > 1]
    runner_up = max(distinct) if distinct else float("nan")
    peak_margin = best_correlation - runner_up if np.isfinite(runner_up) else float("nan")
    return {
        "lag_frames": int(best_lag),
        "peak_correlation": best_correlation,
        "peak_margin": peak_margin,
        "correlation_by_lag": {str(lag): value for lag, value in correlations.items()},
    }


def decode_image_motion(
    video_path: Path,
    *,
    width: int,
    height: int,
    smooth_window: int,
    duplicate_threshold: float,
) -> tuple[np.ndarray, int, float]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    frame_size = width * height
    previous = None
    motion = []
    frame_count = 0
    while True:
        raw = process.stdout.read(frame_size)
        if not raw:
            break
        while len(raw) < frame_size:
            chunk = process.stdout.read(frame_size - len(raw))
            if not chunk:
                break
            raw += chunk
        if len(raw) != frame_size:
            process.kill()
            raise RuntimeError(f"FFmpeg returned a partial frame for {video_path}")
        current = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
        if previous is not None:
            motion.append(float(np.mean(np.abs(current - previous))))
        previous = current
        frame_count += 1

    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed for {video_path}: {stderr[-2000:]}")
    raw_motion = np.asarray(motion)
    duplicate_ratio = (
        float(np.mean(raw_motion <= duplicate_threshold)) if len(raw_motion) else float("nan")
    )
    return _moving_average(raw_motion, smooth_window), frame_count, duplicate_ratio


def _episode_indices(root: Path) -> list[int]:
    path = root / "meta" / "episodes.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        return [int(json.loads(line)["episode_index"]) for line in stream if line.strip()]


def _episode_path(root: Path, kind: str, episode_index: int, camera_key: str | None = None) -> Path:
    chunk = episode_index // 1000
    if kind == "data":
        return root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    assert camera_key is not None
    return root / f"videos/chunk-{chunk:03d}/{camera_key}/episode_{episode_index:06d}.mp4"


def analyze_episode(
    root: Path,
    episode_index: int,
    *,
    fps: int,
    camera_keys: tuple[str, ...],
    max_lag_frames: int,
    decode_width: int,
    decode_height: int,
    smooth_window: int,
    duplicate_threshold: float,
) -> dict:
    table = pq.read_table(
        _episode_path(root, "data", episode_index),
        columns=["observation.state"],
    )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 16:
        raise ValueError(f"episode {episode_index} state shape is {states.shape}, expected (*, 16)")

    camera_results = {}
    for camera_key in camera_keys:
        camera_name = camera_key.rsplit(".", 1)[-1]
        if camera_name not in CAMERA_STATE_INDICES:
            raise ValueError(f"unsupported camera name for state matching: {camera_name}")
        video_path = _episode_path(root, "video", episode_index, camera_key)
        image_motion, video_frames, duplicate_ratio = decode_image_motion(
            video_path,
            width=decode_width,
            height=decode_height,
            smooth_window=smooth_window,
            duplicate_threshold=duplicate_threshold,
        )
        if abs(video_frames - len(states)) > 1:
            raise ValueError(
                f"episode {episode_index} {camera_name} has {video_frames} video frames "
                f"but {len(states)} state rows"
            )
        state_motion = state_motion_signal(
            states,
            CAMERA_STATE_INDICES[camera_name],
            smooth_window,
        )
        estimate = estimate_image_lag(state_motion, image_motion, max_lag_frames)
        estimate.update(
            {
                "lag_ms": estimate["lag_frames"] * 1000.0 / fps,
                "video_frames": video_frames,
                "duplicate_frame_ratio": duplicate_ratio,
                "mean_image_motion": float(np.mean(image_motion)),
                "mean_state_motion": float(np.mean(state_motion)),
            }
        )
        camera_results[camera_name] = estimate
    return {
        "episode_index": episode_index,
        "frame_count": len(states),
        "cameras": camera_results,
    }


def aggregate_results(episodes: list[dict], camera_names: Iterable[str], fps: int) -> dict:
    result = {}
    for camera_name in camera_names:
        values = [episode["cameras"][camera_name] for episode in episodes]
        lags = np.asarray([value["lag_frames"] for value in values], dtype=np.float64)
        correlations = np.asarray([value["peak_correlation"] for value in values])
        margins = np.asarray([value["peak_margin"] for value in values])
        duplicates = np.asarray([value["duplicate_frame_ratio"] for value in values])
        result[camera_name] = {
            "episode_count": len(values),
            "median_lag_frames": float(np.median(lags)),
            "median_lag_ms": float(np.median(lags) * 1000.0 / fps),
            "p05_lag_ms": float(np.quantile(lags, 0.05) * 1000.0 / fps),
            "p95_lag_ms": float(np.quantile(lags, 0.95) * 1000.0 / fps),
            "p95_abs_lag_ms": float(np.quantile(np.abs(lags), 0.95) * 1000.0 / fps),
            "max_abs_lag_ms": float(np.max(np.abs(lags)) * 1000.0 / fps),
            "median_peak_correlation": float(np.median(correlations)),
            "median_peak_margin": float(np.nanmedian(margins)),
            "median_duplicate_frame_ratio": float(np.median(duplicates)),
        }
    return result


def _parse_episode_selection(value: str | None, available: list[int]) -> list[int]:
    if value is None:
        return available
    selected = [int(item.strip()) for item in value.split(",") if item.strip()]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"episodes are not present in the dataset: {missing}")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate image/state lag in a LeRobot v2.1 dataset")
    parser.add_argument("dataset_root")
    parser.add_argument("--label", default="dataset")
    parser.add_argument("--episodes", help="comma-separated episode indices; default is all")
    parser.add_argument("--max-lag-frames", type=int, default=10)
    parser.add_argument("--decode-width", type=int, default=80)
    parser.add_argument("--decode-height", type=int, default=60)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--duplicate-threshold", type=float, default=0.25)
    args = parser.parse_args(argv)

    root = Path(args.dataset_root).expanduser().resolve()
    with (root / "meta" / "info.json").open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    fps = int(info["fps"])
    camera_keys = tuple(
        key
        for key, feature in info["features"].items()
        if feature.get("dtype") == "video" and key.rsplit(".", 1)[-1] in CAMERA_STATE_INDICES
    )
    if set(key.rsplit(".", 1)[-1] for key in camera_keys) != set(CAMERA_STATE_INDICES):
        raise ValueError(f"dataset does not contain the expected three video cameras: {camera_keys}")
    episodes = _parse_episode_selection(args.episodes, _episode_indices(root))
    if not episodes:
        raise ValueError("no episodes selected")

    results = []
    for position, episode_index in enumerate(episodes, start=1):
        _debug(f"{args.label}: 分析 episode {episode_index} ({position}/{len(episodes)})")
        results.append(
            analyze_episode(
                root,
                episode_index,
                fps=fps,
                camera_keys=camera_keys,
                max_lag_frames=args.max_lag_frames,
                decode_width=args.decode_width,
                decode_height=args.decode_height,
                smooth_window=args.smooth_window,
                duplicate_threshold=args.duplicate_threshold,
            )
        )

    camera_names = [key.rsplit(".", 1)[-1] for key in camera_keys]
    report = {
        "label": args.label,
        "dataset_root": str(root),
        "fps": fps,
        "lag_definition": "positive means the image is older than the state stored in the same row",
        "frame_period_ms": 1000.0 / fps,
        "parameters": {
            "max_lag_frames": args.max_lag_frames,
            "decode_size": [args.decode_width, args.decode_height],
            "smooth_window": args.smooth_window,
            "duplicate_threshold": args.duplicate_threshold,
        },
        "aggregate": aggregate_results(results, camera_names, fps),
        "episodes": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
