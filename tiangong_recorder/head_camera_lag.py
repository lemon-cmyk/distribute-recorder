from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq


ARM_JOINTS = {
    "left_arm": (
        "shoulder_pitch_l_joint",
        "shoulder_roll_l_joint",
        "shoulder_yaw_l_joint",
        "elbow_pitch_l_joint",
        "elbow_yaw_l_joint",
        "wrist_pitch_l_joint",
        "wrist_roll_l_joint",
    ),
    "right_arm": (
        "shoulder_pitch_r_joint",
        "shoulder_roll_r_joint",
        "shoulder_yaw_r_joint",
        "elbow_pitch_r_joint",
        "elbow_yaw_r_joint",
        "wrist_pitch_r_joint",
        "wrist_roll_r_joint",
    ),
}
ARM_STATE_INDICES = {
    "left_arm": tuple(range(7)),
    "right_arm": tuple(range(8, 15)),
}
END_EFFECTOR_LINKS = {"left_arm": "hand_left", "right_arm": "hand_right"}
# The head camera uses the robot's first-person view, so each arm stays on the same image side.
ARM_IMAGE_ROI = {"left_arm": "image_left", "right_arm": "image_right"}


def _debug(message: str) -> None:
    print(f"[头部图像偏差] {message}", file=sys.stderr, flush=True)


def _vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(item) for item in text.split()], dtype=np.float64)


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return np.asarray(
        (
            (c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s),
            (y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s),
            (z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c),
        ),
        dtype=np.float64,
    )


def _transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class UrdfForwardKinematics:
    def __init__(self, urdf_path: str | Path):
        root = ET.parse(Path(urdf_path)).getroot()
        self.by_child: dict[str, UrdfJoint] = {}
        for element in root.findall("joint"):
            origin_element = element.find("origin")
            xyz = _vector(origin_element.get("xyz") if origin_element is not None else None, (0, 0, 0))
            rpy = _vector(origin_element.get("rpy") if origin_element is not None else None, (0, 0, 0))
            axis_element = element.find("axis")
            axis = _vector(axis_element.get("xyz") if axis_element is not None else None, (1, 0, 0))
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                continue
            joint = UrdfJoint(
                name=str(element.get("name")),
                joint_type=str(element.get("type", "fixed")),
                parent=str(parent.get("link")),
                child=str(child.get("link")),
                origin=_transform(_rpy_rotation(rpy), xyz),
                axis=axis,
            )
            self.by_child[joint.child] = joint
        self._chain_cache: dict[str, tuple[UrdfJoint, ...]] = {}

    def chain(self, end_link: str) -> tuple[UrdfJoint, ...]:
        if end_link in self._chain_cache:
            return self._chain_cache[end_link]
        result = []
        link = end_link
        while link in self.by_child:
            joint = self.by_child[link]
            result.append(joint)
            link = joint.parent
        result.reverse()
        if not result:
            raise ValueError(f"URDF contains no chain to {end_link!r}")
        self._chain_cache[end_link] = tuple(result)
        return self._chain_cache[end_link]

    def forward(self, end_link: str, joint_positions: dict[str, float]) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        for joint in self.chain(end_link):
            result = result @ joint.origin
            value = float(joint_positions.get(joint.name, 0.0))
            if joint.joint_type in {"revolute", "continuous"}:
                result = result @ _transform(rotation=_axis_rotation(joint.axis, value))
            elif joint.joint_type == "prismatic":
                result = result @ _transform(translation=joint.axis * value)
        return result


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.zeros((3,), dtype=np.float64)
    axis = np.asarray(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    ) / (2.0 * math.sin(angle))
    return axis * angle


def end_effector_motion_features(
    states: np.ndarray,
    arm_name: str,
    kinematics: UrdfForwardKinematics,
) -> np.ndarray:
    joint_names = ARM_JOINTS[arm_name]
    indices = ARM_STATE_INDICES[arm_name]
    poses = []
    for row in states:
        positions = {name: float(row[index]) for name, index in zip(joint_names, indices)}
        poses.append(kinematics.forward(END_EFFECTOR_LINKS[arm_name], positions))

    features = []
    for previous, current in zip(poses, poses[1:]):
        linear = current[:3, 3] - previous[:3, 3]
        local_rotation = previous[:3, :3].T @ current[:3, :3]
        angular = previous[:3, :3] @ _rotation_vector(local_rotation)
        features.append(
            np.concatenate(
                (
                    linear,
                    angular,
                    [np.linalg.norm(linear), np.linalg.norm(angular)],
                )
            )
        )
    return np.asarray(features, dtype=np.float64)


def _roi_slices(height: int, width: int) -> dict[str, tuple[slice, slice]]:
    top = int(round(height * 0.30))
    return {
        "image_left": (slice(top, height), slice(0, int(round(width * 0.55)))),
        "image_right": (slice(top, height), slice(int(round(width * 0.45)), width)),
    }


def decode_head_optical_flow(
    video_path: Path,
    *,
    width: int,
    height: int,
) -> tuple[dict[str, np.ndarray], int, float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("head-camera optical flow analysis requires OpenCV") from exc

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
    rois = _roi_slices(height, width)
    features = {name: [] for name in rois}
    previous = None
    frame_count = 0
    duplicate_count = 0
    frame_size = width * height
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
        gray = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        if previous is not None:
            difference = float(np.mean(np.abs(gray.astype(np.int16) - previous.astype(np.int16))))
            duplicate_count += int(difference <= 0.25)
            flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            for name, slices in rois.items():
                roi = flow[slices]
                magnitude = np.linalg.norm(roi, axis=2)
                threshold = max(0.03, float(np.quantile(magnitude, 0.70)))
                moving = magnitude >= threshold
                if not np.any(moving):
                    features[name].append(np.zeros((4,), dtype=np.float64))
                else:
                    features[name].append(
                        np.asarray(
                            (
                                np.mean(roi[..., 0][moving]),
                                np.mean(roi[..., 1][moving]),
                                np.mean(magnitude[moving]),
                                np.quantile(magnitude, 0.90),
                            ),
                            dtype=np.float64,
                        )
                    )
        previous = gray.copy()
        frame_count += 1
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed for {video_path}: {stderr[-2000:]}")
    transitions = max(1, frame_count - 1)
    return (
        {name: np.asarray(values, dtype=np.float64) for name, values in features.items()},
        frame_count,
        duplicate_count / transitions,
    )


def _inverse_square_root(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    inverse = 1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    return (eigenvectors * inverse) @ eigenvectors.T


def regularized_canonical_correlation(
    state_features: np.ndarray,
    image_features: np.ndarray,
    ridge: float = 1e-3,
) -> float:
    if len(state_features) < 30:
        return float("nan")
    state = np.asarray(state_features, dtype=np.float64)
    image = np.asarray(image_features, dtype=np.float64)
    state = (state - np.mean(state, axis=0)) / np.maximum(np.std(state, axis=0), 1e-8)
    image = (image - np.mean(image, axis=0)) / np.maximum(np.std(image, axis=0), 1e-8)
    count = len(state)
    covariance_state = state.T @ state / max(1, count - 1) + ridge * np.eye(state.shape[1])
    covariance_image = image.T @ image / max(1, count - 1) + ridge * np.eye(image.shape[1])
    cross_covariance = state.T @ image / max(1, count - 1)
    whitened = (
        _inverse_square_root(covariance_state)
        @ cross_covariance
        @ _inverse_square_root(covariance_image)
    )
    return float(np.clip(np.linalg.svd(whitened, compute_uv=False)[0], 0.0, 1.0))


def _align_features(state: np.ndarray, image: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(state), len(image))
    state = state[:size]
    image = image[:size]
    if lag > 0:
        return state[:-lag], image[lag:]
    if lag < 0:
        return state[-lag:], image[:lag]
    return state, image


def correlation_curve(
    state_features: np.ndarray,
    image_features: np.ndarray,
    max_lag_frames: int,
) -> dict[int, float]:
    speed = state_features[:, -2] + 0.25 * state_features[:, -1]
    activity_threshold = float(np.quantile(speed, 0.40))
    result = {}
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        state, image = _align_features(state_features, image_features, lag)
        aligned_speed, _ = _align_features(speed[:, None], image_features, lag)
        active = aligned_speed[:, 0] > activity_threshold
        result[lag] = regularized_canonical_correlation(state[active], image[active])
    return result


def _curve_peak(curve: dict[int, float], fps: int) -> dict:
    finite = [(lag, value) for lag, value in curve.items() if np.isfinite(value)]
    if not finite:
        raise ValueError("no finite optical-flow correlation values")
    best_lag, best_value = max(finite, key=lambda item: item[1])
    distinct = [value for lag, value in finite if abs(lag - best_lag) > 1]
    runner_up = max(distinct) if distinct else float("nan")
    subframe_lag = float(best_lag)
    if best_lag - 1 in curve and best_lag + 1 in curve:
        left, center, right = curve[best_lag - 1], curve[best_lag], curve[best_lag + 1]
        denominator = left - 2.0 * center + right
        if np.isfinite(denominator) and denominator < -1e-8:
            subframe_lag += float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
    return {
        "lag_frames": int(best_lag),
        "subframe_lag_frames": subframe_lag,
        "lag_ms": subframe_lag * 1000.0 / fps,
        "peak_correlation": float(best_value),
        "peak_margin": float(best_value - runner_up),
        "correlation_by_lag": {str(key): value for key, value in curve.items()},
    }


def estimate_episode_lag(
    states: np.ndarray,
    optical_flow: dict[str, np.ndarray],
    kinematics: UrdfForwardKinematics,
    *,
    fps: int,
    max_lag_frames: int,
) -> dict:
    arm_results = {}
    curves = []
    for arm_name in ARM_JOINTS:
        state_features = end_effector_motion_features(states, arm_name, kinematics)
        image_features = optical_flow[ARM_IMAGE_ROI[arm_name]]
        curve = correlation_curve(state_features, image_features, max_lag_frames)
        curves.append(curve)
        arm_results[arm_name] = _curve_peak(curve, fps)

    combined_curve = {
        lag: float(np.mean([curve[lag] for curve in curves]))
        for lag in range(-max_lag_frames, max_lag_frames + 1)
    }
    combined = _curve_peak(combined_curve, fps)
    arm_lags = [value["subframe_lag_frames"] for value in arm_results.values()]
    combined["arm_agreement_frames"] = float(abs(arm_lags[0] - arm_lags[1]))
    combined["reliable"] = bool(
        abs(combined["lag_frames"]) < max_lag_frames
        and combined["peak_correlation"] >= 0.20
        and combined["peak_margin"] >= 0.005
        and combined["arm_agreement_frames"] <= 3.0
    )
    return {"combined": combined, "arms": arm_results}


def _episode_indices(root: Path) -> list[int]:
    with (root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as stream:
        return [int(json.loads(line)["episode_index"]) for line in stream if line.strip()]


def _episode_path(root: Path, kind: str, episode_index: int) -> Path:
    chunk = episode_index // 1000
    if kind == "data":
        return root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    return root / (
        f"videos/chunk-{chunk:03d}/observation.images.front/episode_{episode_index:06d}.mp4"
    )


def _parse_episode_selection(value: str | None, available: list[int]) -> list[int]:
    if value is None:
        return available
    selected = [int(item.strip()) for item in value.split(",") if item.strip()]
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"episodes are not present in the dataset: {missing}")
    return selected


def aggregate_episode_results(episodes: list[dict]) -> dict:
    reliable = [episode for episode in episodes if episode["lag"]["combined"]["reliable"]]
    values = reliable or episodes
    lags = np.asarray([episode["lag"]["combined"]["lag_ms"] for episode in values])
    correlations = np.asarray(
        [episode["lag"]["combined"]["peak_correlation"] for episode in values]
    )
    return {
        "episode_count": len(episodes),
        "reliable_episode_count": len(reliable),
        "statistics_use_reliable_only": bool(reliable),
        "median_lag_ms": float(np.median(lags)),
        "p05_lag_ms": float(np.quantile(lags, 0.05)),
        "p95_lag_ms": float(np.quantile(lags, 0.95)),
        "p95_abs_lag_ms": float(np.quantile(np.abs(lags), 0.95)),
        "max_abs_lag_ms": float(np.max(np.abs(lags))),
        "median_peak_correlation": float(np.median(correlations)),
        "median_duplicate_frame_ratio": float(
            np.median([episode["duplicate_frame_ratio"] for episode in values])
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate Tiangong head-camera/state lag with optical flow")
    parser.add_argument("dataset_root")
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--label", default="dataset")
    parser.add_argument("--episodes", help="comma-separated episode indices; default is all")
    parser.add_argument("--max-lag-frames", type=int, default=10)
    parser.add_argument("--decode-width", type=int, default=160)
    parser.add_argument("--decode-height", type=int, default=120)
    args = parser.parse_args(argv)

    root = Path(args.dataset_root).expanduser().resolve()
    with (root / "meta" / "info.json").open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    fps = int(info["fps"])
    selected = _parse_episode_selection(args.episodes, _episode_indices(root))
    kinematics = UrdfForwardKinematics(args.urdf)
    episodes = []
    for position, episode_index in enumerate(selected, start=1):
        _debug(f"{args.label}: 分析 episode {episode_index} ({position}/{len(selected)})")
        table = pq.read_table(
            _episode_path(root, "data", episode_index),
            columns=["observation.state"],
        )
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 16:
            raise ValueError(f"episode {episode_index} state shape is {states.shape}, expected (*, 16)")
        optical_flow, video_frames, duplicate_ratio = decode_head_optical_flow(
            _episode_path(root, "video", episode_index),
            width=args.decode_width,
            height=args.decode_height,
        )
        if abs(video_frames - len(states)) > 1:
            raise ValueError(
                f"episode {episode_index} has {video_frames} video frames but {len(states)} state rows"
            )
        episodes.append(
            {
                "episode_index": episode_index,
                "frame_count": len(states),
                "duplicate_frame_ratio": duplicate_ratio,
                "lag": estimate_episode_lag(
                    states,
                    optical_flow,
                    kinematics,
                    fps=fps,
                    max_lag_frames=args.max_lag_frames,
                ),
            }
        )

    report = {
        "label": args.label,
        "dataset_root": str(root),
        "fps": fps,
        "lag_definition": "positive means the head image is older than the state in the same row",
        "method": "lower bilateral ROI Farneback flow vs URDF end-effector linear/angular velocity",
        "parameters": {
            "max_lag_frames": args.max_lag_frames,
            "decode_size": [args.decode_width, args.decode_height],
            "arm_image_roi": ARM_IMAGE_ROI,
        },
        "aggregate": aggregate_episode_results(episodes),
        "episodes": episodes,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
