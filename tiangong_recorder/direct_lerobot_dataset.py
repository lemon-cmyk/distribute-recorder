from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .video_progress import encode_rgb_frames_with_jetson_h264


def _debug(stage: str, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[LeRobot转换][{timestamp}][{stage}] {message}", flush=True)


def _sample_indices(data_len: int) -> list[int]:
    min_samples = min(100, data_len)
    sample_count = max(min_samples, min(int(data_len ** 0.75), 10_000))
    return np.round(np.linspace(0, data_len - 1, sample_count)).astype(int).tolist()


def _feature_stats(array: np.ndarray, axis, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def _sample_image_array(images: list[np.ndarray]) -> np.ndarray:
    sampled = []
    for index in _sample_indices(len(images)):
        image = np.asarray(images[index], dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image statistics expected HWC RGB image, got {image.shape}"
            )
        channel_first = np.transpose(image, (2, 0, 1))
        _, height, width = channel_first.shape
        if max(width, height) >= 300:
            factor = int(width / 150) if width > height else int(height / 150)
            channel_first = channel_first[:, ::factor, ::factor]
        sampled.append(channel_first)
    return np.stack(sampled)


def compute_numpy_episode_stats(
    episode_data: dict[str, list[np.ndarray] | np.ndarray],
    features: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute LeRobot v2.1 statistics without writing image files."""
    stats = {}
    for key, data in episode_data.items():
        feature = features[key]
        if feature["dtype"] == "string":
            continue
        if feature["dtype"] in {"image", "video"}:
            array = _sample_image_array(data)
            values = _feature_stats(array, axis=(0, 2, 3), keepdims=True)
            stats[key] = {
                name: value if name == "count" else np.squeeze(value / 255.0, axis=0)
                for name, value in values.items()
            }
            continue

        array = np.asarray(data)
        stats[key] = _feature_stats(
            array,
            axis=0,
            keepdims=array.ndim == 1,
        )
    return stats


@lru_cache(maxsize=1)
def get_direct_lerobot_dataset_class():
    """Build the local subclass lazily so normal recorder tests do not require LeRobot."""
    try:
        import torch
        from lerobot.datasets.lerobot_dataset import (
            CODEBASE_VERSION,
            LeRobotDataset,
        )
        from lerobot.datasets.utils import validate_episode_buffer, validate_frame
    except ImportError as exc:
        raise RuntimeError(
            "Direct LeRobot conversion requires the isolated converter environment"
        ) from exc

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"expected LeRobot v2.1, got {CODEBASE_VERSION!r}")

    class DirectLeRobotDataset(LeRobotDataset):
        """LeRobot v2.1 writer that keeps images in memory and encodes videos directly."""

        direct_numpy_video = True

        def configure_direct_video(
            self,
            *,
            bitrate: int,
        ) -> None:
            self._direct_video_bitrate = int(bitrate)

        def add_frame(
            self, frame: dict, task: str, timestamp: float | None = None
        ) -> None:
            for name, value in frame.items():
                if isinstance(value, torch.Tensor):
                    frame[name] = value.cpu().numpy()
            validate_frame(frame, self.features)

            if self.episode_buffer is None:
                self.episode_buffer = self.create_episode_buffer()
            frame_index = self.episode_buffer["size"]
            if timestamp is None:
                timestamp = frame_index / self.fps
            self.episode_buffer["frame_index"].append(frame_index)
            self.episode_buffer["timestamp"].append(timestamp)
            self.episode_buffer["task"].append(task)

            for key, value in frame.items():
                if key not in self.features:
                    raise ValueError(f"frame key {key!r} is not in dataset features")
                self.episode_buffer[key].append(value)
            self.episode_buffer["size"] += 1

        def _encode_direct_video(
            self,
            frames: list[np.ndarray],
            video_path: Path,
            image_key: str,
        ) -> None:
            label = image_key.rsplit(".", 1)[-1]
            width = int(self.features[image_key]["shape"][1])
            height = int(self.features[image_key]["shape"][0])

            encode_rgb_frames_with_jetson_h264(
                frames,
                video_path,
                fps=self.fps,
                label=label,
                width=width,
                height=height,
                bitrate=self._direct_video_bitrate,
            )

        def save_episode(self, episode_data: dict | None = None) -> None:
            if episode_data is not None:
                raise ValueError(
                    "direct writer only supports its in-memory episode buffer"
                )
            buffer = self.episode_buffer
            validate_episode_buffer(buffer, self.meta.total_episodes, self.features)

            episode_length = int(buffer["size"])
            episode_index = int(buffer["episode_index"])
            tasks = list(buffer["task"])
            episode_tasks = list(dict.fromkeys(tasks))

            task_indices = dict(self.meta.task_to_task_index)
            next_task_index = self.meta.total_tasks
            new_tasks = []
            for task in episode_tasks:
                if task not in task_indices:
                    task_indices[task] = next_task_index
                    next_task_index += 1
                    new_tasks.append(task)

            prepared: dict[str, list[np.ndarray] | np.ndarray] = {}
            for key, feature in self.features.items():
                if key == "episode_index":
                    prepared[key] = np.full((episode_length,), episode_index)
                elif key == "index":
                    prepared[key] = np.arange(
                        self.meta.total_frames,
                        self.meta.total_frames + episode_length,
                    )
                elif key == "task_index":
                    prepared[key] = np.asarray([task_indices[task] for task in tasks])
                elif feature["dtype"] in {"image", "video"}:
                    prepared[key] = list(buffer[key])
                else:
                    prepared[key] = np.stack(buffer[key])

            _debug("阶段 7/8：统计计算", "直接从 NumPy 图像和状态动作计算 episode 统计")
            episode_stats = compute_numpy_episode_stats(prepared, self.features)

            _debug("阶段 7/8：直接视频编码", "跳过临时 PNG，开始编码三路 NumPy 图像")
            for image_key in self.meta.video_keys:
                video_path = self.root / self.meta.get_video_file_path(
                    episode_index,
                    image_key,
                )
                self._encode_direct_video(prepared[image_key], video_path, image_key)

            table_data = {key: prepared[key] for key in self.hf_features}
            _debug("阶段 7/8：写入 Parquet", f"写入 episode {episode_index} 状态动作和索引")
            self._save_episode_table(table_data, episode_index)
            parquet_path = self.root / self.meta.get_data_file_path(episode_index)
            if not parquet_path.is_file():
                raise RuntimeError(f"Parquet was not created: {parquet_path}")

            for task in new_tasks:
                self.meta.add_task(task)
            if episode_index == 0:
                self.meta.update_video_info()
            self.meta.save_episode(
                episode_index,
                episode_length,
                episode_tasks,
                episode_stats,
            )
            self.episode_buffer = self.create_episode_buffer()

        def clear_episode_buffer(self) -> None:
            self.episode_buffer = self.create_episode_buffer()

    DirectLeRobotDataset.__name__ = "DirectLeRobotDataset"
    return DirectLeRobotDataset
