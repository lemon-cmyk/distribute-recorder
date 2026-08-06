from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .direct_lerobot_dataset import get_direct_lerobot_dataset_class


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
ARM_NAMES = ("left_arm", "right_arm")


def _debug(stage: str, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[LeRobot转换][{timestamp}][{stage}] {message}", flush=True)


@dataclass(frozen=True)
class EpisodeWriteResult:
    episode_index: int
    frame_count: int
    dataset_root: Path


class LeRobotV21Writer:
    """Convert one legacy Tiangong PKL episode into LeRobot v2.1."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        repo_id: str,
        task: str,
        fps: int,
        use_videos: bool,
        image_width: int = 640,
        image_height: int = 480,
        state_key: str = "observation.state",
        action_key: str = "action",
        image_prefix: str = "observation.images",
        camera_color_order: Mapping[str, str] | None = None,
        image_writer_threads: int = 2,
        direct_video_bitrate: int = 8_000_000,
        _dataset_cls=None,
    ) -> None:
        if not task.strip():
            raise ValueError("task must not be empty")
        if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")

        orders = dict(
            camera_color_order
            or {
                "head": "rgb",
                "left_wrist": "bgr",
                "right_wrist": "bgr",
            }
        )
        if set(orders) != set(CAMERA_NAMES):
            raise ValueError(f"camera_color_order must contain exactly {CAMERA_NAMES}")
        if any(order not in {"rgb", "bgr"} for order in orders.values()):
            raise ValueError("camera color orders must be 'rgb' or 'bgr'")

        self.root = Path(dataset_root).expanduser()
        self.repo_id = repo_id
        self.task = task.strip()
        self.fps = fps
        self.use_videos = bool(use_videos)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.state_key = self._feature_key(state_key, "state_key")
        self.action_key = self._feature_key(action_key, "action_key")
        self.image_prefix = self._feature_key(image_prefix.rstrip("."), "image_prefix")
        self.camera_color_order = orders
        self.image_writer_threads = int(image_writer_threads)
        self.direct_video_bitrate = int(direct_video_bitrate)
        if self.direct_video_bitrate <= 0:
            raise ValueError("direct_video_bitrate must be positive")
        self._dataset_cls = _dataset_cls or self._import_lerobot_dataset()
        self.dataset = None

        generated_keys = [
            self.state_key,
            self.action_key,
            *(self._image_key(name) for name in CAMERA_NAMES),
        ]
        if len(generated_keys) != len(set(generated_keys)):
            raise ValueError(f"LeRobot feature names must be unique: {generated_keys}")

    @staticmethod
    def _feature_key(value: str, name: str) -> str:
        result = str(value).strip()
        if not result or "/" in result:
            raise ValueError(f"{name} must be non-empty and must not contain '/'")
        return result

    @staticmethod
    def _import_lerobot_dataset():
        try:
            from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
        except ImportError as exc:
            raise RuntimeError(
                "The converter requires lerobot==0.3.3 in its isolated environment"
            ) from exc
        if CODEBASE_VERSION != "v2.1":
            raise RuntimeError(
                f"Expected LeRobot codebase v2.1, got {CODEBASE_VERSION!r}"
            )
        return get_direct_lerobot_dataset_class()

    def _image_key(self, camera_name: str) -> str:
        return f"{self.image_prefix}.{camera_name}"

    @staticmethod
    def _float_vector(value: Any, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32).reshape(-1)
        if result.size == 0:
            raise ValueError(f"{name} is empty")
        if not np.isfinite(result).all():
            raise ValueError(f"{name} contains NaN or infinity")
        return result

    def _arm_vector(self, entry: dict, key: str) -> np.ndarray:
        values = entry.get(key)
        if not isinstance(values, dict):
            raise ValueError(f"missing Tiangong field: {key}")
        return np.concatenate(
            [self._float_vector(values[arm], f"{key}.{arm}") for arm in ARM_NAMES]
        )

    def _hand_vector(self, entry: dict, key: str) -> np.ndarray:
        values = entry.get(key)
        if values is None:
            return np.empty((0,), dtype=np.float32)
        if not isinstance(values, dict):
            raise ValueError(f"{key} must be a dictionary")
        present = [values.get(arm) is not None for arm in ARM_NAMES]
        if any(present) and not all(present):
            raise ValueError(f"{key} must contain both hands or neither hand")
        if not any(present):
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(
            [self._float_vector(values[arm], f"{key}.{arm}") for arm in ARM_NAMES]
        )

    def _state_action(self, entry: dict) -> tuple[np.ndarray, np.ndarray]:
        state = np.concatenate(
            [self._arm_vector(entry, "qpos"), self._hand_vector(entry, "gripper_qpos")]
        ).astype(np.float32, copy=False)
        action = np.concatenate(
            [
                self._arm_vector(entry, "qpos_des"),
                self._hand_vector(entry, "gripper_qpos_des"),
            ]
        ).astype(np.float32, copy=False)
        if state.shape != action.shape:
            raise ValueError(
                f"state shape {state.shape} does not match action shape {action.shape}"
            )
        return state, action

    def _source_image(self, entry: dict, camera_name: str) -> np.ndarray:
        images = entry.get("image")
        camera = images.get(camera_name) if isinstance(images, dict) else None
        value = camera.get("color") if isinstance(camera, dict) else None
        if value is None:
            raise ValueError(f"missing image.{camera_name}.color")
        image = np.asarray(value)
        expected_shape = (self.image_height, self.image_width, 3)
        if image.shape != expected_shape:
            raise ValueError(
                f"image.{camera_name}.color has shape {image.shape}; expected {expected_shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"image.{camera_name}.color has dtype {image.dtype}; expected uint8"
            )
        return image

    def _build_frame(self, entry: dict) -> dict:
        state, action = self._state_action(entry)
        frame = {self.state_key: state, self.action_key: action}
        for camera_name in CAMERA_NAMES:
            image = self._source_image(entry, camera_name)
            if self.camera_color_order[camera_name] == "bgr":
                image = image[..., ::-1]
            frame[self._image_key(camera_name)] = image
        return frame

    def _validate_episode(self, episode_data: Iterable[dict]) -> list[dict]:
        if not isinstance(episode_data, list) or not episode_data:
            raise ValueError("PKL must contain a non-empty List[Dict]")
        for index, entry in enumerate(episode_data):
            if not isinstance(entry, dict):
                raise ValueError(f"frame {index} is not a dictionary")
            self._state_action(entry)
            for camera_name in CAMERA_NAMES:
                self._source_image(entry, camera_name)
        return episode_data

    @staticmethod
    def _vector_names(size: int) -> list[str]:
        arm_size = min(7, size // 2)
        names = [
            f"{arm}_arm_joint_{joint}"
            for arm in ("left", "right")
            for joint in range(arm_size)
        ]
        names.extend(("left_hand", "right_hand")[: max(0, size - len(names))])
        names.extend(f"joint_{index}" for index in range(len(names), size))
        return names[:size]

    def _features(self, first_frame: dict) -> dict:
        state_size = int(first_frame[self.state_key].shape[0])
        names = self._vector_names(state_size)
        features = {
            self.state_key: {
                "dtype": "float32",
                "shape": (state_size,),
                "names": names,
            },
            self.action_key: {
                "dtype": "float32",
                "shape": (state_size,),
                "names": names,
            },
        }
        for camera_name in CAMERA_NAMES:
            features[self._image_key(camera_name)] = {
                "dtype": "video" if self.use_videos else "image",
                "shape": (self.image_height, self.image_width, 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def _ensure_dataset(self, first_frame: dict) -> None:
        if self.dataset is not None:
            self._validate_existing_dataset(first_frame)
            return
        if self.root.exists():
            raise FileExistsError(
                f"dataset already exists; use a new --dataset-name or remove it manually: {self.root}"
            )
        self.root.parent.mkdir(parents=True, exist_ok=True)
        _debug("阶段 5/8：准备数据集", f"创建新的 LeRobot v2.1 数据集：{self.root}")
        self.dataset = self._dataset_cls.create(
            repo_id=self.repo_id,
            root=self.root,
            fps=self.fps,
            robot_type="tiangong",
            features=self._features(first_frame),
            use_videos=self.use_videos,
            image_writer_threads=0,
        )
        configure_direct_video = getattr(self.dataset, "configure_direct_video", None)
        if callable(configure_direct_video):
            configure_direct_video(
                bitrate=self.direct_video_bitrate,
            )
        _debug("阶段 5/8：准备数据集", "数据集创建完成，NumPy 直接视频编码已初始化")

    def _validate_existing_dataset(self, first_frame: dict) -> None:
        if int(self.dataset.fps) != self.fps:
            raise ValueError(
                f"existing dataset FPS is {self.dataset.fps}; expected {self.fps}"
            )
        expected_features = self._features(first_frame)
        expected_keys = set(expected_features)
        existing_features = getattr(self.dataset, "features", None)
        if existing_features is None:
            info_path = self.root / "meta" / "info.json"
            with info_path.open("r", encoding="utf-8") as stream:
                existing_features = (json.load(stream) or {}).get("features")
        if existing_features is None or not expected_keys.issubset(existing_features):
            raise ValueError(
                "existing LeRobot dataset is missing configured feature keys"
            )
        for key, expected in expected_features.items():
            existing = existing_features[key]
            if existing.get("dtype") != expected["dtype"] or tuple(
                existing.get("shape", ())
            ) != tuple(expected["shape"]):
                raise ValueError(
                    f"existing LeRobot feature {key!r} has an incompatible dtype or shape"
                )

    @property
    def next_episode_index(self) -> int:
        if self.dataset is not None:
            return int(self.dataset.num_episodes)
        if self.root.exists():
            raise FileExistsError(
                f"dataset already exists; use a new --dataset-name or remove it manually: {self.root}"
            )
        return 0

    def _metadata_episode_length(self, episode_index: int) -> int | None:
        meta = getattr(self.dataset, "meta", None)
        episodes = getattr(meta, "episodes", None)
        if episodes is not None:
            try:
                episode = episodes[episode_index]
            except (KeyError, IndexError, TypeError):
                episode = None
            if isinstance(episode, dict):
                for key in ("length", "num_frames"):
                    if key in episode:
                        return int(episode[key])

        episodes_path = self.root / "meta" / "episodes.jsonl"
        if episodes_path.is_file():
            with episodes_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    value = json.loads(line)
                    if int(value.get("episode_index", -1)) != episode_index:
                        continue
                    for key in ("length", "num_frames"):
                        if key in value:
                            return int(value[key])
        return None

    def verify_episode(self, episode_index: int, frame_count: int) -> bool:
        if self.dataset is None:
            return False
        if int(self.dataset.num_episodes) <= episode_index:
            return False
        saved_length = self._metadata_episode_length(episode_index)
        return saved_length == frame_count

    def write_episode(self, episode_data: list[dict]) -> EpisodeWriteResult:
        total_frames = len(episode_data)
        stage_started_at = time.monotonic()
        _debug("阶段 4/8：数据校验", f"开始校验 {total_frames} 帧的状态、动作和三路图像")
        self._validate_episode(episode_data)
        first_frame = self._build_frame(episode_data[0])
        state_size = int(first_frame[self.state_key].shape[0])
        _debug(
            "阶段 4/8：数据校验",
            f"校验通过：状态/动作 {state_size} 维，图像 "
            f"{self.image_width}×{self.image_height}，用时 {time.monotonic() - stage_started_at:.1f} 秒",
        )

        self._ensure_dataset(first_frame)
        episode_index = int(self.dataset.num_episodes)
        try:
            stage_started_at = time.monotonic()
            _debug(
                "阶段 6/8：逐帧写入",
                f"开始向 episode {episode_index} 写入 {total_frames} 帧，"
                "三路图像保留为 NumPy 缓冲，不生成临时 PNG",
            )
            self.dataset.add_frame(first_frame, task=self.task)
            progress_step = max(1, total_frames // 10)
            next_progress = progress_step
            for frame_index, entry in enumerate(episode_data[1:], start=2):
                self.dataset.add_frame(self._build_frame(entry), task=self.task)
                if frame_index >= next_progress or frame_index == total_frames:
                    elapsed = time.monotonic() - stage_started_at
                    _debug(
                        "阶段 6/8：逐帧写入",
                        f"episode {episode_index} 已提交 {frame_index}/{total_frames} 帧 "
                        f"({frame_index / total_frames:.0%})，本阶段用时 {elapsed:.1f} 秒",
                    )
                    next_progress += progress_step

            _debug(
                "阶段 7/8：保存 Episode",
                f"逐帧提交完成，开始生成 Parquet 并编码三路 MP4；该阶段可能耗时较长",
            )
            save_started_at = time.monotonic()
            self.dataset.save_episode()
            _debug(
                "阶段 7/8：保存 Episode",
                f"LeRobot save_episode 完成，用时 {time.monotonic() - save_started_at:.1f} 秒",
            )
        except Exception:
            _debug(
                "异常清理",
                f"episode {episode_index} 写入失败，正在清理未完成的 episode buffer",
            )
            clear_buffer = getattr(self.dataset, "clear_episode_buffer", None)
            if callable(clear_buffer):
                try:
                    episode_buffer = getattr(self.dataset, "episode_buffer", None)
                    if isinstance(episode_buffer, dict):
                        episode_buffer["episode_index"] = episode_index
                    clear_buffer()
                except Exception as cleanup_exc:
                    _debug(
                        "异常清理",
                        f"清理 episode {episode_index} 时又发生异常："
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}；"
                        "将保留并继续抛出原始保存异常",
                    )
            raise

        _debug(
            "阶段 8/8：落盘核验",
            f"核验 episode {episode_index} 的元数据和帧数是否等于 {total_frames}",
        )
        if not self.verify_episode(episode_index, len(episode_data)):
            raise RuntimeError(
                f"LeRobot episode {episode_index} failed post-save verification"
            )
        _debug("阶段 8/8：落盘核验", f"episode {episode_index} 核验成功，可以提交完成状态")
        return EpisodeWriteResult(episode_index, len(episode_data), self.root)

    def close(self) -> None:
        if self.dataset is None:
            return
        if getattr(self.dataset, "image_writer", None) is None:
            return
        stop_writer = getattr(self.dataset, "stop_image_writer", None)
        if callable(stop_writer):
            _debug("资源关闭", "正在停止 LeRobot 图像写入线程")
            stop_writer()
