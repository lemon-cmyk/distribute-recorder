from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import pickle
import re
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .conversion_state import ConversionRecord, ConversionState
from .lerobot_writer import CAMERA_NAMES, LeRobotV21Writer


_DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _debug(stage: str, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[LeRobot转换][{timestamp}][{stage}] {message}", flush=True)


@dataclass(frozen=True)
class ConverterConfig:
    input_dir: str = "/home/nvidia/teleop_logs"
    dataset_root: str = "/home/nvidia/lerobot_datasets/tiangong_teleop"
    state_db: str = "/home/nvidia/lerobot_datasets/.tiangong_converter.sqlite3"
    lock_file: str = "/home/nvidia/lerobot_datasets/.tiangong_converter.lock"
    repo_id: str = "local/tiangong_teleop"
    fps: int = 50
    task: str = "Teleoperate the Tiangong robot"
    use_videos: bool = True
    delete_source_pkl: bool = True
    poll_interval_s: float = 2.0
    retry_errors: bool = False
    image_width: int = 640
    image_height: int = 480
    state_key: str = "observation.state"
    action_key: str = "action"
    image_prefix: str = "observation.images"
    image_writer_threads: int = 2
    direct_video_bitrate: int = 8_000_000
    camera_color_order: dict[str, str] = field(
        default_factory=lambda: {
            "head": "rgb",
            "left_wrist": "bgr",
            "right_wrist": "bgr",
        }
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConverterConfig":
        with Path(path).expanduser().open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        if not isinstance(values, dict):
            raise ValueError("converter config must be a YAML mapping")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown converter configuration keys: {sorted(unknown)}")
        if "PROMPT" in os.environ:
            values["task"] = os.environ["PROMPT"]
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        input_dir = Path(self.input_dir).expanduser().resolve()
        dataset_root = Path(self.dataset_root).expanduser().resolve()
        if input_dir == dataset_root or input_dir in dataset_root.parents:
            raise ValueError("dataset_root must not be inside input_dir")
        for name in ("repo_id", "task", "state_key", "action_key", "image_prefix"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if not isinstance(self.fps, int) or isinstance(self.fps, bool) or self.fps <= 0:
            raise ValueError("fps must be a positive integer")
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if self.image_writer_threads < 0:
            raise ValueError("image_writer_threads must not be negative")
        if self.direct_video_bitrate <= 0:
            raise ValueError("direct_video_bitrate must be positive")
        if set(self.camera_color_order) != set(CAMERA_NAMES):
            raise ValueError(f"camera_color_order must contain exactly {CAMERA_NAMES}")


def _apply_cli_overrides(
    config: ConverterConfig,
    *,
    dataset_name: str | None,
    task: str | None,
) -> ConverterConfig:
    """Apply command-line dataset identity without mixing conversion state."""
    overrides: dict[str, Any] = {}

    if dataset_name is not None:
        normalized_name = dataset_name.strip()
        if not _DATASET_NAME_PATTERN.fullmatch(normalized_name):
            raise ValueError(
                "--dataset-name must start with a letter or digit and contain only "
                "letters, digits, '.', '_' or '-' (maximum 128 characters)"
            )

        dataset_parent = Path(config.dataset_root).expanduser().parent
        state_parent = Path(config.state_db).expanduser().parent
        lock_parent = Path(config.lock_file).expanduser().parent
        repo_namespace = (
            config.repo_id.rsplit("/", 1)[0] if "/" in config.repo_id else "local"
        )
        overrides.update(
            dataset_root=str(dataset_parent / normalized_name),
            state_db=str(state_parent / f".{normalized_name}_converter.sqlite3"),
            lock_file=str(lock_parent / f".{normalized_name}_converter.lock"),
            repo_id=f"{repo_namespace}/{normalized_name}",
        )

    if task is not None:
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("--task must not be empty")
        overrides["task"] = normalized_task

    updated = replace(config, **overrides)
    updated.validate()
    return updated


class PklToLeRobotConverter:
    def __init__(
        self,
        config: ConverterConfig,
        *,
        writer: LeRobotV21Writer | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.input_dir = Path(config.input_dir).expanduser().resolve()
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.state = ConversionState(config.state_db)
        self.writer = writer or LeRobotV21Writer(
            dataset_root=config.dataset_root,
            repo_id=config.repo_id,
            task=config.task,
            fps=config.fps,
            use_videos=config.use_videos,
            image_width=config.image_width,
            image_height=config.image_height,
            state_key=config.state_key,
            action_key=config.action_key,
            image_prefix=config.image_prefix,
            camera_color_order=config.camera_color_order,
            image_writer_threads=config.image_writer_threads,
            direct_video_bitrate=config.direct_video_bitrate,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _load_pickle(path: Path) -> list[dict[str, Any]]:
        with path.open("rb") as stream:
            value = pickle.load(stream)
        if not isinstance(value, list) or not value:
            raise ValueError("PKL must contain a non-empty List[Dict]")
        return value

    def _delete_source(self, path: Path, expected_sha256: str) -> None:
        if not path.exists():
            self.state.mark_deleted(path)
            _debug("源文件处理", f"源 PKL 已不存在，仅更新删除状态：{path}")
            return
        _debug("源文件处理", f"删除前再次校验 SHA256：{path.name}")
        actual_sha256 = self._sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "source PKL changed after conversion; refusing to delete it"
            )
        path.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self.state.mark_deleted(path)
        _debug("源文件处理", f"已删除转换成功的源 PKL：{path}")

    def _reconcile_record(
        self,
        path: Path,
        source_sha256: str,
        record: ConversionRecord | None,
    ) -> str | None:
        if record is None:
            _debug("历史状态", "没有历史转换记录，将作为新文件转换")
            return None
        if record.source_sha256 != source_sha256:
            if record.status == "done":
                raise RuntimeError(
                    "a previously converted source path now contains different data"
                )
            _debug("历史状态", "文件内容与旧记录不同，将重新转换")
            return None
        if record.status == "done":
            _debug("历史状态", "该文件已经转换完成，不重复写入 episode")
            if self.config.delete_source_pkl and not record.source_deleted:
                self._delete_source(path, source_sha256)
            return "already_done"

        has_target = (
            record.target_episode_index is not None and record.frame_count is not None
        )
        if record.status in {"converting", "error"} and has_target:
            target_index = int(record.target_episode_index)
            frame_count = int(record.frame_count)
            _debug(
                "中断恢复",
                f"检查之前预留的 episode {target_index}，预期 {frame_count} 帧",
            )
            if self.writer.verify_episode(target_index, frame_count):
                _debug("中断恢复", f"episode {target_index} 已完整落盘，恢复为完成状态")
                self.state.mark_done(path, target_index, frame_count)
                if self.config.delete_source_pkl:
                    self._delete_source(path, source_sha256)
                return "recovered_done"
            if self.writer.next_episode_index > target_index:
                raise RuntimeError(
                    "LeRobot advanced beyond the reserved episode without matching metadata; "
                    "manual inspection is required"
                )

        if record.status == "error" and not self.config.retry_errors:
            _debug("历史状态", "该文件上次转换失败；未启用 --retry-errors，本次跳过")
            return "previous_error"
        return None

    def process_file(self, source_path: str | Path) -> str:
        path = Path(source_path).expanduser().resolve()
        if path.parent != self.input_dir or path.suffix != ".pkl" or not path.is_file():
            raise ValueError(f"not a final PKL in {self.input_dir}: {path}")

        record = self.state.get(path)
        if (
            record is not None
            and record.status == "error"
            and not self.config.retry_errors
        ):
            _debug("文件跳过", f"{path.name} 上次转换失败，等待 --retry-errors")
            return "previous_error"

        source_sha256 = ""
        target_index = None
        file_started_at = time.monotonic()
        try:
            size_gib = path.stat().st_size / (1024 ** 3)
            _debug("文件开始", f"开始处理 {path.name}，大小 {size_gib:.2f} GiB")

            stage_started_at = time.monotonic()
            _debug("阶段 1/8：完整性校验", "开始计算源 PKL 的 SHA256")
            source_sha256 = self._sha256(path)
            _debug(
                "阶段 1/8：完整性校验",
                f"SHA256 完成，用时 {time.monotonic() - stage_started_at:.1f} 秒，"
                f"摘要 {source_sha256[:12]}...",
            )
            reconciled = self._reconcile_record(path, source_sha256, record)
            if reconciled is not None:
                return reconciled

            stage_started_at = time.monotonic()
            _debug("阶段 2/8：读取 PKL", "开始反序列化完整样本，请等待")
            episode_data = self._load_pickle(path)
            _debug(
                "阶段 2/8：读取 PKL",
                f"读取完成，共 {len(episode_data)} 帧，用时 "
                f"{time.monotonic() - stage_started_at:.1f} 秒",
            )

            target_index = self.writer.next_episode_index
            _debug(
                "阶段 3/8：预留序号",
                f"为 {path.name} 预留 LeRobot episode {target_index}，并写入 SQLite 状态",
            )
            self.state.begin(
                path,
                source_sha256,
                len(episode_data),
                target_index,
                self.config.dataset_root,
            )
            result = self.writer.write_episode(episode_data)
            if result.episode_index != target_index or result.frame_count != len(
                episode_data
            ):
                raise RuntimeError(
                    "LeRobot writer returned unexpected episode metadata"
                )
            self.state.mark_done(path, result.episode_index, result.frame_count)
            _debug(
                "转换完成",
                f"{path.name} -> episode {result.episode_index}，"
                f"共 {result.frame_count} 帧，总用时 {time.monotonic() - file_started_at:.1f} 秒",
            )
            if self.config.delete_source_pkl:
                try:
                    self._delete_source(path, source_sha256)
                except Exception:
                    traceback.print_exc()
                    return "converted_not_deleted"
            return "converted"
        except Exception as exc:
            current = self.state.get(path)
            if current is None or current.status != "done":
                release_reservation = False
                if target_index is not None:
                    try:
                        release_reservation = (
                            self.writer.next_episode_index == target_index
                        )
                    except Exception:
                        pass
                self.state.mark_error(
                    path,
                    source_sha256,
                    f"{type(exc).__name__}: {exc}",
                    release_reservation=release_reservation,
                )
            _debug(
                "转换失败",
                f"{path.name} 处理失败：{type(exc).__name__}: {exc}",
            )
            traceback.print_exc()
            return "error"

    def scan_once(self) -> list[tuple[Path, str]]:
        paths = sorted(self.input_dir.glob("*.pkl"))
        _debug("目录扫描", f"在 {self.input_dir} 找到 {len(paths)} 个正式 PKL")
        results = []
        for index, path in enumerate(paths, start=1):
            _debug("目录扫描", f"开始处理第 {index}/{len(paths)} 个文件：{path.name}")
            result = self.process_file(path)
            results.append((path, result))
            if result == "error":
                _debug("目录扫描", "当前数据集出现转换错误，停止处理后续 PKL")
                break
        summary: dict[str, int] = {}
        for _path, result in results:
            summary[result] = summary.get(result, 0) + 1
        _debug("目录扫描", f"本轮扫描结束，结果统计：{summary}")
        return results

    def close(self, *, wait_for_writer: bool = True) -> None:
        if wait_for_writer:
            self.writer.close()
        self.state.close()


def _acquire_singleton_lock(path: str | Path):
    lock_path = Path(path).expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError(f"another converter holds {lock_path}") from exc
    stream.seek(0)
    stream.truncate()
    stream.write(f"{os.getpid()}\n")
    stream.flush()
    os.fsync(stream.fileno())
    return stream


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Tiangong PKLs into LeRobot v2.1"
    )
    parser.add_argument("--config", default="config/converter.yaml")
    parser.add_argument("--once", action="store_true", help="scan once and exit")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--keep-source", action="store_true")
    parser.add_argument(
        "--dataset-name",
        help="LeRobot dataset directory name; also derives state DB, lock, and repo_id",
    )
    parser.add_argument(
        "--task",
        help="task description stored in the LeRobot dataset",
    )
    args = parser.parse_args(argv)

    config = ConverterConfig.from_yaml(args.config)
    config = _apply_cli_overrides(
        config,
        dataset_name=args.dataset_name,
        task=args.task,
    )
    if args.retry_errors:
        config = replace(config, retry_errors=True)
    if args.keep_source:
        config = replace(config, delete_source_pkl=False)

    dataset_root = Path(config.dataset_root).expanduser()
    if dataset_root.exists() or dataset_root.is_symlink():
        _debug(
            "启动拒绝",
            f"同名数据集已经存在：{dataset_root}。" "请使用新的 --dataset-name，或手动删除原目录后再启动。",
        )
        return 2

    lock_stream = _acquire_singleton_lock(config.lock_file)
    _debug("程序启动", f"配置文件：{Path(args.config).expanduser().resolve()}")
    _debug("程序启动", f"输入目录：{Path(config.input_dir).expanduser()}")
    _debug("程序启动", f"输出目录：{Path(config.dataset_root).expanduser()}")
    _debug("程序启动", f"数据集 ID：{config.repo_id}")
    _debug("程序启动", f"任务描述：{config.task}")
    _debug(
        "程序启动",
        f"FPS={config.fps}，视频模式={config.use_videos}，"
        f"保留源 PKL={not config.delete_source_pkl}，重试失败={config.retry_errors}",
    )
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    converter = PklToLeRobotConverter(config)
    exit_code = 0
    interrupted = False
    try:
        while True:
            results = converter.scan_once()
            for path, result in results:
                if result not in {"already_done", "previous_error"}:
                    print(f"Scan result: {path.name}: {result}", flush=True)
            if any(result == "error" for _path, result in results):
                exit_code = 1
                break
            if args.once:
                failed_results = {"error", "previous_error", "converted_not_deleted"}
                exit_code = int(
                    any(result in failed_results for _path, result in results)
                )
                break
            time.sleep(config.poll_interval_s)
    except KeyboardInterrupt:
        interrupted = True
        exit_code = 130
        _debug("停止请求", "收到 Ctrl+C/终止信号，立即中止当前转换并退出")
    finally:
        if interrupted:
            _debug("程序退出", "正在关闭转换状态数据库；不等待后台图像队列")
        else:
            _debug("程序退出", "正在关闭图像写入线程和转换状态数据库")
        converter.close(wait_for_writer=not interrupted)
        lock_stream.close()
        _debug("程序退出", f"转换器已退出，退出码 {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
