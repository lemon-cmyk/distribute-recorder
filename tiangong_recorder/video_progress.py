from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np


class TerminalProgressBar:
    def __init__(
        self,
        label: str,
        *,
        stream: TextIO | None = None,
        width: int = 30,
    ) -> None:
        self.label = label
        self.stream = stream or sys.stdout
        self.width = width
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self._last_logged_bucket = -1
        self._last_rendered_percent = -1

    def update(self, current: int, total: int, *, force: bool = False) -> None:
        total = max(1, int(total))
        current = min(max(0, int(current)), total)
        percent = current / total
        completed = min(self.width, int(percent * self.width))
        bar = "█" * completed + "░" * (self.width - completed)
        message = (
            f"[LeRobot转换][阶段 7/8][MP4编码][{self.label}] "
            f"[{bar}] {percent:6.1%} {current}/{total}"
        )

        if self.interactive:
            rendered_percent = int(percent * 100)
            if (
                not force
                and current != total
                and rendered_percent <= self._last_rendered_percent
            ):
                return
            ending = "\n" if current == total else ""
            print(f"\r{message}", end=ending, file=self.stream, flush=True)
            self._last_rendered_percent = rendered_percent
            return

        bucket = int(percent * 10)
        if force or bucket > self._last_logged_bucket:
            print(message, file=self.stream, flush=True)
            self._last_logged_bucket = bucket

    def break_line(self) -> None:
        if self.interactive:
            print(file=self.stream, flush=True)


class HardwareVideoEncodingError(RuntimeError):
    pass


def _rgb_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    value = np.asarray(frame)
    expected_shape = (height, width, 3)
    if value.shape != expected_shape or value.dtype != np.uint8:
        raise ValueError(
            f"video frame must be uint8 with shape {expected_shape}; "
            f"got {value.dtype} {value.shape}"
        )
    return np.ascontiguousarray(value)


def _stop_subprocess(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def encode_rgb_frames_with_jetson_h264(
    frames: Sequence[np.ndarray],
    video_path: Path | str,
    *,
    fps: int,
    label: str,
    width: int,
    height: int,
    bitrate: int,
) -> None:
    """Stream RGB NumPy frames into the Jetson nvv4l2 H.264 encoder."""
    if not frames:
        raise ValueError("cannot encode an empty video")
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.unlink(missing_ok=True)

    command = [
        "gst-launch-1.0",
        "-q",
        "fdsrc",
        "fd=0",
        "!",
        "rawvideoparse",
        "format=rgb",
        f"width={width}",
        f"height={height}",
        f"framerate={fps}/1",
        "!",
        "videoconvert",
        "!",
        "video/x-raw,format=I420",
        "!",
        "nvvidconv",
        "!",
        "video/x-raw(memory:NVMM),format=NV12",
        "!",
        "nvv4l2h264enc",
        f"bitrate={bitrate}",
        "control-rate=0",
        "preset-level=2",
        "maxperf-enable=true",
        "insert-sps-pps=true",
        f"iframeinterval={fps}",
        "!",
        "h264parse",
        "!",
        "qtmux",
        "!",
        "filesink",
        f"location={video_path}",
    ]
    progress = TerminalProgressBar(label)
    progress.update(0, len(frames), force=True)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    try:
        assert process.stdin is not None
        for index, frame in enumerate(frames, start=1):
            process.stdin.write(_rgb_frame(frame, width, height).tobytes())
            progress.update(index, len(frames))
        process.stdin.close()
        return_code = process.wait()
        stdout = (
            process.stdout.read().decode("utf-8", errors="replace")
            if process.stdout
            else ""
        )
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr
            else ""
        )
    except KeyboardInterrupt:
        progress.break_line()
        _stop_subprocess(process)
        video_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        progress.break_line()
        _stop_subprocess(process)
        stdout = (
            process.stdout.read().decode("utf-8", errors="replace")
            if process.stdout
            else ""
        )
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace")
            if process.stderr
            else ""
        )
        video_path.unlink(missing_ok=True)
        details = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
        raise HardwareVideoEncodingError(
            f"Jetson H.264 pipeline failed: {type(exc).__name__}: {exc}; {details[-2000:]}"
        ) from exc

    if return_code != 0 or not video_path.is_file():
        video_path.unlink(missing_ok=True)
        details = "\n".join(part.strip() for part in (stdout, stderr) if part.strip())
        raise HardwareVideoEncodingError(
            f"Jetson H.264 encoder failed with exit code {return_code}: {details[-2000:]}"
        )

