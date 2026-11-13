from __future__ import annotations

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


def encode_rgb_frames_with_av1(
    frames: Sequence[np.ndarray],
    video_path: Path | str,
    *,
    fps: int,
    label: str,
    width: int,
    height: int,
) -> None:
    """Encode RGB NumPy frames using LeRobot 0.3.3 default AV1 settings."""
    import av
    if not frames:
        raise ValueError("cannot encode an empty video")
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("FPS and image dimensions must be positive")
    av.codec.Codec("libsvtav1", "w")
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = video_path.with_name(f".{video_path.stem}.encoding.mp4")
    progress = TerminalProgressBar(label)
    try:
        with av.open(str(temporary), "w") as output:
            stream = output.add_stream("libsvtav1", fps, options={"g": "2", "crf": "30"})
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            for index, frame in enumerate(frames, 1):
                frame = np.asarray(frame)
                if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                    raise ValueError("video frame must be uint8 HWC with consistent dimensions")
                value = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
                for packet in stream.encode(value):
                    output.mux(packet)
                progress.update(index, len(frames))
            for packet in stream.encode():
                output.mux(packet)
        temporary.replace(video_path)
    except BaseException:
        progress.break_line()
        temporary.unlink(missing_ok=True)
        raise
