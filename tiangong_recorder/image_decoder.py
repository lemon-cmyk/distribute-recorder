from __future__ import annotations

import time

import numpy as np

from .synchronizer import CameraFrame


SUPPORTED_ENCODINGS = {"rgb8", "bgr8"}


def image_timestamp_ns(message, fallback_ns: int | None = None) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if timestamp_ns > 0:
            return timestamp_ns
    return int(fallback_ns if fallback_ns is not None else time.time_ns())


def decode_ros_image(
    message,
    expected_width: int,
    expected_height: int,
    fallback_timestamp_ns: int | None = None,
) -> CameraFrame:
    width = int(message.width)
    height = int(message.height)
    encoding = str(message.encoding).lower()
    step = int(message.step)

    if width != expected_width or height != expected_height:
        raise ValueError(
            f"unexpected image size {width}x{height}; "
            f"expected {expected_width}x{expected_height}"
        )
    if encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(f"unsupported image encoding: {encoding}")

    row_bytes = width * 3
    if step < row_bytes:
        raise ValueError(f"invalid image step {step}; expected at least {row_bytes}")

    flat = np.frombuffer(message.data, dtype=np.uint8)
    expected_size = height * step
    if flat.size < expected_size:
        raise ValueError(
            f"image payload is truncated: {flat.size} bytes; expected {expected_size}"
        )

    image = (
        flat[:expected_size]
        .reshape(height, step)[:, :row_bytes]
        .reshape(height, width, 3)
        .copy()
    )
    return CameraFrame(
        timestamp_ns=image_timestamp_ns(message, fallback_timestamp_ns),
        image=image,
        encoding=encoding,
    )

