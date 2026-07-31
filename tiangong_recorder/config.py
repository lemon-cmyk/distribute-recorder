from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


@dataclass(frozen=True)
class RecorderConfig:
    control_bind_addr: str = "tcp://0.0.0.0:5560"
    advertise_host: str = "192.168.41.2"
    state_port_min: int = 5600
    state_port_max: int = 5699
    output_dir: str = "/home/nvidia/teleop_logs"
    camera_topics: Dict[str, str] = field(
        default_factory=lambda: {
            "head": "/camera/color/image_raw",
            "left_wrist": "/camera/left/image_rgb",
            "right_wrist": "/camera/right/image_rgb",
        }
    )
    image_width: int = 640
    image_height: int = 480
    startup_timeout_s: float = 10.0
    camera_stale_timeout_s: float = 1.0
    state_drain_timeout_s: float = 5.0
    tail_wait_timeout_s: float = 1.0
    state_receive_hwm: int = 10000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RecorderConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        return cls(**values)

    def validate(self) -> None:
        expected_cameras = {"head", "left_wrist", "right_wrist"}
        if set(self.camera_topics) != expected_cameras:
            raise ValueError(f"camera_topics must contain exactly {sorted(expected_cameras)}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.state_port_min <= 0 or self.state_port_max < self.state_port_min:
            raise ValueError("invalid state port range")
        if self.startup_timeout_s <= 0 or self.state_drain_timeout_s <= 0:
            raise ValueError("timeouts must be positive")

