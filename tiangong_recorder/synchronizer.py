from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    timestamp_ns: int
    image: np.ndarray
    encoding: str


class LiveEpisodeSynchronizer:
    """Causally aligns ordered camera frames with ordered robot states."""

    INTERNAL_STATE_KEYS = {
        "_episode_id",
        "_frame_index",
        "_capture_time_ns",
    }

    def __init__(self, camera_names: Iterable[str]):
        self.camera_names = tuple(camera_names)
        self.frames: Dict[str, Deque[CameraFrame]] = {
            name: deque() for name in self.camera_names
        }
        self.pending_states: Deque[dict] = deque()
        self.episode_data: list[dict] = []
        self.dropped_out_of_order_images: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }
        self.fallback_matches: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }
        self.alignment_count: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }
        self.alignment_abs_sum_ns: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }
        self.alignment_abs_max_ns: Dict[str, int] = {
            name: 0 for name in self.camera_names
        }

    def add_camera_frame(self, camera_name: str, frame: CameraFrame) -> bool:
        queue = self.frames[camera_name]
        if queue and frame.timestamp_ns < queue[-1].timestamp_ns:
            self.dropped_out_of_order_images[camera_name] += 1
            return False
        queue.append(frame)
        return True

    def add_state(self, state: dict) -> None:
        timestamp_ns = int(state["_capture_time_ns"])
        if self.pending_states:
            previous = int(self.pending_states[-1]["_capture_time_ns"])
            if timestamp_ns < previous:
                raise ValueError("state timestamps must be monotonic")
        self.pending_states.append(state)

    def cameras_ready(self) -> bool:
        return all(self.frames[name] for name in self.camera_names)

    def can_match(self, state_timestamp_ns: int) -> bool:
        return self.cameras_ready() and all(
            self.frames[name][-1].timestamp_ns >= state_timestamp_ns
            for name in self.camera_names
        )

    @staticmethod
    def _select_frame(
        queue: Deque[CameraFrame],
        state_timestamp_ns: int,
    ) -> tuple[CameraFrame, bool]:
        for frame in reversed(queue):
            if frame.timestamp_ns <= state_timestamp_ns:
                return frame, False
        return queue[0], True

    @staticmethod
    def _prune_before_selected(
        queue: Deque[CameraFrame],
        selected: CameraFrame,
    ) -> None:
        while len(queue) > 1 and queue[0] is not selected:
            queue.popleft()

    def merge_ready(self, force: bool = False) -> int:
        merged = 0
        while self.pending_states:
            state = self.pending_states[0]
            state_timestamp_ns = int(state["_capture_time_ns"])
            if not self.cameras_ready():
                break
            if not force and not self.can_match(state_timestamp_ns):
                break

            selected_frames: Dict[str, CameraFrame] = {}
            for camera_name in self.camera_names:
                selected, used_fallback = self._select_frame(
                    self.frames[camera_name],
                    state_timestamp_ns,
                )
                selected_frames[camera_name] = selected
                if used_fallback:
                    self.fallback_matches[camera_name] += 1

                delta_ns = state_timestamp_ns - selected.timestamp_ns
                abs_delta_ns = abs(delta_ns)
                self.alignment_count[camera_name] += 1
                self.alignment_abs_sum_ns[camera_name] += abs_delta_ns
                self.alignment_abs_max_ns[camera_name] = max(
                    self.alignment_abs_max_ns[camera_name],
                    abs_delta_ns,
                )

            entry = {
                key: value
                for key, value in state.items()
                if key not in self.INTERNAL_STATE_KEYS
            }
            entry["image"] = {
                camera_name: {
                    "color": selected_frames[camera_name].image.copy(),
                }
                for camera_name in self.camera_names
            }
            self.episode_data.append(entry)
            self.pending_states.popleft()
            merged += 1

            for camera_name, selected in selected_frames.items():
                self._prune_before_selected(self.frames[camera_name], selected)

        return merged

    def alignment_summary(self) -> dict:
        summary = {}
        for name in self.camera_names:
            count = self.alignment_count[name]
            average_ms = (
                self.alignment_abs_sum_ns[name] / count / 1_000_000.0
                if count
                else 0.0
            )
            summary[name] = {
                "count": count,
                "mean_abs_delta_ms": average_ms,
                "max_abs_delta_ms": self.alignment_abs_max_ns[name]
                / 1_000_000.0,
                "fallback_matches": self.fallback_matches[name],
                "dropped_out_of_order_images": self.dropped_out_of_order_images[
                    name
                ],
            }
        return summary

