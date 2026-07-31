import numpy as np

from tiangong_recorder.synchronizer import CameraFrame, LiveEpisodeSynchronizer


CAMERAS = ("head", "left_wrist", "right_wrist")


def frame(timestamp_ns: int, value: int) -> CameraFrame:
    return CameraFrame(
        timestamp_ns=timestamp_ns,
        image=np.full((2, 2, 3), value, dtype=np.uint8),
        encoding="rgb8",
    )


def test_waits_for_camera_watermark_then_selects_latest_causal_frame():
    synchronizer = LiveEpisodeSynchronizer(CAMERAS)
    for camera_name in CAMERAS:
        synchronizer.add_camera_frame(camera_name, frame(33, 33))
    synchronizer.add_state(
        {
            "_episode_id": "episode",
            "_frame_index": 0,
            "_capture_time_ns": 50,
            "timestamp": 0.0,
            "qpos": {"left_arm": np.array([1.0])},
        }
    )

    assert synchronizer.merge_ready() == 0

    for camera_name in CAMERAS:
        synchronizer.add_camera_frame(camera_name, frame(66, 66))

    assert synchronizer.merge_ready() == 1
    assert len(synchronizer.episode_data) == 1
    entry = synchronizer.episode_data[0]
    assert "_capture_time_ns" not in entry
    assert "_frame_index" not in entry
    for camera_name in CAMERAS:
        assert np.all(entry["image"][camera_name]["color"] == 33)


def test_force_uses_last_available_frame_for_episode_tail():
    synchronizer = LiveEpisodeSynchronizer(CAMERAS)
    for camera_name in CAMERAS:
        synchronizer.add_camera_frame(camera_name, frame(33, 33))
    synchronizer.add_state(
        {
            "_episode_id": "episode",
            "_frame_index": 0,
            "_capture_time_ns": 50,
            "timestamp": 0.0,
        }
    )

    assert synchronizer.merge_ready(force=True) == 1
    for camera_name in CAMERAS:
        assert np.all(
            synchronizer.episode_data[0]["image"][camera_name]["color"] == 33
        )

