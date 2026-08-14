import json
import shutil

import numpy as np
import pytest

from tiangong_recorder.direct_lerobot_dataset import (
    compute_numpy_episode_stats,
    compute_target_norm_stats,
)
from tiangong_recorder.lerobot_writer import OUTPUT_CAMERA_NAMES, LeRobotV21Writer


def test_numpy_image_stats_match_lerobot_shapes_and_normalization():
    images = [
        np.zeros((4, 6, 3), dtype=np.uint8),
        np.full((4, 6, 3), 255, dtype=np.uint8),
    ]
    features = {
        "observation.images.head": {
            "dtype": "video",
            "shape": (4, 6, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["a", "b"],
        },
    }
    data = {
        "observation.images.head": images,
        "observation.state": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }

    stats = compute_numpy_episode_stats(data, features)

    image_stats = stats["observation.images.head"]
    assert image_stats["min"].shape == (3, 1, 1)
    assert image_stats["max"].shape == (3, 1, 1)
    assert image_stats["mean"].shape == (3, 1, 1)
    assert np.allclose(image_stats["min"], 0.0)
    assert np.allclose(image_stats["max"], 1.0)
    assert np.allclose(image_stats["mean"], 0.5)
    assert image_stats["count"].tolist() == [2]
    assert stats["observation.state"]["mean"].tolist() == [2.0, 3.0]


def _hardware_entry(index: int) -> dict:
    head = np.full((480, 640, 3), (index * 20) % 255, dtype=np.uint8)
    wrist = np.zeros((480, 640, 3), dtype=np.uint8)
    wrist[..., 0] = 10 + index
    wrist[..., 1] = 20
    wrist[..., 2] = 30
    return {
        "qpos": {"left_arm": np.arange(7), "right_arm": np.arange(7, 14)},
        "qpos_des": {
            "left_arm": np.arange(14, 21),
            "right_arm": np.arange(21, 28),
        },
        "gripper_qpos": {"left_arm": [0.1], "right_arm": [0.2]},
        "gripper_qpos_des": {"left_arm": [0.3], "right_arm": [0.4]},
        "image": {
            "head": {"color": head},
            "left_wrist": {"color": wrist},
            "right_wrist": {"color": wrist.copy()},
        },
    }


def test_target_norm_stats_uses_expected_keys_and_quantiles():
    states = np.stack(
        [np.arange(16, dtype=np.float32), np.arange(16, dtype=np.float32) + 10]
    )
    actions = states + 100

    stats = compute_target_norm_stats(states, actions)

    assert set(stats) == {"state", "actions"}
    assert set(stats["state"]) == {"mean", "std", "q01", "q99"}
    assert np.allclose(stats["state"]["mean"], np.arange(16) + 5)
    assert np.allclose(stats["actions"]["q99"], np.arange(16) + 109.9)


def test_direct_writer_hardware_av1_is_lerobot_readable(tmp_path):
    pytest.importorskip("lerobot")
    av = pytest.importorskip("av")
    if shutil.which("gst-launch-1.0") is None:
        pytest.skip("GStreamer is not installed")
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not installed")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "direct_dataset"
    writer = LeRobotV21Writer(
        dataset_root=root,
        repo_id="local/direct-test",
        task="hardware encoding test",
        fps=50,
        use_videos=True,
        image_width=640,
        image_height=480,
        direct_video_bitrates={name: 1_000_000 for name in OUTPUT_CAMERA_NAMES},
    )

    frame_count = 12
    result = writer.write_episode(
        [_hardware_entry(index) for index in range(frame_count)]
    )
    writer.close()

    assert result.episode_index == 0
    assert result.frame_count == frame_count
    assert not (root / "images").exists()
    video_paths = list(root.glob("videos/**/*.mp4"))
    assert len(video_paths) == len(OUTPUT_CAMERA_NAMES)
    assert not list(root.glob("**/*.ivf"))
    assert len(list(root.glob("data/**/*.parquet"))) == 1
    assert (root / "norm_stats.json").is_file()
    with (root / "meta" / "info.json").open(encoding="utf-8") as stream:
        info = json.load(stream)
    assert info["robot_type"] == "Tiangong2"
    for camera_name in OUTPUT_CAMERA_NAMES:
        assert (
            info["features"][f"observation.images.{camera_name}"]["info"]["video.codec"]
            == "av1"
        )
    for video_path in video_paths:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            assert len(stream.codec_context.extradata or b"") >= 17
            keyframe_count = sum(
                packet.is_keyframe for packet in container.demux(stream)
            )
            assert keyframe_count >= frame_count // 2

    import pyarrow.parquet as pq

    parquet_file = next(root.glob("data/**/*.parquet"))
    table = pq.read_table(parquet_file)
    assert table.column_names == [
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    assert str(table.schema.field("action").type).startswith("list<")
    assert not str(table.schema.field("action").type).startswith("fixed_size_list")
    assert b"pandas" in (table.schema.metadata or {})

    loaded = LeRobotDataset(
        repo_id="local/direct-test",
        root=root,
        video_backend="pyav",
    )
    assert loaded.num_episodes == 1
    assert loaded.num_frames == frame_count
    for frame_index in (5, 1, 11, 2, 0):
        frame = loaded[frame_index]
        assert tuple(frame["observation.state"].shape) == (16,)
        assert tuple(frame["observation.images.front"].shape) == (3, 480, 640)
