import json
from pathlib import Path

import numpy as np
import pytest

from tiangong_recorder.lerobot_writer import TARGET_VECTOR_NAMES, LeRobotV21Writer


class FakeLeRobotDataset:
    instances = []

    def __init__(self, repo_id, root):
        self.repo_id = repo_id
        self.root = Path(root)
        with (self.root / "meta" / "info.json").open(encoding="utf-8") as stream:
            info = json.load(stream)
        self.fps = info["fps"]
        self.features = info["features"]
        self.num_episodes = info["num_episodes"]
        self.frames = []
        self.tasks = []
        type(self).instances.append(self)

    @classmethod
    def create(
        cls,
        *,
        repo_id,
        root,
        fps,
        robot_type,
        features,
        use_videos,
        image_writer_threads,
    ):
        root = Path(root)
        (root / "meta").mkdir(parents=True)
        info = {
            "fps": fps,
            "features": features,
            "num_episodes": 0,
            "robot_type": robot_type,
            "use_videos": use_videos,
            "image_writer_threads": image_writer_threads,
        }
        (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
        return cls(repo_id, root)

    def add_frame(self, frame, task):
        self.frames.append(frame)
        self.tasks.append(task)

    def save_episode(self):
        episode_index = self.num_episodes
        with (self.root / "meta" / "episodes.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps({"episode_index": episode_index, "length": len(self.frames)})
                + "\n"
            )
        self.num_episodes += 1
        with (self.root / "meta" / "info.json").open(encoding="utf-8") as stream:
            info = json.load(stream)
        info["num_episodes"] = self.num_episodes
        (self.root / "meta" / "info.json").write_text(
            json.dumps(info), encoding="utf-8"
        )

    def stop_image_writer(self):
        pass


def make_entry():
    head = np.zeros((2, 3, 3), dtype=np.uint8)
    head[0, 0] = [1, 2, 3]
    wrist = np.zeros((2, 3, 3), dtype=np.uint8)
    wrist[0, 0] = [10, 20, 30]
    return {
        "qpos": {"left_arm": np.arange(7), "right_arm": np.arange(7, 14)},
        "qpos_des": {"left_arm": np.arange(14, 21), "right_arm": np.arange(21, 28)},
        "gripper_qpos": {"left_arm": [0.1], "right_arm": [0.2]},
        "gripper_qpos_des": {"left_arm": [0.3], "right_arm": [0.4]},
        "image": {
            "head": {"color": head},
            "left_wrist": {"color": wrist},
            "right_wrist": {"color": wrist.copy()},
        },
    }


def make_writer(tmp_path):
    return LeRobotV21Writer(
        dataset_root=tmp_path / "dataset",
        repo_id="local/test",
        task="pick object",
        fps=50,
        use_videos=True,
        image_width=3,
        image_height=2,
        _dataset_cls=FakeLeRobotDataset,
    )


def test_writer_maps_state_action_and_camera_color(tmp_path, capsys):
    writer = make_writer(tmp_path)
    result = writer.write_episode([make_entry(), make_entry()])

    assert result.episode_index == 0
    assert result.frame_count == 2
    frame = writer.dataset.frames[0]
    assert frame["observation.state"].shape == (16,)
    assert frame["observation.state"].dtype == np.float32
    assert np.allclose(
        frame["observation.state"],
        [*range(7), 0.1, *range(7, 14), 0.2],
    )
    assert frame["action"].shape == (16,)
    assert np.allclose(
        frame["action"],
        [*range(14, 21), 0.3, *range(21, 28), 0.4],
    )
    assert frame["observation.images.front"][0, 0].tolist() == [1, 2, 3]
    assert frame["observation.images.left_wrist"][0, 0].tolist() == [30, 20, 10]
    assert writer.dataset.tasks == ["pick object", "pick object"]
    assert writer.dataset.features["observation.images.front"]["dtype"] == "video"
    assert list(writer.dataset.features)[:2] == ["action", "observation.state"]
    assert writer.dataset.features["action"]["names"] == list(TARGET_VECTOR_NAMES)
    with (writer.root / "meta" / "info.json").open(encoding="utf-8") as stream:
        info = json.load(stream)
    assert info["robot_type"] == "Tiangong2"
    assert writer.verify_episode(0, 2)
    output = capsys.readouterr().out
    for stage in ("阶段 4/8", "阶段 5/8", "阶段 6/8", "阶段 7/8", "阶段 8/8"):
        assert stage in output


def test_writer_rejects_wrong_image_shape_before_creating_dataset(tmp_path):
    writer = make_writer(tmp_path)
    entry = make_entry()
    entry["image"]["head"]["color"] = np.zeros((1, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="expected"):
        writer.write_episode([entry])
    assert not writer.root.exists()


def test_writer_rejects_state_action_size_mismatch(tmp_path):
    writer = make_writer(tmp_path)
    entry = make_entry()
    entry["gripper_qpos_des"] = None

    with pytest.raises(ValueError, match="missing Tiangong field"):
        writer.write_episode([entry])


def test_writer_rejects_existing_empty_dataset_without_deleting_it(tmp_path):
    writer = make_writer(tmp_path)
    (writer.root / "meta").mkdir(parents=True)
    (writer.root / "images" / "head").mkdir(parents=True)
    (writer.root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 0, "total_frames": 0}), encoding="utf-8"
    )
    stale_file = writer.root / "images" / "head" / "interrupted-frame.bin"
    stale_file.write_bytes(b"incomplete")

    with pytest.raises(FileExistsError, match="use a new --dataset-name"):
        _ = writer.next_episode_index
    assert stale_file.is_file()


def test_writer_rejects_existing_dataset_with_completed_files(tmp_path):
    writer = make_writer(tmp_path)
    (writer.root / "meta").mkdir(parents=True)
    (writer.root / "data" / "chunk-000").mkdir(parents=True)
    (writer.root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 0, "total_frames": 0}), encoding="utf-8"
    )
    (writer.root / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"data")

    with pytest.raises(FileExistsError, match="use a new --dataset-name"):
        _ = writer.next_episode_index
    assert (writer.root / "data" / "chunk-000" / "episode_000000.parquet").is_file()


def test_writer_accepts_lerobot_builtin_feature_keys(tmp_path):
    writer = make_writer(tmp_path)
    writer.write_episode([make_entry()])
    writer.dataset.features["timestamp"] = {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }

    writer._validate_existing_dataset(writer._build_frame(make_entry()))


def test_writer_rejects_existing_feature_shape_change(tmp_path):
    writer = make_writer(tmp_path)
    writer.write_episode([make_entry()])
    writer.dataset.features["observation.state"]["shape"] = [99]

    with pytest.raises(ValueError, match="incompatible"):
        writer._validate_existing_dataset(writer._build_frame(make_entry()))


def test_writer_cleanup_does_not_mask_original_save_error(tmp_path, capsys):
    class FailingDataset(FakeLeRobotDataset):
        def save_episode(self):
            self.episode_buffer = {"episode_index": np.array([0, 0])}
            raise RuntimeError("original save failure")

        def clear_episode_buffer(self):
            assert self.episode_buffer["episode_index"] == 0
            raise TypeError("synthetic cleanup failure")

    writer = LeRobotV21Writer(
        dataset_root=tmp_path / "dataset",
        repo_id="local/test",
        task="pick object",
        fps=50,
        use_videos=True,
        image_width=3,
        image_height=2,
        _dataset_cls=FailingDataset,
    )

    with pytest.raises(RuntimeError, match="original save failure"):
        writer.write_episode([make_entry()])

    output = capsys.readouterr().out
    assert "synthetic cleanup failure" in output
    assert "继续抛出原始保存异常" in output
