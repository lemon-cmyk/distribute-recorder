import json
import pickle

import numpy as np
import pytest

from tiangong_recorder.hil_metadata import HIL_FEATURES, extract_hil_metadata
from test_lerobot_writer import make_entry, make_writer


def hil_entry(index):
    entry = make_entry()
    entry["complementary_info"] = {
        "policy_action": np.arange(16, dtype=np.float32) + index,
        "policy_action_valid": np.array([index != 1], dtype=np.bool_),
        "is_intervention": np.array([index == 1], dtype=np.float32),
        "state": np.array([index], dtype=np.float32),
        "collector_policy_id": np.array(["human" if index == 1 else "policy"]),
    }
    entry["episode_success"] = "success" if index == 0 else "failure"
    return entry


def test_legacy_episode_has_no_added_hil_columns(tmp_path):
    writer = make_writer(tmp_path)
    writer.write_episode([make_entry(), make_entry()])
    assert set(writer.dataset.features).isdisjoint(HIL_FEATURES)
    assert all(set(frame).isdisjoint(HIL_FEATURES) for frame in writer.dataset.frames)


def test_all_hil_fields_keep_frame_alignment_and_values(tmp_path):
    entries = [hil_entry(i) for i in range(3)]
    writer = make_writer(tmp_path)
    result = writer.write_episode(entries)
    assert result.frame_count == 3
    assert set(HIL_FEATURES).issubset(writer.dataset.features)
    for i, frame in enumerate(writer.dataset.frames):
        expected = extract_hil_metadata(entries[i])
        for key in HIL_FEATURES:
            if HIL_FEATURES[key]["dtype"] == "string":
                assert frame[key] == expected[key]
            else:
                np.testing.assert_array_equal(frame[key], expected[key])
                assert frame[key].dtype == np.dtype(HIL_FEATURES[key]["dtype"])
    assert writer.dataset.frames[1]["episode_success"] == "failure"
    assert writer.dataset.frames[0]["episode_success"] == "success"


def test_older_hil_episode_without_validity_field(tmp_path):
    entries = [hil_entry(i) for i in range(3)]
    for entry in entries:
        del entry["complementary_info"]["policy_action_valid"]
    writer = make_writer(tmp_path)
    writer.write_episode(entries)
    assert "complementary_info.policy_action_valid" not in writer.dataset.features
    assert "complementary_info.is_intervention" in writer.dataset.features


def test_only_intervention_field_is_supported(tmp_path):
    entry = make_entry()
    entry["complementary_info"] = {"is_intervention": 1}
    writer = make_writer(tmp_path)
    writer.write_episode([entry])
    assert set(writer.dataset.features).intersection(HIL_FEATURES) == {
        "complementary_info.is_intervention"
    }
    assert writer.dataset.frames[0]["complementary_info.is_intervention"].tolist() == [1.0]


def test_scalar_labels_and_array_strings_are_normalized():
    entry = hil_entry(0)
    entry["complementary_info"]["is_intervention"] = 0
    entry["complementary_info"]["state"] = 0.0
    entry["complementary_info"]["policy_action_valid"] = False
    entry["episode_success"] = np.array(["success"])
    fields = extract_hil_metadata(entry)
    assert fields["complementary_info.is_intervention"].tolist() == [0.0]
    assert fields["complementary_info.policy_action_valid"].tolist() == [False]
    assert fields["episode_success"] == "success"


@pytest.mark.parametrize("first_has_hil", [False, True])
def test_inconsistent_episode_is_rejected_before_output(tmp_path, first_has_hil):
    entries = [hil_entry(0), make_entry()]
    if not first_has_hil:
        entries.reverse()
    writer = make_writer(tmp_path)
    with pytest.raises(ValueError, match="frame 1: HIL metadata fields differ"):
        writer.write_episode(entries)
    assert not writer.root.exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("is_intervention", [2]),
        ("is_intervention", [np.nan]),
        ("state", [3]),
        ("policy_action_valid", [2]),
        ("policy_action", np.zeros(15)),
        ("policy_action", np.full(16, np.inf)),
        ("collector_policy_id", [""]),
        ("collector_policy_id", [None]),
    ],
)
def test_bad_annotation_reports_frame_without_partial_output(tmp_path, key, value):
    entries = [hil_entry(0), hil_entry(1)]
    entries[1]["complementary_info"][key] = value
    writer = make_writer(tmp_path)
    with pytest.raises(ValueError, match="frame 1: complementary_info"):
        writer.write_episode(entries)
    assert not writer.root.exists()


@pytest.mark.parametrize("start_with_hil", [False, True])
def test_dataset_schema_change_fails_without_appending_frames(tmp_path, start_with_hil):
    writer = make_writer(tmp_path)
    initial = hil_entry(0) if start_with_hil else make_entry()
    changed = make_entry() if start_with_hil else hil_entry(0)
    writer.write_episode([initial])
    before = len(writer.dataset.frames)
    with pytest.raises(ValueError, match="HIL metadata fields differ.*new --dataset-name"):
        writer.write_episode([changed])
    assert len(writer.dataset.frames) == before
    assert writer.dataset.num_episodes == 1


def test_multiple_hil_episodes_preserve_labels(tmp_path):
    writer = make_writer(tmp_path)
    assert writer.write_episode([hil_entry(0)]).episode_index == 0
    first_label = writer.dataset.frames[0]["complementary_info.is_intervention"].item()
    # The fake retains frames; the real LeRobot writer resets its episode buffer.
    writer.dataset.frames.clear()
    assert writer.write_episode([hil_entry(1)]).episode_index == 1
    second_label = writer.dataset.frames[0]["complementary_info.is_intervention"].item()
    assert [first_label, second_label] == [0, 1]


@pytest.mark.parametrize("with_hil", [False, True])
def test_pkl_converter_parquet_and_lerobot_round_trip(tmp_path, with_hil):
    pytest.importorskip("lerobot")
    pytest.importorskip("av")
    pq = pytest.importorskip("pyarrow.parquet")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tiangong_recorder.lerobot_converter import ConverterConfig, PklToLeRobotConverter

    entries = [hil_entry(i) if with_hil else make_entry() for i in range(3)]
    for i, entry in enumerate(entries):
        for camera in entry["image"].values():
            camera["color"] = np.full((64, 64, 3), i * 40, dtype=np.uint8)
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    source = source_dir / "episode.pkl"
    with source.open("wb") as stream:
        pickle.dump(entries, stream)
    root = tmp_path / "dataset"
    config = ConverterConfig(
        input_dir=str(source_dir),
        dataset_root=str(root),
        state_db=str(tmp_path / "converter.sqlite3"),
        lock_file=str(tmp_path / "converter.lock"),
        repo_id="local/hil-round-trip",
        task="test intervention",
        fps=30,
        image_width=64,
        image_height=64,
        camera_color_order={name: "rgb" for name in ("head", "left_wrist", "right_wrist")},
    )
    converter = PklToLeRobotConverter(config)
    try:
        assert converter.process_file(source) == "converted"
    finally:
        converter.close()
    assert source.exists()
    table = pq.read_table(next(root.glob("data/**/*.parquet")))
    original_columns = [
        "action", "observation.state", "timestamp", "frame_index",
        "episode_index", "index", "task_index",
    ]
    assert table.column_names[:7] == original_columns
    assert table["frame_index"].to_pylist() == [0, 1, 2]
    assert table.num_rows == 3
    info = json.loads((root / "meta/info.json").read_text())
    if not with_hil:
        assert table.column_names == original_columns
        assert set(info["features"]).isdisjoint(HIL_FEATURES)
    else:
        assert set(HIL_FEATURES).issubset(table.column_names)
        assert set(HIL_FEATURES).issubset(info["features"])
        assert table["complementary_info.is_intervention"].to_pylist() == [[0.0], [1.0], [0.0]]
        assert table["complementary_info.state"].to_pylist() == [[0.0], [1.0], [2.0]]
        assert table["complementary_info.policy_action_valid"].to_pylist() == [[True], [False], [True]]
        assert table["complementary_info.collector_policy_id"].to_pylist() == ["policy", "human", "policy"]
        assert table["episode_success"].to_pylist() == ["success", "failure", "failure"]
        for i in range(3):
            assert table["complementary_info.policy_action"][i].as_py() == (np.arange(16) + i).tolist()

    loaded = LeRobotDataset(repo_id=config.repo_id, root=root, video_backend="pyav")
    assert loaded.num_frames == 3
    for i in [2, 0, 1]:
        frame = loaded[i]
        assert tuple(frame["action"].shape) == (16,)
        assert tuple(frame["observation.images.front"].shape) == (3, 64, 64)
        if with_hil:
            assert frame["complementary_info.is_intervention"].item() == (i == 1)
            assert frame["complementary_info.policy_action_valid"].item() == (i != 1)
            assert frame["complementary_info.collector_policy_id"] == ("human" if i == 1 else "policy")
            assert frame["episode_success"] == ("success" if i == 0 else "failure")
