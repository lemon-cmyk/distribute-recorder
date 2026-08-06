import pickle
from pathlib import Path

import pytest

import tiangong_recorder.lerobot_converter as converter_module
from tiangong_recorder.lerobot_converter import ConverterConfig, PklToLeRobotConverter
from tiangong_recorder.lerobot_writer import EpisodeWriteResult


class FakeWriter:
    def __init__(self, dataset_root, fail=False):
        self.dataset_root = Path(dataset_root)
        self.fail = fail
        self.calls = []
        self.saved = {}

    @property
    def next_episode_index(self):
        return len(self.saved)

    def write_episode(self, episode_data):
        self.calls.append(episode_data)
        if self.fail:
            raise RuntimeError("synthetic failure")
        index = self.next_episode_index
        self.saved[index] = len(episode_data)
        return EpisodeWriteResult(index, len(episode_data), self.dataset_root)

    def verify_episode(self, episode_index, frame_count):
        return self.saved.get(episode_index) == frame_count

    def close(self):
        pass


def make_config(tmp_path, **overrides):
    values = {
        "input_dir": str(tmp_path / "input"),
        "dataset_root": str(tmp_path / "dataset"),
        "state_db": str(tmp_path / "state.sqlite3"),
        "lock_file": str(tmp_path / "converter.lock"),
        "repo_id": "local/test",
        "image_width": 3,
        "image_height": 2,
    }
    values.update(overrides)
    return ConverterConfig(**values)


def write_pickle(path, frame_count=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump([{"frame": index} for index in range(frame_count)], stream)


def test_cli_dataset_name_and_task_derive_isolated_paths(tmp_path):
    config = make_config(tmp_path)

    updated = converter_module._apply_cli_overrides(
        config,
        dataset_name="pick_cube_v2",
        task="  Pick up the cube  ",
    )

    assert updated.dataset_root == str(tmp_path / "pick_cube_v2")
    assert updated.state_db == str(tmp_path / ".pick_cube_v2_converter.sqlite3")
    assert updated.lock_file == str(tmp_path / ".pick_cube_v2_converter.lock")
    assert updated.repo_id == "local/pick_cube_v2"
    assert updated.task == "Pick up the cube"


@pytest.mark.parametrize(
    "dataset_name",
    ["", ".hidden", "../escape", "nested/name", "has space", "中文名称"],
)
def test_cli_rejects_unsafe_dataset_name(tmp_path, dataset_name):
    with pytest.raises(ValueError, match="--dataset-name"):
        converter_module._apply_cli_overrides(
            make_config(tmp_path),
            dataset_name=dataset_name,
            task=None,
        )


def test_cli_rejects_empty_task(tmp_path):
    with pytest.raises(ValueError, match="--task"):
        converter_module._apply_cli_overrides(
            make_config(tmp_path),
            dataset_name=None,
            task="   ",
        )


def test_successful_conversion_deletes_source_and_records_done(tmp_path, capsys):
    config = make_config(tmp_path)
    source = Path(config.input_dir) / "episode_0001.pkl"
    write_pickle(source, 3)
    writer = FakeWriter(config.dataset_root)
    converter = PklToLeRobotConverter(config, writer=writer)

    assert converter.process_file(source) == "converted"
    assert not source.exists()
    record = converter.state.get(source.resolve())
    assert record.status == "done"
    assert record.frame_count == 3
    assert record.episode_index == 0
    assert record.source_deleted
    output = capsys.readouterr().out
    assert "阶段 1/8：完整性校验" in output
    assert "阶段 2/8：读取 PKL" in output
    assert "阶段 3/8：预留序号" in output
    assert "转换完成" in output
    converter.close()


def test_failed_conversion_keeps_source_and_records_error(tmp_path):
    config = make_config(tmp_path)
    source = Path(config.input_dir) / "episode_0001.pkl"
    write_pickle(source)
    writer = FakeWriter(config.dataset_root, fail=True)
    converter = PklToLeRobotConverter(config, writer=writer)

    assert converter.process_file(source) == "error"
    assert source.exists()
    record = converter.state.get(source.resolve())
    assert record.status == "error"
    assert record.target_episode_index is None
    assert not record.source_deleted
    assert "synthetic failure" in record.error_message
    converter.close()


def test_completed_source_is_not_converted_twice_when_retained(tmp_path):
    config = make_config(tmp_path, delete_source_pkl=False)
    source = Path(config.input_dir) / "episode_0001.pkl"
    write_pickle(source)
    writer = FakeWriter(config.dataset_root)
    converter = PklToLeRobotConverter(config, writer=writer)

    assert converter.process_file(source) == "converted"
    assert converter.process_file(source) == "already_done"
    assert len(writer.calls) == 1
    assert source.exists()
    converter.close()


def test_scan_ignores_incomplete_pickle(tmp_path):
    config = make_config(tmp_path)
    final_source = Path(config.input_dir) / "episode_0001.pkl"
    incomplete_source = Path(config.input_dir) / "episode_0002.pkl.tmp"
    write_pickle(final_source)
    write_pickle(incomplete_source)
    writer = FakeWriter(config.dataset_root)
    converter = PklToLeRobotConverter(config, writer=writer)

    results = converter.scan_once()
    assert [path.name for path, _result in results] == ["episode_0001.pkl"]
    assert incomplete_source.exists()
    converter.close()


def test_once_returns_nonzero_when_previous_error_exists(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config_path = tmp_path / "converter.yaml"
    config_path.write_text("placeholder", encoding="utf-8")

    class FakeConverter:
        def __init__(self, _config):
            pass

        def scan_once(self):
            return [(Path("failed.pkl"), "previous_error")]

        def close(self, *, wait_for_writer=True):
            pass

    class FakeLock:
        def close(self):
            pass

    monkeypatch.setattr(
        ConverterConfig, "from_yaml", classmethod(lambda cls, path: config)
    )
    monkeypatch.setattr(converter_module, "PklToLeRobotConverter", FakeConverter)
    monkeypatch.setattr(
        converter_module, "_acquire_singleton_lock", lambda path: FakeLock()
    )

    assert converter_module.main(["--config", str(config_path), "--once"]) == 1


def test_ctrl_c_interrupts_current_scan_without_waiting_for_writer(
    tmp_path, monkeypatch, capsys
):
    config = make_config(tmp_path)
    config_path = tmp_path / "converter.yaml"
    config_path.write_text("placeholder", encoding="utf-8")
    close_calls = []

    class FakeConverter:
        def __init__(self, _config):
            pass

        def scan_once(self):
            raise KeyboardInterrupt

        def close(self, *, wait_for_writer=True):
            close_calls.append(wait_for_writer)

    class FakeLock:
        def close(self):
            pass

    monkeypatch.setattr(
        ConverterConfig, "from_yaml", classmethod(lambda cls, path: config)
    )
    monkeypatch.setattr(converter_module, "PklToLeRobotConverter", FakeConverter)
    monkeypatch.setattr(
        converter_module, "_acquire_singleton_lock", lambda path: FakeLock()
    )

    assert converter_module.main(["--config", str(config_path)]) == 130
    assert close_calls == [False]
    assert "立即中止当前转换并退出" in capsys.readouterr().out


def test_main_rejects_existing_dataset_before_lock(tmp_path, monkeypatch, capsys):
    config = make_config(tmp_path)
    Path(config.dataset_root).mkdir()
    config_path = tmp_path / "converter.yaml"
    config_path.write_text("placeholder", encoding="utf-8")
    lock_called = False

    def unexpected_lock(_path):
        nonlocal lock_called
        lock_called = True
        raise AssertionError("lock must not be acquired")

    monkeypatch.setattr(
        ConverterConfig, "from_yaml", classmethod(lambda cls, path: config)
    )
    monkeypatch.setattr(converter_module, "_acquire_singleton_lock", unexpected_lock)

    assert converter_module.main(["--config", str(config_path)]) == 2
    assert not lock_called
    assert "同名数据集已经存在" in capsys.readouterr().out
