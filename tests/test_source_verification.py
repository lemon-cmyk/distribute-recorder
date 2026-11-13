from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

import tiangong_recorder.lerobot_converter as converter_module
from tiangong_recorder.lerobot_converter import ConverterConfig, PklToLeRobotConverter
from test_lerobot_converter import FakeWriter, make_config, write_pickle


@pytest.mark.parametrize("verify_source", [True, False])
def test_source_changes_during_conversion(tmp_path, monkeypatch, verify_source):
    config = make_config(tmp_path, verify_source=verify_source)
    source = Path(config.input_dir) / "episode.pkl"
    write_pickle(source, 2)
    writer = FakeWriter(config.dataset_root)
    original_write = writer.write_episode

    def write_and_change_source(data):
        result = original_write(data)
        write_pickle(source, 3)
        return result

    monkeypatch.setattr(writer, "write_episode", write_and_change_source)
    converter = PklToLeRobotConverter(config, writer=writer)
    if not verify_source:
        monkeypatch.setattr(
            converter, "_sha256",
            lambda path: pytest.fail("disabled verification must not hash"),
        )
    try:
        expected = "converted_not_deleted" if verify_source else "converted"
        assert converter.process_file(source) == expected
        assert source.exists() == verify_source
        record = converter.state.get(source.resolve())
        assert record.status == "done"
        assert record.frame_count == 2
        assert record.source_deleted == (not verify_source)
    finally:
        converter.close()


def test_disabled_verification_deduplicates_changed_source_by_path(tmp_path, monkeypatch):
    config = make_config(tmp_path, verify_source=False, delete_source_pkl=False)
    source = Path(config.input_dir) / "episode.pkl"
    write_pickle(source, 2)
    writer = FakeWriter(config.dataset_root)
    converter = PklToLeRobotConverter(config, writer=writer)
    monkeypatch.setattr(
        converter, "_sha256",
        lambda path: pytest.fail("disabled verification must not hash"),
    )
    try:
        assert converter.process_file(source) == "converted"
        assert converter.state.get(source.resolve()).source_sha256 == "unchecked"
        write_pickle(source, 3)
        assert converter.process_file(source) == "already_done"
        assert len(writer.calls) == 1
        assert source.exists()
    finally:
        converter.close()


def test_disabled_verification_accepts_existing_checked_record(tmp_path):
    config = make_config(tmp_path, delete_source_pkl=False)
    source = Path(config.input_dir) / "episode.pkl"
    write_pickle(source, 2)
    writer = FakeWriter(config.dataset_root)
    checked = PklToLeRobotConverter(config, writer=writer)
    try:
        assert checked.process_file(source) == "converted"
    finally:
        checked.close()
    write_pickle(source, 3)
    unchecked_config = make_config(tmp_path, verify_source=False)
    unchecked = PklToLeRobotConverter(unchecked_config, writer=writer)
    try:
        assert unchecked.process_file(source) == "already_done"
        assert len(writer.calls) == 1
        assert not source.exists()
        assert unchecked.state.get(source.resolve()).source_deleted
    finally:
        unchecked.close()


@pytest.mark.parametrize("use_cli", [True, False])
def test_disable_source_verification_configuration(tmp_path, monkeypatch, use_cli):
    config = make_config(tmp_path, verify_source=use_cli)
    config_path = tmp_path / "converter.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
    configs = []

    class FakeConverter:
        def __init__(self, actual_config):
            configs.append(actual_config)

        def scan_once(self):
            return []

        def close(self, *, wait_for_writer=True):
            pass

    monkeypatch.setattr(converter_module, "PklToLeRobotConverter", FakeConverter)
    args = ["--config", str(config_path), "--once"]
    if use_cli:
        args.append("--no-source-verification")
    assert converter_module.main(args) == 0
    assert configs[0].verify_source is False
    assert configs[0].compute_source_sha256 is True


def test_source_verification_defaults_on_and_requires_boolean(tmp_path):
    assert ConverterConfig().verify_source is True
    with pytest.raises(ValueError, match="verify_source"):
        make_config(tmp_path, verify_source="false").validate()
