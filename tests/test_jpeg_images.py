import io
import pickle

import numpy as np
import pytest
from PIL import Image

from tiangong_recorder.hil_metadata import extract_hil_metadata
from tiangong_recorder.lerobot_writer import CAMERA_NAMES, LeRobotV21Writer
from test_lerobot_writer import make_entry, make_writer


def jpeg_bytes(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="JPEG", quality=95, subsampling=0)
    return stream.getvalue()


def rgb_image(height=2, width=3):
    return np.full((height, width, 3), [210, 40, 15], dtype=np.uint8)


@pytest.mark.parametrize("order", ["rgb", "bgr"])
@pytest.mark.parametrize("container", [bytes, bytearray, memoryview])
@pytest.mark.parametrize("marker", [None, "jpeg"])
def test_jpeg_is_detected_and_rgb_independent_of_array_order(tmp_path, order, container, marker):
    entry = make_entry()
    data = jpeg_bytes(rgb_image())
    for camera in CAMERA_NAMES:
        entry["image"][camera]["color"] = container(data)
        if marker is not None:
            entry["image"][camera]["color_encoding"] = marker
    writer = make_writer(tmp_path, camera_color_order={camera: order for camera in CAMERA_NAMES})
    writer.write_episode([entry])
    with Image.open(io.BytesIO(data)) as image:
        expected = np.array(image.convert("RGB"))
    for camera in ("front", "left_wrist", "right_wrist"):
        actual = writer.dataset.frames[0][f"observation.images.{camera}"]
        np.testing.assert_array_equal(actual, expected)


def test_mixed_pkl_preserves_array_colors_state_and_hil(tmp_path):
    entries = [make_entry(), make_entry()]
    payload = jpeg_bytes(rgb_image())
    entries[0]["image"]["head"].update(color=payload, color_encoding="jpeg")
    entries[1]["image"]["left_wrist"]["color"] = payload
    for index, entry in enumerate(entries):
        entry["complementary_info"] = {"is_intervention": index}
        entry["episode_success"] = "success" if index == 0 else "failure"
    source = tmp_path / "mixed.pkl"
    source.write_bytes(pickle.dumps(entries))
    writer = make_writer(tmp_path, camera_color_order={camera: "bgr" for camera in CAMERA_NAMES})
    writer.write_episode(pickle.loads(source.read_bytes()))
    first, second = writer.dataset.frames
    assert first["observation.images.left_wrist"][0, 0].tolist() == [30, 20, 10]
    assert second["observation.images.front"][0, 0].tolist() == [3, 2, 1]
    np.testing.assert_array_equal(first["action"], second["action"])
    for entry, frame in zip(entries, writer.dataset.frames):
        for key, expected in extract_hil_metadata(entry).items():
            np.testing.assert_array_equal(frame[key], expected)
    assert source.is_file()


@pytest.mark.parametrize("payload", [b"", b"not a jpeg", b"\xff\xd8\xff\xe0truncated"])
def test_invalid_jpeg_is_rejected_before_dataset_creation(tmp_path, payload):
    entry = make_entry()
    entry["image"]["head"]["color"] = payload
    writer = make_writer(tmp_path)
    with pytest.raises(ValueError, match=r"image\.head\.color.*JPEG"):
        writer.write_episode([entry])
    assert not writer.root.exists()


def test_wrong_encoding_marker_and_non_jpeg_bytes_are_rejected(tmp_path):
    entry = make_entry()
    entry["image"]["head"].update(color=jpeg_bytes(rgb_image()), color_encoding="png")
    writer = make_writer(tmp_path)
    with pytest.raises(ValueError, match="color_encoding"):
        writer.write_episode([entry])
    stream = io.BytesIO()
    Image.fromarray(rgb_image()).save(stream, format="PNG")
    entry["image"]["head"] = {"color": stream.getvalue()}
    with pytest.raises(ValueError, match="JPEG"):
        writer.write_episode([entry])
    assert not writer.root.exists()


def test_jpeg_uses_existing_dimension_validation(tmp_path):
    entry = make_entry()
    entry["image"]["right_wrist"]["color"] = jpeg_bytes(rgb_image(height=4))
    writer = make_writer(tmp_path)
    with pytest.raises(ValueError, match="expected"):
        writer.write_episode([entry])
    assert not writer.root.exists()


def test_jpeg_infers_dimensions(tmp_path):
    entry = make_entry()
    for camera in CAMERA_NAMES:
        entry["image"][camera]["color"] = jpeg_bytes(rgb_image())
    from test_lerobot_writer import FakeLeRobotDataset

    writer = LeRobotV21Writer(
        dataset_root=tmp_path / "dynamic", repo_id="local/jpeg-size",
        task="test", fps=30, use_videos=True, _dataset_cls=FakeLeRobotDataset,
    )
    writer.write_episode([entry])
    assert (writer.image_height, writer.image_width) == (2, 3)


def test_jpeg_pkl_encodes_mp4_with_existing_lerobot_pipeline(tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("lerobot")
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    entries = []
    for index in range(4):
        entry = make_entry()
        data = jpeg_bytes(rgb_image(height=48, width=64))
        for camera in CAMERA_NAMES:
            entry["image"][camera] = {"color": data, "color_encoding": "jpeg"}
        entry["complementary_info"] = {"is_intervention": index % 2}
        entry["episode_success"] = "failure" if index == 3 else "success"
        entries.append(entry)
    source = tmp_path / "jpeg.pkl"
    source.write_bytes(pickle.dumps(entries))
    root = tmp_path / "jpeg_dataset"
    writer = LeRobotV21Writer(
        dataset_root=root, repo_id="local/jpeg-test", task="jpeg conversion test",
        fps=30, use_videos=True,
        camera_color_order={camera: "bgr" for camera in CAMERA_NAMES},
    )
    result = writer.write_episode(pickle.loads(source.read_bytes()))
    writer.close()
    assert result.frame_count == len(entries)
    videos = list(root.glob("videos/**/*.mp4"))
    assert len(videos) == 3
    for path in videos:
        with av.open(str(path)) as container:
            frames = list(container.decode(video=0))
            assert len(frames) == len(entries)
            image = frames[0].to_ndarray(format="rgb24")
            assert image.shape == (48, 64, 3)
            assert image[..., 0].mean() > image[..., 2].mean() + 100
    table = pq.read_table(next(root.glob("data/**/*.parquet")))
    assert table.num_rows == len(entries)
    assert table["episode_success"].to_pylist() == ["success", "success", "success", "failure"]
    assert table["complementary_info.is_intervention"].to_pylist() == [[0.0], [1.0], [0.0], [1.0]]
    loaded = LeRobotDataset(repo_id="local/jpeg-test", root=root, video_backend="pyav")
    assert loaded.num_frames == len(entries)
    assert tuple(loaded[2]["observation.images.front"].shape) == (3, 48, 64)
    assert source.is_file()
    assert not (root / "images").exists()
