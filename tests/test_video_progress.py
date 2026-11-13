import io

from tiangong_recorder.video_progress import TerminalProgressBar


class InteractiveStream(io.StringIO):
    def isatty(self):
        return True


def test_terminal_progress_rewrites_one_line_and_finishes_with_newline():
    stream = InteractiveStream()
    progress = TerminalProgressBar("head", stream=stream, width=10)

    progress.update(0, 100)
    progress.update(50, 100)
    progress.update(100, 100)

    output = stream.getvalue()
    assert "\r" in output
    assert "50.0% 50/100" in output
    assert "100.0% 100/100\n" in output


def test_noninteractive_progress_logs_only_ten_percent_steps():
    stream = io.StringIO()
    progress = TerminalProgressBar("left_wrist", stream=stream, width=10)

    for current in range(101):
        progress.update(current, 100)

    lines = stream.getvalue().splitlines()
    assert len(lines) == 11
    assert "0.0% 0/100" in lines[0]
    assert "100.0% 100/100" in lines[-1]



def test_av1_roundtrip_and_failure_cleanup(tmp_path):
    import av
    import numpy as np
    import pytest
    from tiangong_recorder.video_progress import encode_rgb_frames_with_av1
    path = tmp_path / "video.mp4"
    frames = [np.full((64, 64, 3), [240, 10, 10], dtype=np.uint8) for _ in range(8)]
    encode_rgb_frames_with_av1(frames, path, fps=50, label="test", width=64, height=64)
    with av.open(str(path)) as container:
        assert float(container.streams.video[0].average_rate) == 50
        decoded = list(container.decode(video=0))
        assert len(decoded) == 8
        rgb = decoded[0].to_ndarray(format="rgb24").mean(axis=(0, 1))
        assert rgb[0] > rgb[2] + 150
    original = path.read_bytes()
    with pytest.raises(ValueError):
        encode_rgb_frames_with_av1(frames + [np.zeros((32, 32, 3), np.uint8)], path,
                                  fps=50, label="test", width=64, height=64)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.encoding.mp4"))
