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

