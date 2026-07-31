from types import SimpleNamespace

import numpy as np

from tiangong_recorder.image_decoder import decode_ros_image


def test_decode_ros_image_preserves_channel_bytes_and_removes_row_padding():
    rows = np.array(
        [
            [1, 2, 3, 4, 5, 6, 99, 99],
            [7, 8, 9, 10, 11, 12, 99, 99],
        ],
        dtype=np.uint8,
    )
    message = SimpleNamespace(
        width=2,
        height=2,
        encoding="rgb8",
        step=8,
        data=rows.tobytes(),
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=123, nanosec=456),
        ),
    )

    frame = decode_ros_image(message, expected_width=2, expected_height=2)

    assert frame.timestamp_ns == 123_000_000_456
    assert frame.encoding == "rgb8"
    assert frame.image.shape == (2, 2, 3)
    assert frame.image.tolist() == [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]

