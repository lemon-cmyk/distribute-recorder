import numpy as np

from tiangong_recorder.image_state_lag import estimate_image_lag


def test_estimate_positive_image_lag():
    random = np.random.default_rng(42)
    state = random.normal(size=500)
    image = np.zeros_like(state)
    image[3:] = state[:-3]

    result = estimate_image_lag(state, image, max_lag_frames=10)

    assert result["lag_frames"] == 3
    assert result["peak_correlation"] > 0.99


def test_estimate_negative_image_lag():
    random = np.random.default_rng(7)
    state = random.normal(size=500)
    image = np.zeros_like(state)
    image[:-2] = state[2:]

    result = estimate_image_lag(state, image, max_lag_frames=10)

    assert result["lag_frames"] == -2
    assert result["peak_correlation"] > 0.99
