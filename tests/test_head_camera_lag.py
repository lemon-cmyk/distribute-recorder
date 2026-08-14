import numpy as np

from tiangong_recorder.head_camera_lag import (
    _curve_peak,
    correlation_curve,
    regularized_canonical_correlation,
)


def test_canonical_correlation_recovers_unknown_linear_projection():
    random = np.random.default_rng(12)
    state = random.normal(size=(1000, 8))
    projection = random.normal(size=(8, 4))
    image = state @ projection + random.normal(scale=0.01, size=(1000, 4))

    assert regularized_canonical_correlation(state, image) > 0.99


def test_multivariate_curve_recovers_positive_image_lag():
    random = np.random.default_rng(23)
    state = random.normal(size=(1000, 8))
    state[:, -2:] = np.abs(state[:, -2:]) + 0.1
    projection = random.normal(size=(8, 4))
    image = np.zeros((1000, 4))
    image[3:] = state[:-3] @ projection

    curve = correlation_curve(state, image, max_lag_frames=10)
    result = _curve_peak(curve, fps=50)

    assert result["lag_frames"] == 3
    assert result["peak_correlation"] > 0.99
