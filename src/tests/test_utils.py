import numpy as np

from src.utils import feature_distance_matrix


def test_feature_distance_matrix_hamming_returns_ndarray():
    X = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    dist = feature_distance_matrix(X, "hamming")

    assert isinstance(dist, np.ndarray)
    assert dist.shape == (3, 3)
    assert np.allclose(np.diag(dist), 0.0)
    assert np.allclose(dist, dist.T)  # symmetric


def test_feature_distance_matrix_varinfo_returns_ndarray():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(200, 4))
    dist = feature_distance_matrix(X, "varinfo")

    assert isinstance(dist, np.ndarray)
    assert dist.shape == (4, 4)
    assert np.allclose(np.diag(dist), 0.0)


def test_feature_distance_matrix_invalid_metric_raises():
    X = np.zeros((5, 3))
    try:
        feature_distance_matrix(X, "not_a_metric")
        assert False, "expected ValueError"
    except ValueError:
        pass
