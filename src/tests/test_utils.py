import numpy as np

from src.utils import FeatureQuantizer, feature_distance_matrix


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


def _quantizers(data, bits=4):
    return [FeatureQuantizer.fit(data, bits, kind) for kind in ("minmax", "arcsinh")]


def test_decode_levels_lower_edge_keeps_round_trip_below_input():
    """The default (lower-edge) decode must stay the exact inverse-up-to-truncation of `levels`."""
    rng = np.random.default_rng(0)
    data = rng.standard_t(3, size=(500, 2)) * 0.02
    for q in _quantizers(data):
        back = q.decode(q.encode(data))
        # every level decodes to its own bin's lower edge, so no reconstruction overshoots its input
        # (up to the clipped end bins, where out-of-range values are folded inwards)
        inside = (q.levels(data) > 0) & (q.levels(data) < q.max_int)
        assert np.all(back[inside] <= data[inside] + 1e-12)


def test_decode_levels_center_halves_the_reconstruction_bias():
    """Bin-center decoding removes the lower-edge convention's one-sided half-bin bias."""
    rng = np.random.default_rng(0)
    data = rng.standard_t(3, size=(500, 2)) * 0.02
    for q in _quantizers(data):
        edge_err = q.decode(q.encode(data)) - data
        center_err = q.decode(q.encode(data), center=True) - data
        # the edge convention is biased low everywhere; centering leaves a near-zero mean error
        assert edge_err.mean() < 0
        assert abs(center_err.mean()) < abs(edge_err.mean())
        assert np.sqrt((center_err ** 2).mean()) < np.sqrt((edge_err ** 2).mean())


def test_decode_levels_center_shifts_by_half_a_bin_in_warped_space():
    rng = np.random.default_rng(1)
    data = rng.standard_t(3, size=(200, 2)) * 0.02
    k = np.arange(16)
    for q in _quantizers(data):
        for feature in range(2):
            lo, hi = q.w_min[feature], q.w_max[feature]
            half = 0.5 * (hi - lo) / q.max_int
            edge = q.warp(q.decode_levels(k, feature), feature)
            center = q.warp(q.decode_levels(k, feature, center=True), feature)
            assert np.allclose(center - edge, half)
