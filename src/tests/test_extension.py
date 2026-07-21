import numpy as np
import pytest

from src.extension import (linear_topology, nearest_neighbor_topology, random_topology,
                           chow_liu_topology, connection_threshold_curve, knee_threshold,
                           percolation_threshold, select_threshold, metric_based_topology)


init_topologies = {
    "BAS_3x3" : [0, 1, 2, 5, 4, 3, 6 ,7, 8],
    "SR_12Q" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

def test_linear_topology():

    correct_topologies = {
        "BAS_3x3" : [(0, 1), (1, 2), (3, 4), (6, 7), (4, 5), (3, 6), (2, 5), (7, 8)],
        "SR_12Q" : [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)],
    }

    # linear_topology returns a sorted, de-duplicated list, so compare against the sorted set
    assert linear_topology(init_topologies["BAS_3x3"]) == sorted(set([tuple(sorted(c)) for c in correct_topologies["BAS_3x3"]]))
    assert linear_topology(init_topologies["SR_12Q"]) == sorted(set([tuple(sorted(c)) for c in correct_topologies["SR_12Q"]]))


def test_nearest_neighbor_topology():
    
    correct_extension = [(0, 3), (1, 4), (4, 7), (5, 8)]
    linear_connections = linear_topology(init_topologies["BAS_3x3"])
    nn_connections = nearest_neighbor_topology(3, 3)
    extension = set(nn_connections) - set(linear_connections)  # remove linear connections

    assert sorted(set([tuple(sorted(c)) for c in extension])) == sorted(set([tuple(sorted(c)) for c in correct_extension]))


def test_random_topology():
    
    iterations = 1000
    num_qubits = 9
    num_connections = 5
    exclude_list = [(1, 2), (3, 4), (5, 6), (7, 8)]
    exclude_list = list(set([tuple(sorted(c)) for c in exclude_list]))  # sort exclued list

    for _ in range(iterations):
        connections = random_topology(num_qubits, num_connections, exclude_list)
        extensions = set(connections) - set(exclude_list)
        assert len(set(extensions)) == num_connections # check for correct number of connections
        assert len(set(connections)) == len(connections) # check for duplicates
        assert all([tuple(sorted(c)) not in exclude_list for c in extensions])  # check for connections not in exclude_list


def test_chow_liu_topology_picks_max_weight_tree():
    # Highest-affinity edges are (0,1)=.9, (1,2)=.8, (2,3)=.7; they form the path 0-1-2-3 with no
    # cycle, so the maximum-weight spanning tree must select exactly those three.
    affinity = np.array([
        [0.0, 0.9, 0.1, 0.1],
        [0.9, 0.0, 0.8, 0.1],
        [0.1, 0.8, 0.0, 0.7],
        [0.1, 0.1, 0.7, 0.0],
    ])
    assert chow_liu_topology(affinity) == [(0, 1), (1, 2), (2, 3)]


def test_chow_liu_topology_is_a_spanning_tree():
    # For any affinity over n nodes the output must be a spanning tree: exactly n-1 edges, every
    # node covered, and acyclic (no edge joins two already-connected components).
    n = 9
    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    A = (A + A.T) / 2  # symmetric affinity

    edges = chow_liu_topology(A)

    assert len(edges) == n - 1                         # a spanning tree of n nodes has n-1 edges
    assert len({q for e in edges for q in e}) == n     # every qubit is connected
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in edges:
        ri, rj = find(i), find(j)
        assert ri != rj, "cycle detected -- Chow-Liu output must be acyclic"
        parent[ri] = rj


def _two_cluster_distmat(n_per_cluster=3, within=0.1, between=0.9):
    # two equal-size clusters: small within-cluster distance, large between-cluster distance --
    # a clean synthetic case for the threshold curve / knee tests.
    n = 2 * n_per_cluster
    dist = np.full((n, n), between)
    for g in range(2):
        idx = range(g * n_per_cluster, (g + 1) * n_per_cluster)
        for i in idx:
            for j in idx:
                dist[i, j] = within
    np.fill_diagonal(dist, 0.0)
    return dist


def test_connection_threshold_curve_is_monotonic_and_bounded():
    n = 9
    rng = np.random.default_rng(1)
    dist = rng.random((n, n))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)

    thresholds, counts = connection_threshold_curve(dist, num_steps=50)

    assert len(thresholds) == len(counts) == 50
    assert np.all(np.diff(counts) >= 0)          # non-decreasing as threshold rises
    assert counts[0] >= 0
    assert counts[-1] <= n * (n - 1) / 2          # can never exceed all possible undirected pairs


def test_connection_threshold_curve_matches_metric_based_topology():
    n = 6
    dist = _two_cluster_distmat(n_per_cluster=3)
    thresholds, counts = connection_threshold_curve(dist, num_steps=100)
    # spot-check a handful of thresholds against the actual topology builder they characterize
    for t in (0.05, 0.5, 0.95):
        idx = int(np.argmin(np.abs(thresholds - t)))
        assert counts[idx] == len(metric_based_topology(dist, thresholds[idx]))


def test_knee_threshold_lands_in_a_real_gap():
    # with a clean bimodal distance distribution, the knee must fall strictly between the two
    # distinct distance values actually present in the matrix -- not before the first jump and not
    # at/after the curve's saturation point.
    dist = _two_cluster_distmat(n_per_cluster=3, within=0.1, between=0.9)
    knee = knee_threshold(dist, num_steps=100)
    assert 0.1 < knee <= 0.9


def test_knee_threshold_is_deterministic():
    dist = _two_cluster_distmat()
    assert knee_threshold(dist, num_steps=100) == knee_threshold(dist, num_steps=100)


def test_knee_threshold_handles_degenerate_curve():
    # all-identical distances -> every threshold in (0, 1] admits every pair -> the curve is flat
    # (no variation), which must hit the documented fallback (curve midpoint) rather than raising.
    n = 5
    dist = np.zeros((n, n))
    thresholds, _ = connection_threshold_curve(dist, num_steps=100)
    knee = knee_threshold(dist, num_steps=100)
    assert knee == thresholds[len(thresholds) // 2]


def _is_connected(n: int, edges: list) -> bool:
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in edges:
        parent[find(i)] = find(j)
    return len({find(i) for i in range(n)}) == 1


def test_percolation_threshold_connects_the_graph():
    n = 9
    rng = np.random.default_rng(2)
    dist = rng.random((n, n))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)

    threshold = percolation_threshold(dist)
    edges = metric_based_topology(dist, threshold)

    assert _is_connected(n, edges)


def test_percolation_threshold_is_minimal():
    # two clusters bridged only by distance-0.9 pairs: the bridge is the MST's heaviest edge, so the
    # threshold must sit just above 0.9 (connects) while 0.9 itself must NOT connect the graph, since
    # metric_based_topology filters with a strict '<' and would drop every 0.9-valued bridge edge.
    n = 6
    dist = _two_cluster_distmat(n_per_cluster=3, within=0.1, between=0.9)

    threshold = percolation_threshold(dist)

    assert 0.9 < threshold < 0.9 + 1e-6
    assert not _is_connected(n, metric_based_topology(dist, 0.9))
    assert _is_connected(n, metric_based_topology(dist, threshold))


def test_percolation_threshold_handles_zero_distances():
    # scipy's MST routines treat an exact 0.0 entry as "no edge" even on a dense array -- a
    # zero-distance pair (e.g. two identical feature vectors) must still be handled correctly rather
    # than silently dropped or raising.
    dist = np.array([
        [0.0, 0.0, 0.3, 0.8],
        [0.0, 0.0, 0.3, 0.8],
        [0.3, 0.3, 0.0, 0.5],
        [0.8, 0.8, 0.5, 0.0],
    ])
    threshold = percolation_threshold(dist)
    edges = metric_based_topology(dist, threshold)
    assert _is_connected(4, edges)


def test_percolation_threshold_is_deterministic():
    dist = _two_cluster_distmat()
    assert percolation_threshold(dist) == percolation_threshold(dist)


def test_select_threshold_dispatches_by_rule():
    dist = _two_cluster_distmat(n_per_cluster=3, within=0.1, between=0.9)

    assert select_threshold(dist, "knee") == knee_threshold(dist)
    assert select_threshold(dist, "percolation") == percolation_threshold(dist)

    with pytest.raises(ValueError):
        select_threshold(dist, "bogus")