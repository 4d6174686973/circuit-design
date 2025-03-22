from src.extension import linear_topology, nearest_neighbor_topology, random_topology


init_topologies = {
    "BAS_3x3" : [0, 1, 2, 5, 4, 3, 6 ,7, 8],
    "SR_12Q" : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

def test_linear_topology():

    correct_topologies = {
        "BAS_3x3" : [(0, 1), (1, 2), (3, 4), (6, 7), (4, 5), (3, 6), (2, 5), (7, 8)],
        "SR_12Q" : [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)],
    }

    assert linear_topology(init_topologies["BAS_3x3"]) == list(set([tuple(sorted(c)) for c in correct_topologies["BAS_3x3"]]))
    assert linear_topology(init_topologies["SR_12Q"]) == list(set([tuple(sorted(c)) for c in correct_topologies["SR_12Q"]]))


def test_nearest_neighbor_topology():
    
    correct_extension = [(0, 3), (1, 4), (4, 7), (5, 8)]
    linear_connections = linear_topology(init_topologies["BAS_3x3"])
    nn_connections = nearest_neighbor_topology(3, 3)
    extension = set(nn_connections) - set(linear_connections)  # remove linear connections

    assert list(set([tuple(sorted(c)) for c in extension])) == list(set([tuple(sorted(c)) for c in correct_extension]))


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