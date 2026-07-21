import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UGate, RXXGate, RYYGate, RZZGate
from qiskit.circuit import ParameterVector
import logging

### Topologies ###

def linear_topology(init_order: list) -> list:
    """ Generates a linear topology from an initial qubit order.
    Args:
        init_order (list): A list of qubit indices representing the initial order of qubits.
    Returns:
        list: A list of qubit pairs representing the connections between qubits.
    """
    connections = []
    for i in range(len(init_order)-1):
        connections.append((init_order[i], init_order[i+1]))
    connections = sorted(list(set([tuple(sorted(c)) for c in connections])))
    return connections

def all_to_all_topology(num_qubits: int) -> list:
    """ Generates an all-to-all topology for a given number of qubits.
    Args:
        num_qubits (int): The number of qubits.
    Returns:
        list: A list of qubit pairs representing the connections between qubits.
    """
    connections = [(i, j) for i in range(num_qubits) for j in range(i+1, num_qubits)]
    connections = sorted(list(set([tuple(sorted(c)) for c in connections])))
    return connections

def nearest_neighbor_topology(w: int, h: int):
    """ Generates a nearest-neighbor topology for a 2D grid of qubits.
    Remarks:
        It should at least work for the 3x3 BAS grid.
    Args:
        w (int): The width of the grid.
        h (int): The height of the grid.
    Returns:
        list: A list of qubit pairs representing the connections between qubits.
    """
    connections = []
    for i in range(w*h):
        if i%w != w-1:
            connections.append((i, i+1))
        if i//w != h-1:
            connections.append((i, i+w))
    connections = sorted(list(set([tuple(sorted(c)) for c in connections])))
    return connections

def metric_based_topology(distmat: np.ndarray, threshold: float) -> list:
    dist_filter = np.zeros_like(distmat)
    dist_filter[distmat < threshold] = 1.0  # filter connections based on distance threshold
    dist_filter = dist_filter - np.eye(distmat.shape[0])  # remove self-connections
    connections = np.argwhere(dist_filter == 1.0)
    connections = sorted(list(set([tuple(sorted(c)) for c in connections])))
    return connections

def connection_threshold_curve(distmat: np.ndarray, num_steps: int = 100) -> tuple:
    """Sweep the metric_based_topology threshold and count the resulting undirected connections.

    Mirrors metric_based_topology's own convention (dist < threshold, self-connections removed) so
    the curve and the actual extension graph it characterizes are always consistent.

    Args:
        distmat (np.ndarray): Symmetric pairwise distance matrix (diagonal ignored).
        num_steps (int): Number of threshold samples spanning (0, 1].
    Returns:
        tuple: (thresholds, counts) -- thresholds is a linspace over (0, 1]; counts[i] is the
            number of undirected connections at thresholds[i]. counts is monotonically
            non-decreasing in thresholds.
    """
    distmat = np.asarray(distmat, dtype=float)
    dim = distmat.shape[0]
    thresholds = np.linspace(1 / num_steps, 1, num_steps)
    counts = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        dist_filter = np.zeros_like(distmat)
        dist_filter[distmat < threshold] = 1.0
        dist_filter -= np.eye(dim)
        counts[i] = np.sum(dist_filter) / 2
    return thresholds, counts

def knee_threshold(distmat: np.ndarray, num_steps: int = 100) -> float:
    """Auto-select the metric_based_topology threshold at the knee/elbow of
    connection_threshold_curve, replacing a hand-tuned threshold.

    Implements the Kneedle max-distance-to-chord heuristic (Satopaa et al., 2011): the curve's
    axes are min-max normalized to [0, 1], and the knee is the point of maximum perpendicular
    distance from the chord joining the curve's first and last points. Using the ABSOLUTE distance
    (rather than a signed one that assumes a known concave/convex direction) makes this robust
    regardless of whether a given dataset's curve happens to be concave, convex, or S-shaped.

    Args:
        distmat (np.ndarray): Symmetric pairwise distance matrix, as consumed by
            metric_based_topology.
        num_steps (int): Resolution of the threshold sweep (see connection_threshold_curve).
    Returns:
        float: The threshold at the detected knee.
    """
    thresholds, counts = connection_threshold_curve(distmat, num_steps)

    x_range = thresholds.max() - thresholds.min()
    y_range = counts.max() - counts.min()
    if x_range == 0 or y_range == 0:
        logger = logging.getLogger('QCBM')
        logger.warning("knee_threshold: degenerate connection-threshold curve (no variation), "
                        "falling back to the curve midpoint.")
        return float(thresholds[len(thresholds) // 2])

    x = (thresholds - thresholds.min()) / x_range
    y = (counts - counts.min()) / y_range

    # perpendicular distance of every (x, y) point from the chord through the curve's first and
    # last points; the knee is the point furthest from that chord, in either direction.
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    numer = np.abs((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0))
    denom = np.hypot(x1 - x0, y1 - y0)
    distances = numer / denom

    knee_idx = int(np.argmax(distances))
    return float(thresholds[knee_idx])

def percolation_threshold(distmat: np.ndarray) -> float:
    """Auto-select the metric_based_topology threshold as the bond-percolation threshold: the
    smallest threshold at which the resulting graph is fully connected (every qubit reachable from
    every other), replacing a hand-tuned threshold.

    Args:
        distmat (np.ndarray): Symmetric pairwise distance matrix, as consumed by
            metric_based_topology.
    Returns:
        float: The percolation threshold.
    """
    distmat = np.asarray(distmat, dtype=float)
    shift = max(distmat.max(), 1.0) * 1e-9
    weights = distmat + shift
    np.fill_diagonal(weights, 0.0)
    mst = minimum_spanning_tree(weights)
    rows, cols = mst.nonzero()
    max_dist = distmat[rows, cols].max()
    return float(np.nextafter(max_dist, np.inf))

def select_threshold(distmat: np.ndarray, rule: str = "knee", num_steps: int = 100) -> float:
    """Auto-select the metric_based_topology threshold via the given rule.

    Args:
        distmat (np.ndarray): Symmetric pairwise distance matrix, as consumed by
            metric_based_topology.
        rule (str): "knee" (see knee_threshold) or "percolation" (see percolation_threshold).
        num_steps (int): Resolution of the threshold sweep; only used by rule="knee".
    Returns:
        float: The selected threshold.
    """
    if rule == "knee":
        return knee_threshold(distmat, num_steps)
    if rule == "percolation":
        return percolation_threshold(distmat)
    raise ValueError(f"Unknown threshold_rule '{rule}', expected 'knee' or 'percolation'.")

def chow_liu_topology(affinity: np.ndarray) -> list:
    """Generate a Chow-Liu dependency-tree topology from a pairwise affinity matrix.

    The Chow-Liu tree (Chow & Liu, 1968) is the maximum-weight spanning tree over the features
    whose edge weights are their pairwise mutual information. Among all tree-structured
    distributions it is the one with minimal KL divergence to the true joint distribution, i.e. the
    information-theoretically optimal *tree* of pairwise dependencies. It always yields exactly n-1
    edges forming a single connected, acyclic graph.

    Args:
        affinity (np.ndarray): Symmetric (n x n) matrix where a HIGHER value means a STRONGER
            pairwise dependency (e.g. mutual information). The diagonal is ignored.
    Returns:
        list: A sorted, de-duplicated list of the n-1 tree edges as qubit pairs.
    """
    affinity = np.asarray(affinity, dtype=float)
    # scipy only provides a MINIMUM spanning tree and treats a 0 entry as "no edge". Convert the
    # max-weight problem on `affinity` into a min-weight one via weights = (max + 1) - affinity:
    # subtracting a constant from every edge leaves the optimal spanning tree unchanged, while the
    # +1 offset keeps all off-diagonal weights strictly positive so scipy sees the COMPLETE graph
    # (otherwise a zero-MI pair would drop out and could disconnect the tree). Diagonal -> 0 (no
    # self-loops, never selected by an MST).
    weights = affinity.max() + 1.0 - affinity
    np.fill_diagonal(weights, 0.0)
    mst = minimum_spanning_tree(weights)
    rows, cols = mst.nonzero()
    connections = sorted(set(tuple(sorted((int(i), int(j)))) for i, j in zip(rows, cols)))
    return connections

def random_topology(num_qubits: int, num_extensions: int, base_topology: list = [], seed: int  = 42) -> list:
    np.random.seed(seed)
    connections = base_topology.copy()
    while len(connections) < len(base_topology) + num_extensions:
        i, j = np.random.choice(num_qubits, 2, replace=False)
        new_connection = tuple(sorted((i, j)))
        if new_connection not in connections:
            connections.append(new_connection)
    connections = sorted(list(set([tuple(sorted(c)) for c in connections])))  # sort qubit pairs
    return connections

### Circuit extension functions ###

def add_su4_gate(circuit: QuantumCircuit, qubit1: int, qubit2: int, params: list):
    """
    Adds a parameterized SU(4) gate decomposition to a Qiskit QuantumCircuit.

    Args:
        circuit (QuantumCircuit): The quantum circuit to which the gate will be added.
        qubit1 (int): The index of the first qubit.
        qubit2 (int): The index of the second qubit.
        params (list): A list of 15 parameters for the SU(4) decomposition.
    """
    assert len(params) == 15, "SU(4) decomposition requires 15 parameters."
    assert isinstance(circuit, QuantumCircuit), "Input circuit must be a QuantumCircuit."
    
    
    # U(2) gates on individual qubits
    circuit.append(UGate(theta=params[0], phi=params[1], lam=params[2]), [qubit1])  # U(2)_i(θ1:3)
    circuit.append(UGate(theta=params[3], phi=params[4], lam=params[5]), [qubit2])  # U(2)_j(θ4:6)
    
    # Entangling gates
    circuit.append(RXXGate(params[6]), [qubit1, qubit2])  # XX(θ7)
    circuit.append(RYYGate(params[7]), [qubit1, qubit2])  # YY(θ8)
    circuit.append(RZZGate(params[8]), [qubit1, qubit2])  # ZZ(θ9)
    
    # Additional U(2) gates
    circuit.append(UGate(theta=params[9], phi=params[10], lam=params[11]), [qubit1])  # U(2)_i(θ10:12)
    circuit.append(UGate(theta=params[12], phi=params[13], lam=params[14]), [qubit2])  # U(2)_j(θ13:15)


def extend_circuit(circuit: QuantumCircuit, init_topology: list, extension_topology: list, mean=0, stddev=0.01):
    """
    Extends a QuantumCircuit by adding SU(4) gates between given qubit pairs.

    Args:
        circuit (QuantumCircuit): The quantum circuit to be extended, must be unparameterized.
        mean (float): The mean of the normal distribution for the SU(4) gate parameters.
        stddev (float): The standard deviation of the normal distribution for the SU(4) gate parameters.

    Returns:
        QuantumCircuit: The extended quantum circuit.
    """

    qc_ext = circuit

    connections = sorted(list(set(extension_topology) - set(init_topology)))

    logger = logging.getLogger('QCBM')
    logger.info(f"Initial connections: {len(init_topology)}")
    logger.info(f"{init_topology}")
    logger.info(f"Extension connections: {len(extension_topology)}")
    logger.info(f"{extension_topology}")
    logger.info(f"New connections: {len(connections)}")
    logger.info(f"{connections}")

    for i, j in connections:
        params = np.random.normal(mean, stddev, 15)
        add_su4_gate(qc_ext, i, j, params)

    return qc_ext


def compose_parameterized_circuit(circuit: QuantumCircuit):

    qc = circuit.copy()  # work on a copy of the input circuit

    # Count the number of parameters in the circuit
    num_params = 0
    for i in range(len(qc)):
        num_params += len(qc.data[i].params)

    # Create a ParameterVector to replace numeric parameters
    param_vector = ParameterVector('θ', num_params)
    param_values = np.zeros(num_params)

    # New QuantumCircuit for updated CircuitInstructions
    qr = QuantumRegister(qc.num_qubits, 'q')
    new_qc = QuantumCircuit(qr)

    # Replace numeric parameters in CircuitInstructions with ParameterVector elements
    param_idx = 0
    for i in range(len(qc.data)):
        instr = qc.data[i]

        if instr.params:  # some instructions have no parameters

            param_vec_elements = []
            for j in range(len(instr.params)):
                param_values[param_idx] = instr.operation.params[j]  # Save the numeric parameter values
                param_vec_elements.append(param_vector[param_idx])  # Create a list of ParameterVector elements
                param_idx += 1

            # Replace params with corresponding ParameterVector elements
            instr.operation.params = param_vec_elements 

        # add the updated instruction to the new QuantumCircuit
        new_qc.append(instr.operation, instr.qubits, instr.clbits)

    return new_qc, param_values


