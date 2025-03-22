import numpy as np
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


