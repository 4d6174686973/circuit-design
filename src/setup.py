import pandas as pd
import numpy as np
from omegaconf import DictConfig
import logging
from scipy.spatial import distance
import time
import os
import yaml

# qiskit stuff
from qiskit import qpy
from qiskit import QuantumCircuit
from scikit_tt import TT
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_transpiler_service.transpiler_service import TranspilerService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# own modules
from src.qcbm import QCBM
from src.mps import MPS
from src.data import DataLoader, BAS, JGB
from src.extension import compose_parameterized_circuit, linear_topology, all_to_all_topology, nearest_neighbor_topology, metric_based_topology, extend_circuit, random_topology
from src.utils import varInfoMat
from src.decompositon import mps2circuit


def setup_qiskit_simulator(simulator: str) -> tuple:
    """Setup the Qiskit simulator.
    Args:
        simulator (str): The simulator to use "aer_statevec_cpu" or "aer_kawasaki"
    Returns:
        tuple: The sampler and backend
    """

    if simulator == "aer_statevec_cpu":
        backend = AerSimulator(method="statevector")
        sampler = SamplerV2(backend)
    elif simulator == "aer_statevec_gpu":
        backend = AerSimulator(method="statevector", device="GPU")
        sampler = SamplerV2(backend)
    elif simulator == "aer_kawasaki":
        service = setup_ibm_service()
        backend = service.backend("ibm_kawasaki")
        backend_sim = AerSimulator.from_backend(backend)
        sampler = SamplerV2(backend_sim)
    else:
        raise ValueError("Invalid simulator.")

    return sampler, backend

def setup_ibm_service(instance: str = "utokyo-kawasaki/keio-internal/keio-students") -> QiskitRuntimeService:
    """Setup the IBM qiskit runtime service.
    Args:
        instance (str, optional): IBM device instance "utokyo-kawasaki/..." (default) or "ibm-q-utokyo/..."
    Returns:
        QiskitRuntimeService: The qiskit runtime service
    """
    
    with open("ibm_token.txt", "r") as file:
        token = file.read()

    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        instance=instance,
        token=token,
        set_as_default=True,
        overwrite=True,
    )

    service = QiskitRuntimeService()
    return service

def transpile_circuit(circuit: QuantumCircuit, transpiler: str = "service") -> QuantumCircuit:
    """Transpile the circuit with the transpiler service."""

    if transpiler == "service":
        transpiler = TranspilerService( 
            backend_name="ibm_kawasaki", 
            ai=True,
            optimization_level=3
        )
        isa_circuit = transpiler.run(circuit)

    elif transpiler == "local":
        pm = generate_preset_pass_manager(backend="ibm_kawasaki", optimization_level=3)
        isa_circuit = pm.run(circuit)

    # log num of gates and depth
    logger = logging.getLogger("QCBM")
    logger.info(f"ISA Circuit - Gates: {isa_circuit.count_ops()}")
    logger.info(f"ISA Circuit - Num Params: {isa_circuit.num_parameters}")
    logger.info(f"ISA Circuit - Depth: {isa_circuit.depth()}")

    return isa_circuit

def setup_and_train_mps(cfg: DictConfig, X_train: np.ndarray) -> QuantumCircuit:

    # config parameters
    n_qubits = cfg["N_qubits"]
    cutoff = cfg["cutoff"]
    descenting_step_length = cfg["descenting_step_length"]
    descent_steps = cfg["descent_steps"]
    train_loops = cfg["train_loops"]

    # train MPS
    mps = MPS(n_qubits)
    mps.left_cano()
    mps.designate_data(X_train)
    mps.init_cumulants()
    mps.cutoff = cutoff
    mps.descenting_step_length = descenting_step_length
    mps.descent_steps = descent_steps
    mps.train(train_loops, rec_cut=False)
    
    save_dir = 'mps'
    mps.saveMPS(save_dir)  # save dir will be created here

    # MPS to PQC decomposition
    print("Converting MPS to PQC")
    matrices = [np.expand_dims(mat, axis=2) for mat in mps.matrices]
    mps_tt = TT(matrices)
    circuit = mps2circuit(mps_tt)
    with open(f"{save_dir}/circuit.qpy", "wb") as file:
        qpy.dump(circuit, file)

    return circuit

def check_existing_mps(cfg: DictConfig) -> tuple[bool, str]:

    # if no mps folder exists
    if not os.path.exists("outputs/mps"):
        return False, None
    
    # check if same mps run is in folder
    for date in os.listdir("outputs/mps"):
        for run in os.listdir(f"outputs/mps/{date}"):
            curr_dir = f"outputs/mps/{date}/{run}"
            with open(f"{curr_dir}/config.yaml") as file:
                cfg_mps = yaml.full_load(file)

            if  cfg_mps["N_qubits"] == cfg["N_qubits"] and \
                cfg_mps["dataset"] == cfg["dataset"] and \
                cfg_mps["train_split"] == cfg["train_split"] and \
                cfg_mps["cutoff"] == cfg["cutoff"] and \
                cfg_mps["descenting_step_length"] == cfg["descenting_step_length"] and \
                cfg_mps["descent_steps"] == cfg["descent_steps"] and \
                cfg_mps["train_loops"] == cfg["train_loops"]:
                
                if cfg["dataset"] == "JGB" and cfg_mps["N_features"] == cfg["N_features"]:
                    return True, curr_dir
                elif cfg["dataset"] == "BAS" and cfg_mps["width"] == cfg["width"] and cfg_mps["height"] == cfg["height"]:
                    return True, curr_dir
    
    return False, None

def setup_circuit_extensions(cfg: DictConfig, mps_circuit: QuantumCircuit, X_train: pd.DataFrame) -> QuantumCircuit:

    # config parameters
    n_qubits = cfg["N_qubits"]
    n_random_extensions = cfg["N_random_extensions"]
    dataset = cfg["dataset"]
    extension = cfg["extension"]
    extension_metric = cfg["extension_metric"]
    extension_threshhold = cfg["extension_threshhold"]
    width = cfg["width"]
    height = cfg["height"]
    random_seed = cfg["random_seed"]

    logger = logging.getLogger('QCBM')

    # linear baseline connections
    init_order = list(range(cfg["N_qubits"]))
    init_connections = linear_topology(init_order)

    # extend circuit by method

    # no extension
    if extension == "none":
        extension_connections = []
        extended_circuit = mps_circuit
        logger.info(f"No extension applied.")
    
    # linear extension
    elif extension == "all_to_all":
        extension_connections = all_to_all_topology(n_qubits)
        extended_circuit = extend_circuit(mps_circuit, init_connections, extension_connections)
    
    # nearest neighbor extension
    elif extension == "nearest_neighbor":
        assert dataset == "BAS", "Nearest neighbor extension only implemented for BAS dataset."
        extension_connections = nearest_neighbor_topology(width, height)
        extended_circuit = extend_circuit(mps_circuit, init_connections, extension_connections)

    # metric based extension
    elif extension == "metric_based":
        if extension_metric == "hamming":
            dist = distance.cdist(X_train.T, X_train.T, 'hamming')
        elif extension_metric == "varinfo":
            dist = varInfoMat(pd.DataFrame(X_train), norm=True)
        else:
            raise ValueError("Invalid extension metric.")
        threshhold = extension_threshhold
        extension_connections = metric_based_topology(dist, threshhold)
        extended_circuit = extend_circuit(mps_circuit, init_connections, extension_connections)

    # random extension
    elif extension == "random":
        extension_connections = random_topology(n_qubits, n_random_extensions, init_connections, random_seed)
        extended_circuit = extend_circuit(mps_circuit, init_connections, extension_connections)


    else:
        raise ValueError("Invalid extension method.")

    # compose circuit and measure
    circuit, init_params = compose_parameterized_circuit(extended_circuit)
    circuit.measure_all()

    # log circuit info
    logger.info(f"Circuit - Gates: {circuit.count_ops()}")
    logger.info(f"Circuit - Num Params: {circuit.num_parameters}")
    logger.info(f"Circuit - Depth: {circuit.depth()}")

    return circuit, init_params

def setup_dataloader(cfg: DictConfig) -> DataLoader:
    if cfg["dataset"] == "BAS":
        dataset = BAS(cfg["width"], cfg["height"])
    elif cfg["dataset"] == "JGB":
        dataset = JGB(cfg["N_qubits"], cfg["N_features"])

    return DataLoader(dataset)

def setup_and_train_qcbm(cfg: DictConfig):
    
    # config parameters
    simulator = cfg["simulator"]
    train_split = cfg["train_split"]
    shots = cfg["N_shots"]
    batchsize = cfg["batchsize"]
    loss_func = cfg["loss_func"]
    sigmas = cfg["sigmas"]
    iterations = cfg["iterations"]
    adam_learning_rate = cfg["adam_learning_rate"]
    finite_diff_epsilon = cfg["finite_diff_epsilon"]
    kernel_multithreading = cfg["kernel_multithreading"]

    # variables
    save_dir = 'qcbm'
    logger = logging.getLogger("QCBM")
    start_time = time.time()
    logger.info("Program started")

    # Setup dataloader and split data
    dataloader = setup_dataloader(cfg)
    X_train, _, _, _ = dataloader.train_test_split(train_split)

    # Train new MPS or load existing circuit
    mps_exists, mps_path = check_existing_mps(cfg)
    if mps_exists:
        with open(f"{mps_path}/circuit.qpy", 'rb') as file:
            circuit_mps = qpy.load(file)[0]
    else:
        circuit_mps = setup_and_train_mps(cfg, X_train)

    # Setup circuit extensions and save extended circuit
    circuit_ext, init_params = setup_circuit_extensions(cfg, circuit_mps, X_train)
    with open(f"{save_dir}/ext_circuit.qpy", 'wb') as file:
        qpy.dump(circuit_ext, file)

    # Transpile Circuit if using real device backend
    if simulator == "aer_kawasaki":
        circuit = transpile_circuit(circuit, "service")
    else:
        circuit = circuit_ext.copy()

    # Train QCBM
    sampler, backend = setup_qiskit_simulator(cfg["simulator"])
    qcbm = QCBM(sampler, backend, circuit, init_params, adam_learning_rate, finite_diff_epsilon, kernel_multithreading)
    qcbm.stochastic_gradient_descent(dataloader, iterations, shots, batchsize, loss_func, sigmas, train_split)

    # Save model
    qcbm.save(save_dir)
    logger.info("Program finished")
    end_time = time.time()
    logger.info(f"Program execution time: {round((end_time - start_time) / 60, 2)} minutes")
