import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
import logging
from scipy.spatial import distance
import time
import os
import sys
import socket
import hashlib
from datetime import datetime

# qiskit stuff
from qiskit import qpy
from qiskit import QuantumCircuit
from scikit_tt import TT
from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2, QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_transpiler_service.transpiler_service import TranspilerService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from hydra.utils import to_absolute_path

# own modules
from src.qcbm import QCBM
from src.mps import MPS
from src.data import DataLoader, BAS, JGB
from src.extension import compose_parameterized_circuit, linear_topology, all_to_all_topology, nearest_neighbor_topology, metric_based_topology, extend_circuit, random_topology
from src.utils import varInfoMat
from src.decompositon import mps2circuit


def setup_qiskit_simulator(cfg: DictConfig) -> tuple:
    """Setup the Qiskit runner + backend.

    For simulation this returns a configured ``AerSimulator`` used via
    ``run(circuit, parameter_binds=[{param: values}])`` — Aer's native fast path that binds all
    2P+1 parameterizations in one C++ call. Empirically this is ~9x faster than routing the same
    options through ``qiskit_aer.primitives.SamplerV2`` (which rebinds/experiments per circuit).
    The real-device (``aer_kawasaki``) path keeps ``qiskit_ibm_runtime.SamplerV2``.

    Returns:
        (runner, backend, use_parameter_binds)
    """
    simulator = cfg["simulator"]

    if simulator in ("aer_statevec_cpu", "aer_statevec_gpu"):
        device = "GPU" if simulator == "aer_statevec_gpu" else "CPU"
        backend_options = {
            "method": "statevector",
            "device": device,
            "runtime_parameter_bind_enable": True,
            "max_parallel_experiments": cfg["aer_max_parallel_experiments"],
            "max_parallel_shots": cfg["aer_max_parallel_shots"],
            "seed_simulator": cfg["random_seed"],
        }
        if device == "GPU":
            backend_options["batched_shots_gpu"] = True
            backend_options["batched_shots_gpu_max_qubits"] = cfg["aer_batched_shots_gpu_max_qubits"]
            if cfg["N_qubits"] >= cfg["aer_blocking_qubits_threshold"]:
                backend_options["blocking_enable"] = True
                backend_options["blocking_qubits"] = cfg["aer_blocking_qubits"]
        backend = AerSimulator(**backend_options)
        return backend, backend, True

    elif simulator == "aer_kawasaki":
        service = setup_ibm_service()
        device_backend = service.backend("ibm_kawasaki")
        backend = AerSimulator.from_backend(device_backend)
        sampler = RuntimeSamplerV2(backend)
        return sampler, backend, False
    else:
        raise ValueError("Invalid simulator.")

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


def mps_cache_dir(cfg: DictConfig, X_train: np.ndarray) -> str:
    """Absolute cache directory for a trained MPS.

    Keyed on the MPS-relevant hyperparameters plus a hash of the actual training data, so runs that
    share an MPS (same dataset/qubits/params and same train split — the common case across a sweep's
    seeds and extensions) reuse one cache entry, while genuinely different train sets (e.g. BAS
    holdout with different seeds) get their own.
    """
    data_hash = hashlib.md5(np.ascontiguousarray(X_train).tobytes()).hexdigest()[:10]
    key = (f"{cfg['dataset']}_q{cfg['N_qubits']}_cut{cfg['cutoff']}"
           f"_dsl{cfg['descenting_step_length']}_ds{cfg['descent_steps']}"
           f"_tl{cfg['train_loops']}_{data_hash}")
    return to_absolute_path(os.path.join("outputs", "mps_cache", key))


def train_mps(cfg: DictConfig, X_train: np.ndarray, save_dir: str) -> QuantumCircuit:
    """Train an MPS on X_train, decompose to a PQC, and persist both to save_dir."""
    n_qubits = cfg["N_qubits"]

    mps = MPS(n_qubits)
    mps.left_cano()
    mps.designate_data(X_train)
    mps.init_cumulants()
    mps.cutoff = cfg["cutoff"]
    mps.descenting_step_length = cfg["descenting_step_length"]
    mps.descent_steps = cfg["descent_steps"]
    mps.train(cfg["train_loops"], rec_cut=False)

    os.makedirs(save_dir, exist_ok=True)
    mps.saveMPS(save_dir)  # writes tensors/ etc.

    print("Converting MPS to PQC")
    matrices = [np.expand_dims(mat, axis=2) for mat in mps.matrices]
    mps_tt = TT(matrices)
    circuit = mps2circuit(mps_tt)
    with open(f"{save_dir}/circuit.qpy", "wb") as file:
        qpy.dump(circuit, file)

    return circuit


def get_or_train_mps(cfg: DictConfig, X_train: np.ndarray) -> QuantumCircuit:
    """Return the shared MPS-derived circuit, training it once into the cache on a miss.

    The MPS depends only on the training data + MPS hyperparameters (not on extension/seed), so the
    whole sweep shares cache entries. Pre-training once (see src/__main__.pretrain_mps) before the
    parallel fan-out means the seed-workers only ever read this cache — no training race.
    """
    cache_dir = mps_cache_dir(cfg, X_train)
    circuit_path = os.path.join(cache_dir, "circuit.qpy")
    if os.path.exists(circuit_path):
        with open(circuit_path, "rb") as file:
            return qpy.load(file)[0]
    return train_mps(cfg, X_train, cache_dir)

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


def compute_split(cfg: DictConfig, dataloader: DataLoader) -> tuple:
    """Compute the leakage-safe 3-way split once, using the per-run seed for BAS holdout."""
    return dataloader.train_val_test_split(
        cfg["train_split"], cfg["val_split"],
        seed=cfg["random_seed"], bas_split_mode=cfg["bas_split_mode"])


def _coerce(value: str):
    """Best-effort int/float coercion for a CLI override value; falls back to the raw string."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def build_sweep_config(cfg: DictConfig) -> dict:
    """Build a wandb sweep_config describing the actual search space of this invocation.

    The real grid is owned by Hydra (--multirun key=v1,v2,...), not wandb's own search engine, so
    this only DOCUMENTS that grid for the wandb Sweep object/UI (parallel-coordinates, filtering) —
    it is never used to drive execution. Detected from sys.argv: any `key=v1,v2,...` override
    (Hydra's multirun list syntax) becomes a swept parameter; `key=[1,2]` (a literal list value,
    e.g. sigmas) is left alone. The per-seed `random_seed` values (derived from
    initial_random_seed + runs_batch_size, not a literal CLI override) are added explicitly.
    """
    parameters = {}
    for arg in sys.argv[1:]:
        if arg.startswith("-") or "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        if value.startswith("[") and value.endswith("]"):
            continue  # a literal list value (e.g. sigmas=[1.0]), not a multirun sweep list
        if "," in value:
            parameters[key] = {"values": [_coerce(v) for v in value.split(",")]}

    seeds = [cfg["initial_random_seed"] + i for i in range(cfg["runs_batch_size"])]
    parameters["random_seed"] = {"values": seeds}

    return {"method": "grid", "parameters": parameters}


def get_or_create_wandb_sweep(cfg: DictConfig) -> str:
    """Return the shared wandb sweep_id for this (multi)run, creating it on first use.

    Call this ONCE per process, before fanning out parallel seed-workers — it sets the
    WANDB_SWEEP_ID environment variable, which every subsequent `wandb.init()` call (including in
    spawned ProcessPoolExecutor children, which inherit the parent's environment) picks up
    automatically to join the same real wandb Sweep, with no need for `wandb.agent()`.

    Idempotent via the env var: if WANDB_SWEEP_ID is already set — because a previous Hydra job in
    this same multirun process already created it, or because it was exported by a shell launcher
    (see scripts/sweep.sh) — it's reused as-is and no new sweep is created.

    Only real wandb.sweep() calls happen when wandb_mode == "online" (it requires the backend);
    for offline/disabled runs a local synthetic id is used instead so local/test runs need no
    network access.

    If WANDB_SWEEP_ID_FILE is set (used by scripts/sweep.sh for multi-node SLURM array launches,
    where separate nodes are separate processes with no shared env to inherit from), the freshly
    created id is also written to that path on a shared filesystem, so other nodes can poll for it
    and export it themselves instead of each independently creating (and colliding on) their own
    sweep. Only written on actual creation, never on the early-return reuse path above.
    """
    if os.environ.get("WANDB_SWEEP_ID"):
        return os.environ["WANDB_SWEEP_ID"]

    if cfg["wandb_mode"] == "online":
        import wandb
        entity = cfg["wandb_entity"]
        sweep_id = wandb.sweep(build_sweep_config(cfg), project=cfg["wandb_project"],
                               entity=entity if entity else None)
    else:
        sweep_id = f"local_{datetime.now():%Y%m%d_%H%M%S}_{socket.gethostname()}"

    os.environ["WANDB_SWEEP_ID"] = sweep_id

    sweep_id_file = os.environ.get("WANDB_SWEEP_ID_FILE")
    if sweep_id_file:
        with open(sweep_id_file, "w") as f:
            f.write(sweep_id)

    return sweep_id


def _init_wandb(cfg: DictConfig, group: str, sweep_id: str):
    """Initialize a wandb run for one seed; returns the run (or None if disabled).

    Joining the sweep is done via an EXPLICIT `settings=wandb.Settings(sweep_id=...)` override,
    not by relying on wandb.init() picking up the WANDB_SWEEP_ID env var on its own. That env-var
    path is unreliable here: wandb.sweep() (in get_or_create_wandb_sweep) triggers a login call
    that snapshots os.environ into a process-wide Settings singleton BEFORE we set WANDB_SWEEP_ID,
    so every later wandb.init() in that process (e.g. every sequential Hydra job when
    runs_batch_size==1, since BasicLauncher reuses one process) would silently reuse that
    stale, sweep-less snapshot instead of re-reading the env var — this is exactly what produced
    runs that were created but not attached to the sweep. Passing sweep_id explicitly here is a
    per-call override applied on top of that singleton, so it's correct regardless of caching.
    """
    import wandb
    entity = cfg["wandb_entity"]
    return wandb.init(
        project=cfg["wandb_project"],
        entity=entity if entity else None,
        group=group,
        # no explicit `name`: let wandb assign its default generated name — the swept params
        # (group) and seed are already stored in config and don't need to be baked into it.
        job_type="train",
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg["wandb_mode"],
        settings=wandb.Settings(sweep_id=sweep_id),
        reinit=True,
    )


def _configure_worker_logging(output_dir: str, seed: int) -> None:
    """Configure logging inside a (possibly spawned) worker process.

    ProcessPoolExecutor's default "spawn" start method (used on macOS/Windows, and available on
    Linux) starts each worker as a fresh interpreter that does NOT inherit the parent's logging
    config — hydra's INFO-level setup only exists in the parent, so without this, every
    logger.info() call in a parallel (runs_batch_size > 1) run is silently dropped below WARNING.
    When train_worker is instead called directly in the parent process (runs_batch_size == 1, no
    pool), the "QCBM" logger already inherits hydra's INFO level, so this is a no-op.

    We check/configure the "QCBM" logger specifically (not root): some imported library (observed:
    wandb) attaches its own handler to the root logger even in a fresh spawned process, so
    `root.hasHandlers()` is not a reliable signal — but that handler sits at the default WARNING
    level, so `isEnabledFor(INFO)` on our own logger correctly detects whether INFO records would
    actually get through.

    Adds a console handler (seed-tagged, since parallel workers interleave on stdout) and a per-seed
    log file under the run's output directory, so training progress is always visible somewhere.
    """
    logger = logging.getLogger("QCBM")
    if logger.isEnabledFor(logging.INFO):
        return  # INFO already reaches a handler (e.g. hydra configured this in the parent process)

    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid double/mis-formatted output via whatever root already has

    formatter = logging.Formatter(
        f"[%(asctime)s][seed={seed}][%(name)s][%(levelname)s] - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = os.path.join(output_dir, f"qcbm_seed{seed}")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def train_worker(cfg_container: dict, seed: int, worker_index: int, group: str, output_dir: str) -> None:
    """Top-level, picklable worker for ProcessPoolExecutor (spawn-safe).

    Pins per-worker resources (GPU / threads), sets the run's seed, then trains one QCBM. Defined
    here (not in __main__) so spawned child processes can import it.
    """
    _configure_worker_logging(output_dir, seed)

    cfg = OmegaConf.create(cfg_container)
    cfg.random_seed = seed

    n_gpus = int(cfg.get("gpus_per_node", 0) or 0)
    if n_gpus > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_index % n_gpus)

    workers = max(1, int(cfg.get("runs_batch_size", 1)))
    n_cpu = os.cpu_count() or 1
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, n_cpu // workers)))

    setup_and_train_qcbm(cfg, group=group, output_dir=output_dir)


def setup_and_train_qcbm(cfg: DictConfig, group: str = "single", output_dir: str = "."):
    """Train one QCBM for a single (already-seeded) config; one wandb run per call.

    output_dir is the hydra run directory, passed explicitly because spawned worker processes do not
    have an initialized HydraConfig (and version_base=None does not chdir into the run dir).
    """

    logger = logging.getLogger("QCBM")
    start_time = time.time()

    # Idempotent: normally already created/set by __main__.py before the parallel fan-out (so
    # every seed-worker inherits the same WANDB_SWEEP_ID); calling it again here is a no-op in that
    # case, but also makes this function correct standalone (e.g. called directly, no __main__.py).
    sweep_id = get_or_create_wandb_sweep(cfg)
    logger.info(f"Program started (seed={cfg['random_seed']}, group={group}, sweep_id={sweep_id})")

    run = _init_wandb(cfg, group, sweep_id)

    try:
        # Setup dataloader and 3-way split (computed once, reused for MPS + QCBM)
        dataloader = setup_dataloader(cfg)
        X_train, X_val, X_test, X_train_count, X_val_count, X_test_count = compute_split(cfg, dataloader)

        # Shared MPS (trained once upstream; cache hit here)
        circuit_mps = get_or_train_mps(cfg, X_train)

        # Circuit extensions
        circuit_ext, init_params = setup_circuit_extensions(cfg, circuit_mps, X_train)

        # per-seed output directory under the hydra run dir
        save_dir = os.path.join(output_dir, f"qcbm_seed{cfg['random_seed']}")
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/ext_circuit.qpy", "wb") as file:
            qpy.dump(circuit_ext, file)

        # Transpile only for the real-device backend
        if cfg["simulator"] == "aer_kawasaki":
            circuit = transpile_circuit(circuit_ext, "service")
        else:
            circuit = circuit_ext.copy()

        # Train QCBM
        sampler, backend, use_parameter_binds = setup_qiskit_simulator(cfg)
        qcbm = QCBM(sampler, backend, circuit, init_params,
                    cfg["adam_learning_rate"], cfg["finite_diff_epsilon"], cfg["gradient_workers"],
                    use_parameter_binds=use_parameter_binds)
        qcbm.stochastic_gradient_descent(
            X_train, X_train_count, X_val_count, X_test_count,
            cfg["iterations"], cfg["N_shots"], cfg["mmd_batch_size"],
            cfg["loss_func"], cfg["sigmas"],
            eval_every=cfg["eval_every"], model_selection_metric=cfg["model_selection_metric"],
            wandb_run=run)

        # Save model + checkpoint
        qcbm.save(save_dir)

        # Upload artifact and record best-model summary for the selection step
        if run is not None:
            import wandb
            run.summary["best_mmd_val"] = qcbm.best_metric
            run.summary["best_iter"] = qcbm.best_iter
            run.summary["total_measurements"] = qcbm.total_measurements
            artifact = wandb.Artifact(f"qcbm_{run.id}", type="model",
                                      metadata={"seed": cfg["random_seed"], "group": group})
            artifact.add_dir(save_dir)
            run.log_artifact(artifact, aliases=[f"seed{cfg['random_seed']}"])

        logger.info("Program finished")
        logger.info(f"Program execution time: {round((time.time() - start_time) / 60, 2)} minutes")
    finally:
        if run is not None:
            run.finish()
