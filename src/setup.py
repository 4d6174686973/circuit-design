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
    simulator = cfg.ibm.simulator

    if simulator in ("aer_statevec_cpu", "aer_statevec_gpu"):
        device = "GPU" if simulator == "aer_statevec_gpu" else "CPU"
        backend_options = {
            "method": "statevector",
            "device": device,
            "runtime_parameter_bind_enable": True,
            "max_parallel_experiments": cfg.aer.aer_max_parallel_experiments,
            "max_parallel_shots": cfg.aer.aer_max_parallel_shots,
            "seed_simulator": cfg.sweep.random_seed,
        }
        # Cap Aer's OpenMP pool to this run's thread budget so W parallel runs don't each grab all
        # cores (oversubscription). threads_per_run is resolved per-run by __main__/train_worker;
        # 0 (a bare single run that never went through the planner) leaves Aer's default (all cores).
        # GPU NOTE: on device="GPU" this CPU cap is irrelevant -- the heavy work runs on-device.
        threads_per_run = int(cfg.sweep.threads_per_run or 0)
        if device == "CPU" and threads_per_run > 0:
            backend_options["max_parallel_threads"] = threads_per_run
        if device == "GPU":
            backend_options["batched_shots_gpu"] = True
            backend_options["batched_shots_gpu_max_qubits"] = cfg.aer.aer_batched_shots_gpu_max_qubits
            if cfg.data.N_qubits >= cfg.aer.aer_blocking_qubits_threshold:
                backend_options["blocking_enable"] = True
                backend_options["blocking_qubits"] = cfg.aer.aer_blocking_qubits
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
    key = (f"{cfg.data.dataset}_q{cfg.data.N_qubits}_cut{cfg.mps.cutoff}"
           f"_dsl{cfg.mps.descenting_step_length}_ds{cfg.mps.descent_steps}"
           f"_tl{cfg.mps.train_loops}_{data_hash}")
    return to_absolute_path(os.path.join("outputs", "mps_cache", key))


def train_mps(cfg: DictConfig, X_train: np.ndarray, save_dir: str) -> QuantumCircuit:
    """Train an MPS on X_train, decompose to a PQC, and persist both to save_dir."""
    n_qubits = cfg.data.N_qubits

    mps = MPS(n_qubits)
    mps.left_cano()
    mps.designate_data(X_train)
    mps.init_cumulants()
    mps.cutoff = cfg.mps.cutoff
    mps.descenting_step_length = cfg.mps.descenting_step_length
    mps.descent_steps = cfg.mps.descent_steps
    mps.train(cfg.mps.train_loops, rec_cut=False)

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
    n_qubits = cfg.data.N_qubits
    n_random_extensions = cfg.circuit.N_random_extensions
    dataset = cfg.data.dataset
    extension = cfg.circuit.extension
    extension_metric = cfg.circuit.extension_metric
    extension_threshhold = cfg.circuit.extension_threshhold
    width = cfg.data.width
    height = cfg.data.height
    random_seed = cfg.sweep.random_seed

    logger = logging.getLogger('QCBM')

    # linear baseline connections
    init_order = list(range(cfg.data.N_qubits))
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
    if cfg.data.dataset == "BAS":
        dataset = BAS(cfg.data.width, cfg.data.height)
    elif cfg.data.dataset == "JGB":
        dataset = JGB(cfg.data.N_qubits, cfg.data.N_features)

    return DataLoader(dataset)


def compute_split(cfg: DictConfig, dataloader: DataLoader) -> tuple:
    """Compute the leakage-safe 3-way split once, using the per-run seed for BAS holdout."""
    return dataloader.train_val_test_split(
        cfg.data.train_split, cfg.data.val_split,
        seed=cfg.sweep.random_seed, bas_split_mode=cfg.data.bas_split_mode)


def _coerce(value: str):
    """Best-effort int/float coercion for a CLI override value; falls back to the raw string."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _grid_value_lists() -> dict:
    """Parse the Hydra multirun grid from sys.argv into {key: [values]}.

    Detects `key=v1,v2,...` (Hydra's multirun list syntax); a `key=[1,2]` literal list value
    (e.g. sigmas) is left out, since that is one value, not a sweep axis. Shared by
    build_sweep_config (for the wandb Sweep object) and count_grid_combos (for resource planning).
    """
    grid = {}
    for arg in sys.argv[1:]:
        if arg.startswith("-") or "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        if value.startswith("[") and value.endswith("]"):
            continue  # a literal list value (e.g. sigmas=[1.0]), not a multirun sweep list
        if "," in value:
            grid[key] = value.split(",")
    return grid


def count_grid_combos() -> int:
    """Number of Hydra multirun grid combinations in this invocation (product of axis lengths)."""
    combos = 1
    for values in _grid_value_lists().values():
        combos *= len(values)
    return combos


def build_sweep_config(cfg: DictConfig) -> dict:
    """Build a wandb sweep_config describing the actual search space of this invocation.

    The real grid is owned by Hydra (--multirun key=v1,v2,...), not wandb's own search engine, so
    this only DOCUMENTS that grid for the wandb Sweep object/UI (parallel-coordinates, filtering) —
    it is never used to drive execution. The per-seed `random_seed` values (derived from
    initial_random_seed + runs_batch_size, not a literal CLI override) are added explicitly.
    """
    parameters = {key: {"values": [_coerce(v) for v in values]}
                  for key, values in _grid_value_lists().items()}

    seeds = [cfg.sweep.initial_random_seed + i for i in range(cfg.sweep.runs_batch_size)]
    parameters["random_seed"] = {"values": seeds}

    return {"method": "grid", "parameters": parameters}


# --------------------------------------------------------------------------------------------------
# CPU resource planning for the whole sweep
#
# GPU NOTE (future work): everything below plans a *CPU* budget -- it splits cores across many
# concurrent CPU-bound runs. A GPU-efficient mode would be a different split (≈1 run per GPU, pinned
# via CUDA_VISIBLE_DEVICES, with each run's Aer on device="GPU"), and would also want the many small
# per-iteration numpy ops (kernel/gradient in src/cost.py, src/qcbm.py) moved onto the GPU (e.g.
# cupy) to avoid a CPU<->GPU round-trip every iteration -- otherwise the hybrid is dominated by
# transfer overhead. The swap points are marked "GPU NOTE" across plan_resources (below),
# apply_thread_env (below), train_worker / setup_qiskit_simulator, and __main__'s pool sizing.
# --------------------------------------------------------------------------------------------------
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def available_cpus() -> int:
    """Logical CPUs usable by this process, honoring SLURM/cgroup pinning where possible."""
    try:
        return len(os.sched_getaffinity(0))  # respects cpuset/SLURM --cpus-per-task on Linux
    except AttributeError:
        return os.cpu_count() or 1            # macOS / Windows fallback


def plan_resources(cfg: DictConfig) -> tuple:
    """Resolve (max_parallel_runs, threads_per_run) for the whole sweep from config + hardware.

    total_runs = (grid combos) x runs_batch_size. Precedence:
      - both configured (>0): used as-is (may intentionally over/undersubscribe).
      - only max_parallel_runs set: threads_per_run = cores // max_parallel_runs.
      - only threads_per_run set: max_parallel_runs = min(total_runs, cores // threads_per_run).
      - neither: max_parallel_runs = min(total_runs, cores); threads_per_run = cores // that.
    So a small sweep gives each run many threads (fast Aer sampling), while a large sweep trades
    threads for run-level parallelism -- always using ~all cores.

    GPU NOTE: for a GPU run this whole calculation changes (parallelism is bounded by GPU count/VRAM,
    not CPU cores); branch here on cfg.sweep.gpus_per_node when GPU support lands.
    """
    n_cpus = available_cpus()
    total_runs = max(1, count_grid_combos() * max(1, int(cfg.sweep.runs_batch_size)))
    mpr = int(cfg.sweep.max_parallel_runs or 0)
    tpr = int(cfg.sweep.threads_per_run or 0)

    if mpr <= 0:
        mpr = min(total_runs, n_cpus // tpr) if tpr > 0 else min(total_runs, n_cpus)
        mpr = max(1, mpr)
    if tpr <= 0:
        tpr = max(1, n_cpus // mpr)
    return mpr, tpr


def apply_thread_env(threads_per_run: int) -> None:
    """Cap BLAS/OpenMP threads per run via environment variables.

    Set in the PARENT before the spawn pool is created so each fresh worker inherits it *before* it
    imports numpy/scipy/Aer, which is the only reliable moment to size those libraries' thread pools
    (setting it after import is a no-op for already-initialized pools). With W parallel runs each
    capped at threads_per_run, total threads stay ~= cores instead of W*cores (oversubscription).
    """
    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(max(1, int(threads_per_run)))


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

    if cfg.logging.wandb_mode == "online":
        import wandb
        entity = cfg.logging.wandb_entity
        sweep_id = wandb.sweep(build_sweep_config(cfg), project=cfg.logging.wandb_project,
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
    entity = cfg.logging.wandb_entity
    # Overhead control: with many parallel runs, wandb's per-run background system-stats monitor
    # (a thread polling CPU/mem every few seconds + periodic network posts) and metadata/code/git
    # scans add up to N times the cost for no benefit here -- we only log our own scalar metrics.
    # Disable them. Kept: the actual metric logging (one buffered, non-blocking log() per iteration).
    settings = wandb.Settings(
        sweep_id=sweep_id,
        x_disable_stats=True,        # no per-run system-metrics monitor thread / posts
        x_disable_meta=True,         # skip machine/git/code metadata collection at init
        disable_git=True,
        disable_code=True,
    )
    return wandb.init(
        project=cfg.logging.wandb_project,
        entity=entity if entity else None,
        group=group,
        # no explicit `name`: let wandb assign its default generated name — the swept params
        # (group) and seed are already stored in config and don't need to be baked into it.
        job_type="train",
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.logging.wandb_mode,
        settings=settings,
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

    Thread budget: cfg.sweep.threads_per_run is resolved by __main__ (plan_resources) and the same
    value is exported to the environment there before the pool is spawned, so a fresh worker's BLAS
    already inits at the right size. We re-assert the env here (harmless, and covers a direct
    non-pooled call), cap Aer's own thread pool via cfg.sweep.threads_per_run
    (see setup_qiskit_simulator), and size the gradient ThreadPoolExecutor to match.
    """
    _configure_worker_logging(output_dir, seed)

    cfg = OmegaConf.create(cfg_container)
    cfg.sweep.random_seed = seed

    # GPU NOTE: round-robin GPU pinning is stubbed here; CPU sweeps leave gpus_per_node=0. A real
    # GPU mode would also skip the CPU thread-capping below (a GPU run wants full BLAS for its host
    # side, and its heavy work is on-device).
    n_gpus = int(cfg.sweep.gpus_per_node or 0)
    if n_gpus > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_index % n_gpus)

    threads_per_run = max(1, int(cfg.sweep.threads_per_run or 1))
    apply_thread_env(threads_per_run)
    cfg.aer.gradient_workers = threads_per_run  # per-parameter gradient ThreadPoolExecutor size

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
    logger.info(f"Program started (seed={cfg.sweep.random_seed}, group={group}, sweep_id={sweep_id})")

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
        save_dir = os.path.join(output_dir, f"qcbm_seed{cfg.sweep.random_seed}")
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/ext_circuit.qpy", "wb") as file:
            qpy.dump(circuit_ext, file)

        # Transpile only for the real-device backend
        if cfg.ibm.simulator == "aer_kawasaki":
            circuit = transpile_circuit(circuit_ext, "service")
        else:
            circuit = circuit_ext.copy()

        # Train QCBM
        sampler, backend, use_parameter_binds = setup_qiskit_simulator(cfg)
        qcbm = QCBM(sampler, backend, circuit, init_params,
                    cfg.qcbm.adam_learning_rate, cfg.qcbm.finite_diff_epsilon, cfg.aer.gradient_workers,
                    use_parameter_binds=use_parameter_binds)
        qcbm.stochastic_gradient_descent(
            X_train, X_train_count, X_val_count, X_test_count,
            cfg.qcbm.iterations, cfg.qcbm.N_shots, cfg.qcbm.mmd_batch_size,
            cfg.qcbm.loss_func, cfg.qcbm.sigmas,
            eval_every=cfg.qcbm.eval_every, model_selection_metric=cfg.qcbm.model_selection_metric,
            wandb_run=run)

        # Save model + checkpoint
        qcbm.save(save_dir)

        # Upload artifact and record best-model summary for the selection step.
        # Only circuit.qpy + best_params.npy are uploaded: they're the sole files needed to
        # reconstruct/sample the best checkpoint (src/benchmark.py::load_checkpoint) and aren't
        # representable as wandb scalar metrics. Everything else `qcbm.save()` writes locally
        # (losses.parquet, checkpoint_meta.json, full params.npy history) duplicates data already
        # logged as wandb metrics/summary/config (mmd_train/val/test, best_mmd_val, best_iter,
        # total_measurements, num_parameters, ...) and is not re-uploaded.
        if run is not None:
            import wandb
            run.summary["best_mmd_val"] = qcbm.best_metric
            run.summary["best_iter"] = qcbm.best_iter
            run.summary["total_measurements"] = qcbm.total_measurements
            artifact = wandb.Artifact(f"qcbm_{run.id}", type="model",
                                      metadata={"seed": cfg.sweep.random_seed, "group": group})
            artifact.add_file(f"{save_dir}/circuit.qpy")
            artifact.add_file(f"{save_dir}/best_params.npy")
            run.log_artifact(artifact, aliases=[f"seed{cfg.sweep.random_seed}"])

        logger.info("Program finished")
        logger.info(f"Program execution time: {round((time.time() - start_time) / 60, 2)} minutes")
    finally:
        if run is not None:
            run.finish()
