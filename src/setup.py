import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
import contextlib
import logging
import signal
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
from src import wandb_logging
from src.qcbm import QCBM
from src.mps import MPS
from src.data import DataLoader, BAS, JGB
from src.extension import compose_parameterized_circuit, linear_topology, all_to_all_topology, nearest_neighbor_topology, metric_based_topology, chow_liu_topology, extend_circuit, random_topology, select_threshold
from src.utils import mutual_info_matrix, feature_distance_matrix
from src.decompositon import mps2circuit


# Set by get_or_create_wandb_sweep when THIS process creates the sweep; read by finish_owned_sweep to
# close it out at exit. Empty in every other process (workers, and array tasks that inherited the id).
_OWNED_SWEEP: dict = {}


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

    Keyed on the MPS-relevant hyperparameters plus a hash of the actual training data. Since the
    train split is derived from initial_random_seed (see compute_split), all seeds in a sweep share
    one train set and hence one cache entry; genuinely different train sets (different dataset /
    qubits / MPS hyperparameters, or a different initial_random_seed) get their own.
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

def _metric_based_connections(X_train: pd.DataFrame, extension_metric: str, threshold_rule: str,
                              threshold: float = None) -> tuple:
    """Distance matrix + threshold + the resulting metric_based edges. The threshold is the explicit
    cfg.circuit.threshold if set, otherwise auto-selected via threshold_rule (knee or percolation).
    Shared by the metric_based and random branches of setup_circuit_extensions -- random is sized to
    match this exactly, so it's a fair random baseline for that comparison."""
    dist = feature_distance_matrix(X_train, extension_metric)
    threshhold = threshold if threshold is not None else select_threshold(dist, threshold_rule)
    return metric_based_topology(dist, threshhold), threshhold


def setup_circuit_extensions(cfg: DictConfig, mps_circuit: QuantumCircuit, X_train: pd.DataFrame) -> QuantumCircuit:

    # config parameters
    n_qubits = cfg.data.N_qubits
    dataset = cfg.data.dataset
    extension = cfg.circuit.extension
    extension_metric = cfg.circuit.extension_metric
    threshold_rule = cfg.circuit.threshold_rule
    threshold = cfg.circuit.threshold
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

    # metric based extension: threshold is auto-selected via cfg.circuit.threshold_rule -- either the
    # knee of the connections-vs-threshold curve (extension.knee_threshold) or the bond-percolation
    # threshold (extension.percolation_threshold) -- rather than hand-tuned, so it adapts to the
    # dataset/metric.
    elif extension == "metric_based":
        extension_connections, threshhold = _metric_based_connections(X_train, extension_metric, threshold_rule, threshold)
        source = "explicit" if threshold is not None else f"auto @ {threshold_rule}"
        logger.info(f"metric_based ({extension_metric}) threshold ({source}) = {threshhold:.4f}")
        extended_circuit = extend_circuit(mps_circuit, init_connections, extension_connections)

    # chow-liu dependency-tree extension: the maximum-mutual-information spanning tree over the
    # feature-bits. Same dependency signal as the metric-based method, but sparsified into the
    # optimal n-1-edge tree instead of thresholded -- a parameter-free, connected backbone.
    elif extension == "chow_liu":
        affinity = mutual_info_matrix(np.asarray(X_train))
        extension_connections = chow_liu_topology(affinity)
        extended_circuit = extend_circuit(mps_circuit, init_connections, extension_connections)

    # random extension: sized to exactly match the number of NEW connections metric_based would add
    # (same dataset/metric), so it's a fair random baseline for that comparison rather than an
    # arbitrarily chosen count.
    elif extension == "random":
        metric_connections, _ = _metric_based_connections(X_train, extension_metric, threshold_rule, threshold)
        n_random_extensions = len(set(metric_connections) - set(init_connections))
        logger.info(f"random extension: matching metric_based ({extension_metric}) connection "
                    f"count = {n_random_extensions}")
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
        dataset = JGB(cfg.data.N_qubits, cfg.data.N_features, cfg.data.quantizer)

    return DataLoader(dataset)


def compute_split(cfg: DictConfig, dataloader: DataLoader) -> tuple:
    """Compute the leakage-safe 3-way split once.

    The split is derived from the sweep's ``initial_random_seed``, NOT the per-run ``random_seed``,
    so every run in a sweep sees an identical train/val/test split -- and therefore an identical
    data-driven topology (metric_based/chow_liu) and MPS. """
    return dataloader.train_val_test_split(
        cfg.data.train_split, cfg.data.val_split,
        seed=cfg.sweep.initial_random_seed, bas_split_mode=cfg.data.bas_split_mode)


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

    On creation the sweep is also moved out of PENDING into RUNNING, and this process is recorded as
    the sweep's OWNER (see finish_owned_sweep) -- a sweep driven by Hydra has no wandb agent to
    advance its state, so nothing else ever would.
    """
    global _OWNED_SWEEP

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

    _OWNED_SWEEP = {"id": sweep_id, "entity": cfg.logging.wandb_entity,
                    "project": cfg.logging.wandb_project, "mode": cfg.logging.wandb_mode}
    wandb_logging.set_sweep_state(sweep_id, "RUNNING", entity=_OWNED_SWEEP["entity"],
                                  project=_OWNED_SWEEP["project"], mode=_OWNED_SWEEP["mode"])
    return sweep_id


def finish_owned_sweep() -> None:
    """Mark the sweep FINISHED, if this process is the one that created it. Never raises.

    Called once the launcher has joined all of its runs (src/__main__.py). Without it the sweep stays
    in whatever state it was last put in and the dashboard never shows it as done -- there is no
    wandb agent here to close it out.

    Only the CREATING process does this, so in a multi-node SLURM array the nodes that merely
    inherited WANDB_SWEEP_ID (via the shared sweep-id file) never touch the state. The creating node
    can still finish before the others; FINISHED means "start no new runs, let running ones finish",
    so their runs keep reporting normally.
    """
    if not _OWNED_SWEEP:
        return
    wandb_logging.set_sweep_state(_OWNED_SWEEP["id"], "FINISHED", entity=_OWNED_SWEEP["entity"],
                                  project=_OWNED_SWEEP["project"], mode=_OWNED_SWEEP["mode"])


def _configure_worker_logging(output_dir: str, seed: int, log_level: str = "INFO") -> None:
    """Configure logging inside a (possibly spawned) worker process.

    ProcessPoolExecutor's default "spawn" start method (used on macOS/Windows, and available on
    Linux) starts each worker as a fresh interpreter that does NOT inherit the parent's logging
    config — hydra's INFO-level setup only exists in the parent, so without this, every
    logger.info() call in a parallel (runs_batch_size > 1) run is silently dropped below WARNING.
    When train_worker is instead called directly in the parent process (runs_batch_size == 1, no
    pool), the "QCBM" logger already inherits hydra's INFO level from that setup instead.

    `log_level` (cfg.logging.log_level) is the single config-driven verbosity control for the
    "QCBM" logger -- the only logger every module in this codebase logs through -- so setting it
    here is enough to quiet (e.g. "WARNING", for real/production sweeps) or restore ("INFO") every
    training log line, in EITHER process (parent or spawned worker): the isEnabledFor(INFO) probe
    just below runs BEFORE this is applied, so it still correctly detects "parent (hydra already
    configured)" vs "fresh spawned worker" regardless of which level was actually requested.

    We check/configure the "QCBM" logger specifically (not root): some imported library (observed:
    wandb) attaches its own handler to the root logger even in a fresh spawned process, so
    `root.hasHandlers()` is not a reliable signal — but that handler sits at the default WARNING
    level, so `isEnabledFor(INFO)` on our own logger correctly detects whether INFO records would
    actually get through.

    Adds a console handler (seed-tagged, since parallel workers interleave on stdout) and a per-seed
    log file under the run's output directory, so training progress is always visible somewhere.
    """
    logger = logging.getLogger("QCBM")
    already_configured = logger.isEnabledFor(logging.INFO)
    logger.setLevel(getattr(logging, log_level.upper()))
    if already_configured:
        return  # a handler already exists (e.g. hydra configured this in the parent process)

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


def train_worker(cfg_container: dict, seed: int, worker_index: int, combo: str, output_dir: str) -> None:
    """Top-level, picklable worker for ProcessPoolExecutor (spawn-safe).

    Pins per-worker resources (GPU / threads), sets the run's seed, then trains one QCBM. Defined
    here (not in __main__) so spawned child processes can import it.

    Thread budget: cfg.sweep.threads_per_run is resolved by __main__ (plan_resources) and the same
    value is exported to the environment there before the pool is spawned, so a fresh worker's BLAS
    already inits at the right size. We re-assert the env here (harmless, and covers a direct
    non-pooled call), cap Aer's own thread pool via cfg.sweep.threads_per_run
    (see setup_qiskit_simulator), and size the gradient ThreadPoolExecutor to match.
    """
    cfg = OmegaConf.create(cfg_container)
    cfg.sweep.random_seed = seed
    _configure_worker_logging(output_dir, seed, cfg.logging.log_level)

    # GPU NOTE: round-robin GPU pinning is stubbed here; CPU sweeps leave gpus_per_node=0. A real
    # GPU mode would also skip the CPU thread-capping below (a GPU run wants full BLAS for its host
    # side, and its heavy work is on-device).
    n_gpus = int(cfg.sweep.gpus_per_node or 0)
    if n_gpus > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_index % n_gpus)

    threads_per_run = max(1, int(cfg.sweep.threads_per_run or 1))
    apply_thread_env(threads_per_run)
    cfg.aer.gradient_workers = threads_per_run  # per-parameter gradient ThreadPoolExecutor size

    setup_and_train_qcbm(cfg, combo=combo, output_dir=output_dir)


def _sigterm_as_interrupt():
    """Make SIGTERM raise KeyboardInterrupt; returns the previous handler (for _restore_sigterm).

    Python's default SIGTERM disposition kills the process outright: no exception, no `finally`, so a
    `scancel`ed or user-killed run would lose everything it had trained -- nothing saved to disk, no
    artifact. Turning it into an exception routes termination through the same unwinding path as a
    crash or Ctrl-C, which is what lets _finalize_run below still checkpoint and upload.

    SIGINT is left alone (Python already raises KeyboardInterrupt for it). SIGKILL cannot be caught
    by anyone, so a `kill -9` (or SLURM's post-KillWait kill) still loses the run -- which is why the
    interrupted path skips the artifact stagger: whatever grace period we have may be seconds long.

    signal.signal only works in a process's main thread; a worker calling this from elsewhere gets a
    ValueError, which is swallowed (the run then behaves as it did before this existed).
    """
    def _handler(signum, frame):
        raise KeyboardInterrupt(f"terminated by signal {signum}")

    try:
        return signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError, AttributeError) as exc:
        logging.getLogger("QCBM").debug(f"[finalize] no SIGTERM handler installed: {exc!r}")
        return None


def _restore_sigterm(previous) -> None:
    """Undo _sigterm_as_interrupt (a pool worker outlives one run and must not keep the handler)."""
    if previous is None:
        return
    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGTERM, previous)


def _finalize_run(cfg, qcbm, save_dir: str, wandb_run, combo: str, status: str) -> None:
    """Write the checkpoints and push the model artifact. Runs on EVERY exit path; never raises.

    `status` is "completed", "crashed" or "interrupted" -- the last two mean training stopped early,
    so the checkpoints hold whatever the run had reached (best_params is the best model seen so far;
    checkpoint_meta.json's final_iter says how far it actually got). It is recorded both in the run
    summary and in the artifact metadata so an early-stopped model is never mistaken for a full one.

    Errors here are logged and swallowed: this is called from a `finally`, and raising would mask the
    exception that stopped training in the first place.
    """
    logger = logging.getLogger("QCBM")
    if qcbm is None or not save_dir:
        return                                     # died before there was a model to save

    try:
        qcbm.save(save_dir)
    except Exception as exc:
        logger.error(f"[finalize] could not save model to {save_dir}: {exc!r}")
        return                                     # nothing on disk -> nothing to upload

    # Record the best-model summary for the selection step, plus the local save_dir as the
    # fallback source for checkpoints if the artifact upload below is refused
    # (benchmark.load_checkpoint uses it). One batched update, not several server round-trips.
    wandb_run.summary({"best_mmd_val": qcbm.best_metric,
                       "best_iter": qcbm.best_iter,
                       "total_measurements": qcbm.total_measurements,
                       "iterations_run": qcbm.iterations_run,
                       "measurements_per_step": qcbm.measurements_per_step,
                       # iterations_run is the PLANNED length (fixed when SGD starts, budget-derived);
                       # final_iter is how far this run actually got, which differs when it died early
                       "final_iter": len(qcbm.parameter_hist) - 1,
                       "run_status": status,
                       "save_dir": os.path.abspath(save_dir)})

    # Model artifact -- the heaviest per-run burst of API calls in a sweep (create + per-file
    # upload + commit), and runs doing identical work all reach it at the same time, so it is
    # staggered like init and skippable entirely (logging.wandb_log_artifacts); the same files
    # stay on disk in save_dir either way, which benchmark.load_checkpoint falls back to.
    if not (cfg.logging.wandb_log_artifacts and wandb_run.run_id is not None):
        return
    if status == "completed":
        wandb_logging.stagger(cfg, cfg.sweep.random_seed, "artifact")   # see docstring: not when dying
    files = [p for p in (f"{save_dir}/circuit.qpy", f"{save_dir}/best_params.npy",
                         f"{save_dir}/final_params.npy") if os.path.exists(p)]
    aliases = [f"seed{cfg.sweep.random_seed}"] + ([] if status == "completed" else [status])
    wandb_run.log_artifact(
        f"qcbm_{wandb_run.run_id}",
        files,
        metadata={"seed": cfg.sweep.random_seed, "combo": combo, "status": status,
                  "best_iter": qcbm.best_iter, "final_iter": len(qcbm.parameter_hist) - 1},
        aliases=aliases)


def setup_and_train_qcbm(cfg: DictConfig, combo: str = "single", output_dir: str = "."):
    """Train one QCBM for a single (already-seeded) config; one wandb run per call.

    `combo` is the swept-parameter combination string, used only for local log lines / artifact
    metadata to tell parallel runs apart in the .err file -- it is NOT sent to wandb as a group (the
    swept params live in the run's config; plotting/benchmarking filter on those directly).

    output_dir is the hydra run directory, passed explicitly because spawned worker processes do not
    have an initialized HydraConfig (and version_base=None does not chdir into the run dir).
    """

    logger = logging.getLogger("QCBM")
    start_time = time.time()

    # Idempotent: normally already created/set by __main__.py before the parallel fan-out (so
    # every seed-worker inherits the same WANDB_SWEEP_ID); calling it again here is a no-op in that
    # case, but also makes this function correct standalone (e.g. called directly, no __main__.py).
    sweep_id = get_or_create_wandb_sweep(cfg)
    logger.info(f"Program started (seed={cfg.sweep.random_seed}, combo={combo}, sweep_id={sweep_id})")

    # Staggered + retried init, then a buffered logger that chunks metric pushes and swallows wandb
    # errors -- so neither a 429 at init nor one mid-training can take the training run down.
    run = wandb_logging.init_run(cfg, sweep_id, cfg.sweep.random_seed)
    wandb_run = wandb_logging.logger_for(cfg, run, label=f"seed{cfg.sweep.random_seed}")

    # Set as soon as they exist so the finalizer in the `finally` below can checkpoint + upload
    # whatever training reached, on any exit path (see _finalize_run / _sigterm_as_interrupt).
    qcbm = None
    save_dir = ""
    status = "completed"
    previous_sigterm = _sigterm_as_interrupt()

    try:
        # Setup dataloader and 3-way split (computed once, reused for MPS + QCBM)
        dataloader = setup_dataloader(cfg)
        X_train, X_val, X_test, X_train_count, X_val_count, X_test_count = compute_split(cfg, dataloader)

        # Shared MPS (trained once upstream; cache hit here)
        circuit_mps = get_or_train_mps(cfg, X_train)

        # Linear (unextended) baseline: the shared pre-extension starting point common to EVERY
        # connectivity. Compose it now, before setup_circuit_extensions -- extend_circuit mutates
        # circuit_mps in place (appends the extension gates), so this must be captured first.
        # compose_parameterized_circuit copies internally, so circuit_mps stays pristine for the
        # extension step below. Sampled at step 0 in training for a common baseline across runs.
        linear_circuit, linear_params = compose_parameterized_circuit(circuit_mps)
        linear_circuit.measure_all()

        # Circuit extensions
        circuit_ext, init_params = setup_circuit_extensions(cfg, circuit_mps, X_train)

        # per-seed output directory under the hydra run dir
        save_dir = os.path.join(output_dir, f"qcbm_seed{cfg.sweep.random_seed}")
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/ext_circuit.qpy", "wb") as file:
            qpy.dump(circuit_ext, file)

        # Transpile only for the real-device backend (both the training circuit and the baseline)
        if cfg.ibm.simulator == "aer_kawasaki":
            circuit = transpile_circuit(circuit_ext, "service")
            linear_circuit = transpile_circuit(linear_circuit, "service")
        else:
            circuit = circuit_ext.copy()

        # Train QCBM
        sampler, backend, use_parameter_binds = setup_qiskit_simulator(cfg)
        qcbm = QCBM(sampler, backend, circuit, init_params,
                    cfg.qcbm.adam_learning_rate, cfg.qcbm.finite_diff_epsilon, cfg.aer.gradient_workers,
                    use_parameter_binds=use_parameter_binds)
        qcbm.stochastic_gradient_descent(
            X_train, X_train_count, X_val_count, X_test_count,
            cfg.qcbm.iterations, cfg.qcbm.N_shots, cfg.qcbm.mmd_batch_fraction,
            cfg.qcbm.loss_func, cfg.qcbm.sigmas,
            eval_every=cfg.qcbm.eval_every, model_selection_metric=cfg.qcbm.model_selection_metric,
            mode=cfg.qcbm.mode,
            measurement_budget=cfg.qcbm.measurement_budget,
            wandb_run=wandb_run,
            dataset_kind=cfg.data.dataset,
            valid_patterns=(dataloader.dataset.binary if cfg.data.dataset == "BAS" else None),
            baseline_circuit=linear_circuit, baseline_params=linear_params,
            baseline_seed=cfg.sweep.initial_random_seed)

        logger.info("Program finished")
        logger.info(f"Program execution time: {round((time.time() - start_time) / 60, 2)} minutes")
    except (KeyboardInterrupt, SystemExit) as exc:
        # Ctrl-C, `scancel`, or any other SIGTERM (see _sigterm_as_interrupt). Re-raised so the
        # process still dies -- but only after the finalizer below has checkpointed and uploaded.
        status = "interrupted"
        logger.warning(f"Run interrupted ({exc!r}) after "
                       f"{round((time.time() - start_time) / 60, 2)} minutes -- saving and uploading "
                       f"the model trained so far before exiting.")
        raise
    except Exception:
        status = "crashed"
        logger.exception("Run crashed -- saving and uploading the model trained so far.")
        raise
    finally:
        # Save + summary + artifact happen HERE, not on the success path, so a run that crashes or is
        # killed mid-training still leaves a loadable checkpoint on disk and in wandb. SIGTERM is
        # IGNORED for the duration: a second `scancel` (the natural reaction to a job that doesn't
        # die instantly) would otherwise kill the process exactly during the upload it is waiting
        # for. Ctrl-C/SIGINT is deliberately left working as the escape hatch from a stuck upload.
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            _finalize_run(cfg, qcbm, save_dir, wandb_run, combo, status)
            wandb_run.finish()  # final flush of buffered rows, then close the run
        finally:
            _restore_sigterm(previous_sigterm)
