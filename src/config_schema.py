"""Structured (typed) Hydra config schema for src/conf/config.yaml.

Importing this module registers `Config` with Hydra's ConfigStore under the name
"config_schema" (see src/conf/config.yaml's `defaults` list), so every composed run config is
validated against these types/field names -- a typo'd or wrong-typed override fails fast at
startup instead of surfacing as a confusing AttributeError deep in training.

Fields are grouped into nested sub-configs (data, ibm, aer, mps, circuit, qcbm, sweep, logging)
matching the section comments in config.yaml, e.g. `cfg.mps.cutoff` rather than `cfg.cutoff`.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


@dataclass
class DataConfig:
    N_qubits: int = 12                     # number of qubits
    dataset: str = "JGB"                   # BAS, JGB
    train_split: float = 0.7               # pct of trainset samples between 0 and 1
    val_split: float = 0.15                # pct of validation samples; test = 1 - train_split - val_split
    bas_split_mode: str = "full_support"   # full_support (all patterns in every split) or holdout (seeded disjoint partition)
    width: int = 3                         # width of the BAS image
    height: int = 3                        # height of the BAS image
    N_features: int = 3                    # number of features for JGB dataset, 3 or 4


@dataclass
class IbmConfig:
    simulator: str = "aer_statevec_cpu"    # aer_statevec_cpu, aer_statevec_gpu, aer_kawasaki
    transpiler: str = "service"            # local or service (not used in statevector simulation)
    optimization_level: int = 3            # optimization for running the quantum circuit: 0, 1, 2, 3 (not used in statevector simulation)
    ai_transpiler: bool = True             # True or False (not used in statevector simulation)


@dataclass
class AerConfig:
    aer_max_parallel_experiments: int = 0       # 0 = auto (across bound circuits); >0 caps parallel experiments
    aer_max_parallel_shots: int = 0             # 0 = auto; setting 1 disables batched parameter execution (much slower)
    aer_batched_shots_gpu_max_qubits: int = 16  # GPU batched-shots cutoff (only for aer_statevec_gpu)
    aer_blocking_qubits_threshold: int = 28     # enable multi-GPU blocking at/above this qubit count
    aer_blocking_qubits: int = 24               # chunk size for multi-GPU blocking
    gradient_workers: int = 8                   # threads for the per-parameter gradient loop


@dataclass
class MpsConfig:
    cutoff: float = 5.0e-05                # cutoff precision for the MPS
    descenting_step_length: float = 0.05   # step length for the descent
    descent_steps: int = 10                # number of steps for the descent
    train_loops: int = 10                  # number of loops for training


@dataclass
class CircuitConfig:
    extension: str = "metric_based"        # none, all_to_all, nearest_neighbor, metric_based, chow_liu, random
    extension_metric: str = "hamming"      # hamming, varinfo
    threshold_rule: str = "knee"           # knee, percolation -- how metric_based auto-selects its threshold
    threshold: Optional[float] = None      # explicit metric_based threshold; overrides threshold_rule unless None


@dataclass
class QcbmConfig:
    mode: str = "iterations"                 # iterations, measurements
    measurement_budget: int = 1_000_000_000  # circuit measurements per run (mode=measurements)
    iterations: int = 10                   # number of training iterations (mode=iterations)
    mmd_batch_fraction: float = 0.0        # 0 = full train set; (0,1] = that FRACTION of the train set per step
    N_shots: int = 1000                    # number of shots in sampling
    loss_func: str = "MMD"                 # MMD, KL (KL not working yet)
    sigmas: List[float] = field(default_factory=lambda: [1.0])  # Bandwidth for MMD Kernel
    finite_diff_epsilon: float = 1.0e-8    # epsilon used in finite difference sampling for KL (does not work yet)
    adam_learning_rate: float = 0.01       # initial learning rate for ADAM
    eval_every: int = 1                    # compute full train/val/test MMD every k iterations (+ final)
    model_selection_metric: str = "mmd_val"  # metric used to pick the best checkpoint (mmd_val recommended)


@dataclass
class SweepConfig:
    runs_batch_size: int = 1               # number of auto-seeded training runs per hydra job (seeds)
    initial_random_seed: int = 42          # base seed; run i in a batch uses initial_random_seed + i
    gpus_per_node: int = 0                 # >0 round-robins runs across GPUs via CUDA_VISIBLE_DEVICES; 0 = CPU mode
    max_parallel_runs: int = 0             # concurrent training runs across the whole sweep (0 = auto)
    threads_per_run: int = 0               # OMP/BLAS/Aer/gradient threads per run (0 = auto, resolved at runtime)

    # Effective per-run seed, derived from initial_random_seed + batch index and assigned before
    # training starts (see src/__main__.py::pretrain_mps, src/setup.py::train_worker) -- never read
    # before that assignment, so no meaningful default exists.
    random_seed: Optional[int] = None


@dataclass
class LoggingConfig:
    wandb_project: str = "qcbm-circuit-design"
    wandb_entity: Optional[str] = None     # null = your default wandb entity
    wandb_mode: str = "online"             # online, offline, disabled
    log_level: str = "INFO"                # INFO (verbose, default) or WARNING (quiet -- real/production runs)

    # --- wandb request-rate control (see src/wandb_logging.py) ---
    wandb_flush_every: int = 100           # buffered iteration rows per push (1 = push every iteration)
    wandb_flush_interval_s: float = 300.0  # also push if this long since the last push (slow runs)
    wandb_transmit_interval_s: float = 60.0  # wandb-core filestream transmit interval
    wandb_heartbeat_s: int = 30            # run keepalive interval; raising it cuts the request floor
    wandb_log_artifacts: bool = True       # per-run model artifact upload (needed by benchmark.py)

    # --- wandb robustness ---
    wandb_init_stagger_s: float = 0.2      # init jitter window = this x sweep.max_parallel_runs
    wandb_init_retries: int = 5            # extra wandb.init() attempts before training without wandb
    wandb_retry_max: int = 5               # retries per wandb call (also wandb-core's own retry cap)
    wandb_retry_wait_max_s: float = 60.0   # backoff cap for those retries
    wandb_max_failures: int = 5            # consecutive failed pushes before dropping wandb for a run
    wandb_init_timeout_s: float = 300.0    # wandb.init() timeout (default 90s is tight at high fan-out)
    wandb_service_wait_s: float = 120.0    # wait for the local wandb-core service (default 30s)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    ibm: IbmConfig = field(default_factory=IbmConfig)
    aer: AerConfig = field(default_factory=AerConfig)
    mps: MpsConfig = field(default_factory=MpsConfig)
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    qcbm: QcbmConfig = field(default_factory=QcbmConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)


def from_run_config(config: dict) -> DictConfig:
    """Wrap a wandb run's logged config dict (from src.wandb_logging.init_run, itself
    OmegaConf.to_container(cfg, resolve=True)) back into this schema, so callers get the same dot
    access (cfg.data.dataset) and validation as a live Hydra run, instead of dict indexing.

    wandb's public API already strips its own internal keys (_wandb, wandb_version) from
    Run.config, so this only ever sees the fields we logged ourselves.
    """
    return OmegaConf.merge(OmegaConf.structured(Config), config)
