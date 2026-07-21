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
    extension: str = "metric_based"        # none, all_to_all, nearest_neighbor, metric_based, random
    N_random_extensions: int = 10          # number of random extensions to compare with metric_based: 10 for BAS, 13 for JGB
    extension_metric: str = "hamming"      # hamming, varinfo
    extension_threshhold: float = 0.5      # 0.5 for BAS + hamming, 0.95 for JGB + varinfo


@dataclass
class QcbmConfig:
    iterations: int = 10                   # number of training iterations
    mmd_batch_size: int = 1000             # 0 for using the full train set or positive integer for MMD mini-batch size
    N_shots: int = 1000                    # number of shots in sampling
    loss_func: str = "MMD"                 # MMD, KL (KL not working yet)
    sigmas: List[float] = field(default_factory=lambda: [1.0])  # Bandwidth for MMD Kernel
    finite_diff_epsilon: float = 1.0e-8    # epsilon used in finite difference sampling for KL (does not work yet)
    adam_learning_rate: float = 0.01       # initial learning rate for ADAM
    eval_every: int = 1                    # compute full train/val/test MMD every k iterations (+ final)
    model_selection_metric: str = "mmd_val"  # metric used to pick the best checkpoint (mmd_val recommended)


@dataclass
class SweepConfig:
    # initial_random_seed is the ONLY seed you set. runs_batch_size auto-seeded runs are spawned per
    # hydra job, each derived as initial_random_seed + i (i = 0 .. runs_batch_size-1) and logged as
    # the run's effective seed.
    runs_batch_size: int = 1               # number of auto-seeded training runs per hydra job (seeds)
    initial_random_seed: int = 42          # base seed; run i in a batch uses initial_random_seed + i
    gpus_per_node: int = 0                 # >0 round-robins runs across GPUs via CUDA_VISIBLE_DEVICES; 0 = CPU mode

    # CPU parallelism budget for the whole sweep. All (grid combo x seed) runs share one global
    # process pool sized by these, so the entire --multirun runs concurrently up to the hardware
    # limit -- not one grid combo at a time. Both default to 0 = auto: max_parallel_runs is then
    # min(total_runs, n_cpus) and threads_per_run is n_cpus // max_parallel_runs, so few runs each
    # get many threads (fast sampling) and many runs trade threads for run-level parallelism.
    # See src/setup.py::plan_resources.
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
    """Wrap a wandb run's logged config dict (from src.setup._init_wandb, itself
    OmegaConf.to_container(cfg, resolve=True)) back into this schema, so callers get the same dot
    access (cfg.data.dataset) and validation as a live Hydra run, instead of dict indexing.

    wandb's public API already strips its own internal keys (_wandb, wandb_version) from
    Run.config, so this only ever sees the fields we logged ourselves.
    """
    return OmegaConf.merge(OmegaConf.structured(Config), config)
