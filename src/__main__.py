"""Entrypoint: expand the Hydra (multi)run grid and train every (grid combo x seed) run in parallel.

Parallelism model
-----------------
`--multirun a=1,2 b=x,y` makes Hydra call main() once per grid combo, sequentially, in ONE process
(the default BasicLauncher). Each call fans out `sweep.runs_batch_size` auto-seeded runs. To run the
WHOLE sweep concurrently (not one combo at a time), every combo submits its seed-runs to a single
GLOBAL process pool that persists across all the sequential main() calls; submission is
non-blocking, so combo N's runs start executing while Hydra moves on to compose combo N+1. The pool
is sized once (plan_resources) to the hardware, so total concurrency = max_parallel_runs regardless
of how many combos there are. All runs are joined at interpreter exit (_drain_pool via atexit) --
there is no per-combo hook for "the whole multirun is done", so atexit is the barrier.

Why one shared process (not Hydra's joblib launcher): the wandb Sweep is created once and the shared
MPS is pretrained once, both keyed off process-local state (env var / disk cache). Running combos in
separate processes (joblib) would race to create duplicate sweeps and retrain the MPS. Keeping combo
dispatch in one process avoids both; only the (independent) training runs are parallelized.

GPU NOTE: the pool here is CPU-oriented (many workers, few threads each). A GPU mode would size the
pool to GPU count instead and pin each worker to a device -- see plan_resources / train_worker.
"""

import atexit
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

import src.config_schema  # noqa: F401 -- registers Config with Hydra's ConfigStore
from src.setup import (train_worker, get_or_create_wandb_sweep, plan_resources, apply_thread_env)

# Global cross-combo process pool + submitted futures (see module docstring).
_POOL = None
_FUTURES = []          # list of (future, label) for exit-time joining/error reporting
_NEXT_WORKER_INDEX = 0  # global run counter (used for GPU round-robin; harmless on CPU)


def _get_pool(max_workers: int) -> ProcessPoolExecutor:
    """Lazily create the one shared pool. Spawn context so workers inherit the thread-cap env vars
    (set by apply_thread_env before this is called) at import time -- the only point BLAS/OpenMP
    pool sizes can be fixed reliably."""
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"))
        atexit.register(_drain_pool)
    return _POOL


def _drain_pool() -> None:
    """Join all submitted runs at interpreter exit; report failures and exit non-zero if any."""
    if _POOL is None:
        return
    logger = logging.getLogger("QCBM")
    failures = []
    for future, label in _FUTURES:
        try:
            future.result()
        except Exception as exc:  # one run failing must not lose the others
            failures.append(label)
            logger.error(f"run failed [{label}]: {exc!r}")
    _POOL.shutdown(wait=True)
    if failures:
        logger.error(f"{len(failures)}/{len(_FUTURES)} runs failed: {', '.join(failures)}")
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)  # force non-zero exit from atexit (child runs already joined above)


def pretrain_mps(cfg: DictConfig, seeds: list) -> None:
    """Train/cache the shared MPS(s) once, before the parallel fan-out, to avoid training races.

    For JGB and BAS full_support the train set is seed-independent (one MPS); for BAS holdout each
    seed has its own train set, so we dedup by cache key and pretrain each distinct one.
    """
    from src.setup import setup_dataloader, compute_split, get_or_train_mps, mps_cache_dir

    trained = set()
    for seed in seeds:
        c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        c.sweep.random_seed = seed
        dataloader = setup_dataloader(c)
        X_train, *_ = compute_split(c, dataloader)
        key = mps_cache_dir(c, X_train)
        if key in trained:
            continue
        get_or_train_mps(c, X_train)
        trained.add(key)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    global _NEXT_WORKER_INDEX

    seeds = [cfg.sweep.initial_random_seed + i for i in range(cfg.sweep.runs_batch_size)]
    hydra_cfg = HydraConfig.get()
    # Local-only label for the .err log line and pool failure reporting (NOT a wandb group -- the
    # swept params are stored in each run's wandb config and are filtered on from there instead).
    combo = hydra_cfg.job.override_dirname or "single"
    output_dir = hydra_cfg.runtime.output_dir
    is_multirun = hydra_cfg.mode == RunMode.MULTIRUN

    # Create (or reuse) the shared wandb Sweep and set WANDB_SWEEP_ID once for the whole process, so
    # every worker (which inherits this environment) joins the same real Sweep. Idempotent across
    # the sequential multirun combos.
    get_or_create_wandb_sweep(cfg)

    # Resolve the CPU budget for the whole sweep and export the per-run thread cap BEFORE the pool is
    # spawned, so freshly-spawned workers inherit it at import time. Written back into cfg so each
    # worker (and Aer) reads a concrete threads_per_run.
    max_parallel_runs, threads_per_run = plan_resources(cfg)
    cfg.sweep.threads_per_run = threads_per_run
    apply_thread_env(threads_per_run)

    # Train the shared MPS once before fanning out so parallel workers never race on it.
    pretrain_mps(cfg, seeds)

    cfg_container = OmegaConf.to_container(cfg, resolve=True)

    # Fast path: a single, non-multirun run -> execute inline (no pool: lower overhead, exceptions
    # surface directly, simpler to debug/attach).
    if not is_multirun and cfg.sweep.runs_batch_size == 1:
        train_worker(cfg_container, seeds[0], 0, combo, output_dir)
        return

    # Submit this combo's seed-runs to the shared pool WITHOUT blocking, so Hydra proceeds to the
    # next combo and its runs overlap with this one's. Everything is joined at exit (_drain_pool).
    pool = _get_pool(max_parallel_runs)
    for seed in seeds:
        idx = _NEXT_WORKER_INDEX
        _NEXT_WORKER_INDEX += 1
        label = f"{combo}/seed{seed}"
        _FUTURES.append(
            (pool.submit(train_worker, cfg_container, seed, idx, combo, output_dir), label))


if __name__ == "__main__":
    main()
