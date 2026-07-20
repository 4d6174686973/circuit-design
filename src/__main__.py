from concurrent.futures import ProcessPoolExecutor

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.setup import train_worker, get_or_create_wandb_sweep


def pretrain_mps(cfg: DictConfig, seeds: list) -> None:
    """Train/cache the shared MPS(s) once, before the parallel fan-out, to avoid training races.

    For JGB and BAS full_support the train set is seed-independent (one MPS); for BAS holdout each
    seed has its own train set, so we dedup by cache key and pretrain each distinct one.
    """
    from src.setup import setup_dataloader, compute_split, get_or_train_mps, mps_cache_dir

    trained = set()
    for seed in seeds:
        c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        c.random_seed = seed
        dataloader = setup_dataloader(c)
        X_train, *_ = compute_split(c, dataloader)
        key = mps_cache_dir(c, X_train)
        if key in trained:
            continue
        get_or_train_mps(c, X_train)
        trained.add(key)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    seeds = [cfg.initial_random_seed + i for i in range(cfg.runs_batch_size)]
    hydra_cfg = HydraConfig.get()
    group = hydra_cfg.job.override_dirname or "single"
    output_dir = hydra_cfg.runtime.output_dir

    # Create (or reuse) the shared wandb Sweep and set WANDB_SWEEP_ID before spawning parallel
    # seed-workers, so every one of them (which inherit this process's environment) joins the same
    # real wandb Sweep automatically on wandb.init() -- idempotent, so on a --multirun this only
    # actually creates a sweep on the first job (BasicLauncher runs all jobs in this same process);
    # later jobs, and workers spawned in this or later jobs, just reuse the id already in the env.
    get_or_create_wandb_sweep(cfg)

    # Train the shared MPS once before fanning out so parallel workers never race on it.
    pretrain_mps(cfg, seeds)

    cfg_container = OmegaConf.to_container(cfg, resolve=True)

    if cfg.runs_batch_size == 1:
        train_worker(cfg_container, seeds[0], 0, group, output_dir)
        return

    # Fan out the auto-seeded batch across processes (one wandb run per seed).
    with ProcessPoolExecutor(max_workers=cfg.runs_batch_size) as executor:
        futures = [executor.submit(train_worker, cfg_container, seed, i, group, output_dir)
                   for i, seed in enumerate(seeds)]
        for f in futures:
            f.result()  # re-raise any worker exception


if __name__ == "__main__":
    main()
