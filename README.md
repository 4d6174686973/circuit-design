# Circuit Design based on Feature Similarity for Quantum Generative Modeling
This repository includes the source code for simulations carried out in Ref. [[1](#reference)].

## First setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv once (e.g. `brew install uv`), then from the repo root:

```sh
uv sync
```

This creates `.venv/` with the pinned Python version (see `.python-version`) and installs all dependencies from the lockfile `uv.lock`, including the dev tools (pytest, ipykernel). No manual venv activation is needed — prefix commands with `uv run` instead.

Download JGB dataset from: [https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/index.htm](https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/index.htm) and insert in this project `data/jgbcme_all.csv`

## Project Structure

- `data/` need to create this folder by yourself when adding data files
- `results v1/`
    - `runs/...` results of the numerical simulations 
    - `plot_figures.ipynb` plot the figures for the paper 
- `src/` code for running simulations
    - `src/conf/config.yaml` configuration for the simulation
    - `benchmark.py` held-out generative-model metrics, classical baselines, best-model selection
    - `cost.py` loss functions, gradients, and meta-optimizer
    - `data.py` dataclasses and dataloader
    - `decomposition.py` decompose MPS to PQC by Yuki Sato [@yksat](https://github.com/yksat)
    - `extension.py` functions for extending the decomposed circuit
    - `mps.py` train MPS by [@congzlwag](https://github.com/congzlwag)
    - `plotting.py` wandb-driven figure generation for a sweep (replaces `results v1/plot_figures.ipynb`)
    - `qcbm.py` train QCBM using qiskit
    - `setup.py` setup the whole simulation from config including pretraining, wandb, and sweep creation
    - `utils.py` general utility functions and plotting
- `scripts/sweep.sh` SLURM-submittable launcher for cluster/multi-node sweeps (see below)

## Singlerun
Running a single simulation based on current config file `src/conf/config.yaml`
```sh
uv run python -m src
```
Results will be saved in `outputs/`

## Multirun
Running multiple simulations sequentially based on same config file but changing config parameters e.g. the extension method
```sh
uv run python -m src --multirun extension=none,metric_based,all_to_all
```
Results will be saved in `multirun/`

## Parallel seeds (batching)
For each hyperparameter combination (e.g. each `extension` above), you can run several repeats with
different random seeds in parallel. You only ever set `initial_random_seed`; the `runs_batch_size`
parallel repeats are auto-seeded as `initial_random_seed + i` and logged individually — there is no
separate seed parameter to set per run, so runs can't accidentally collide on the same seed.
```sh
uv run python -m src --multirun extension=none,metric_based,all_to_all \
    runs_batch_size=5 initial_random_seed=42
```
This runs 3 extensions x 5 seeds (42-46) = 15 trainings, 5 running in parallel at a time.

## Experiment tracking (wandb)

Every invocation above — a single run, a local `--multirun`, or a `scripts/sweep.sh` cluster launch
— is automatically registered as a real **wandb Sweep** (visible under the Sweeps tab, with
parallel-coordinates plots etc.), no extra flags needed. The Hydra grid (and `runs_batch_size`'s
auto-seeded repeats) is what actually determines what gets run — wandb is not used to choose
hyperparameters (no `wandb agent`), only to track and organize the resulting runs. Each run's
`group` is its swept-parameter combination (e.g. `extension=metric_based`); seeds within a group are
its repeats.

By default `wandb_mode: online` in `src/conf/config.yaml`, so **run `wandb login` once** before your
first sweep (or pass `wandb_mode=offline` to log locally and `wandb sync` later, or
`wandb_mode=disabled` to skip wandb entirely, e.g. for quick local testing):
```sh
uv run python -m src --multirun extension=none,metric_based,all_to_all wandb_mode=offline
```
Project/entity are set via `wandb_project` / `wandb_entity` in the config (or as CLI overrides).

For a cluster/SLURM launch (multi-node, proper resource requests, automatic disjoint seed-sharding
across nodes, and leader/worker synchronization so every node's runs join the *same* sweep), use
`scripts/sweep.sh` instead of invoking `python -m src` directly — see the comments at the top of
that file for `sbatch`/multi-node usage.

Once a sweep has runs, regenerate all figures (including the paper's dataset/topology plots, MMD
over cumulative measurements, and best-model QQ/benchmark plots) as PDF:
```sh
# BAS
uv run python -m src.plotting --sweep-id "<sweep_id>" --project qcbm-circuit-design \
    --dataset BAS --width 3 --height 3

# JGB
uv run python -m src.plotting --sweep-id "<sweep_id>" --project qcbm-circuit-design \
    --dataset JGB --n-qubits 12 --n-features 3
```
The `sweep_id` is printed in every run's log line (`Program started (..., sweep_id=...)`), or find it
under the wandb project's Sweeps tab. Run `uv run python -m src.plotting --help` for all options
(`--entity`, `--group-by` to legend by a different swept parameter, `--metrics`, `--plots-dir`, ...).
Figures are saved under `<plots-dir>/<sweep_id>-<dataset>/` (default `plots/<sweep_id>-<dataset>/`),
so different sweeps/datasets never collide or mix in one flat folder.

## Testing and code coverage
```sh
uv run pytest src/tests
uv run pytest --cov=src
```

## Updating dependencies

Add or remove packages with `uv add <package>` / `uv remove <package>` (use `--dev` for dev-only tools). Upgrade everything within the constraints in `pyproject.toml` with `uv lock --upgrade && uv sync`.

## Reference

[1] Mathis Makarski, Jumpei Kato, Yuki Sato, Naoki Yamamoto, [Circuit Design based on Feature Similarity for Quantum Generative Modeling](https://doi.org/10.48550/arXiv.2503.11983), arXiv preprint (2025)