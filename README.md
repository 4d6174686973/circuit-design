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

Dataset-specific settings (`N_qubits`, `width`/`height`, `N_features`, `extension_metric`, ...) live in
`src/conf/dataset/BAS.yaml` and `src/conf/dataset/JGB.yaml`, both merged into the same base
`src/conf/config.yaml` layout. Switch between them with the single `dataset` override, on the CLI or
via `scripts/sweep.sh`'s `DATASET` env var:
```sh
uv run python -m src dataset=BAS
uv run python -m src dataset=JGB
```
Every other field is grouped hierarchically to match the config's sections -- `data`, `ibm`, `aer`,
`mps`, `circuit`, `qcbm`, `sweep`, `logging` -- e.g. `cfg.mps.cutoff`, `cfg.qcbm.iterations`,
`cfg.sweep.runs_batch_size`, so an override looks like `mps.cutoff=1e-4` or `qcbm.iterations=50`.
`src/config_schema.py` defines a typed schema (`Config`, with one nested dataclass per group) for
every field above; composing an unknown key or a wrongly-typed override fails fast at startup
instead of surfacing as a runtime `AttributeError`.

## Multirun
Running multiple simulations based on the same config file but changing config parameters, e.g. the
extension method (and any other field — thresholds, cutoffs, ...):
```sh
uv run python -m src --multirun circuit.extension=none,metric_based,all_to_all
```
Results will be saved in `multirun/`.

## Parallel seeds (batching)
For each hyperparameter combination (e.g. each `circuit.extension` above), you can run several repeats
with different random seeds. You only ever set `sweep.initial_random_seed`; the
`sweep.runs_batch_size` repeats are auto-seeded as `initial_random_seed + i` and logged
individually — there is no separate seed parameter to set per run, so runs can't accidentally
collide on the same seed.
```sh
uv run python -m src --multirun circuit.extension=none,metric_based,all_to_all \
    sweep.runs_batch_size=5 sweep.initial_random_seed=42
```

## Parallelism (CPU)
The **entire** sweep — every `(grid combo × seed)` run — is executed concurrently through a single
process pool sized to the machine, not one grid combo at a time. For the example above (3 extensions
× 5 seeds = 15 runs) all 15 are scheduled at once, capped only by the hardware.

The CPU budget is auto-derived and needs no tuning: `sweep.max_parallel_runs` concurrent runs, each
with `sweep.threads_per_run` BLAS/OpenMP/Aer/gradient threads, chosen so their product ≈ the node's
core count. Both default to `0` (auto): a small sweep gives each run many threads (faster Aer
sampling), a large sweep trades threads for more concurrent runs — always using ~all cores. Override
either to steer the trade-off (the other is derived to fill the cores):
```sh
# force 4 threads per run (concurrency = cores // 4); or cap concurrent runs directly:
uv run python -m src --multirun circuit.extension=none,metric_based,all_to_all \
    sweep.runs_batch_size=5 sweep.threads_per_run=4
uv run python -m src --multirun circuit.extension=none,metric_based,all_to_all \
    sweep.runs_batch_size=5 sweep.max_parallel_runs=16
```
This design is CPU-oriented (each run is many small numpy ops plus one Aer sampling step, so
run-level parallelism scales better than threads-per-run). GPU scheduling is deliberately left as
**future work** — see the *GPU (future work)* note below.

## Experiment tracking (wandb)

Every invocation above — a single run, a local `--multirun`, or a `scripts/sweep.sh` cluster launch
— is automatically registered as a real **wandb Sweep** (visible under the Sweeps tab, with
parallel-coordinates plots etc.), no extra flags needed. The Hydra grid (and `runs_batch_size`'s
auto-seeded repeats) is what actually determines what gets run — wandb is not used to choose
hyperparameters (no `wandb agent`), only to track and organize the resulting runs. Each run's
`group` is its swept-parameter combination (e.g. `circuit.extension=metric_based`); seeds within a
group are its repeats.

By default `logging.wandb_mode: online` in `src/conf/config.yaml`, so **run `wandb login` once**
before your first sweep (or pass `logging.wandb_mode=offline` to log locally and `wandb sync` later,
or `logging.wandb_mode=disabled` to skip wandb entirely, e.g. for quick local testing):
```sh
uv run python -m src --multirun circuit.extension=none,metric_based,all_to_all logging.wandb_mode=offline
```
Project/entity are set via `logging.wandb_project` / `logging.wandb_entity` in the config (or as CLI overrides).

To keep per-run overhead low at high parallelism, each wandb run disables its background
system-metrics monitor and metadata/git/code scanning (only the training metrics we explicitly log
are kept) — otherwise every one of the dozens of concurrent runs would spawn its own polling thread
and periodic uploads.

For a cluster/SLURM launch (multi-node, proper resource requests, automatic disjoint seed-sharding
across nodes, and leader/worker synchronization so every node's runs join the *same* sweep), use
`scripts/sweep.sh` instead of invoking `python -m src` directly — see the comments at the top of
that file for `sbatch`/multi-node usage.

## GPU (future work)
This project currently runs on **CPU only**. Each training run is dominated by many small numpy/scipy
operations (kernel, gradient, Adam) with a single heavier Aer statevector *sampling* step, so on the
9–12 qubit problems here throughput comes from running many CPU runs in parallel (above), and a GPU
helps only at higher qubit counts. A naive "sample on GPU, everything else on CPU" hybrid is
dominated by per-iteration CPU↔GPU transfers, so making GPUs pay off is a larger change, left as
future work. The swap points are marked with `GPU NOTE` comments in the code:
- `src/setup.py::plan_resources` — size the pool to GPU count/VRAM instead of CPU cores.
- `src/setup.py::train_worker` — GPU pinning (`CUDA_VISIBLE_DEVICES`) is stubbed via
  `sweep.gpus_per_node`; skip CPU thread-capping for GPU runs.
- `src/setup.py::setup_qiskit_simulator` — the `device="GPU"` path (`batched_shots_gpu`, multi-GPU
  `blocking`) already exists; verify VRAM/blocking sizing for the target GPUs.
- `src/cost.py`, `src/qcbm.py` — move the per-iteration kernel/gradient math onto the GPU (e.g.
  `cupy`) so a run stays device-resident across the whole iteration, not just during sampling.
- `scripts/sweep.sh` — add `#SBATCH --gres=gpu:N`, set `GPUS_PER_NODE=N`, `SIMULATOR=aer_statevec_gpu`.

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