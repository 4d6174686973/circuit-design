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
    - `cost.py` loss functions, gradients, and meta-optimizer
    - `data.py` dataclasses and dataloader
    - `decomposition.py` decompose MPS to PQC by Yuki Sato [@yksat](https://github.com/yksat)
    - `extension.py` functions for extending the decomposed circuit
    - `mps.py` train MPS by [@congzlwag](https://github.com/congzlwag)
    - `qcbm.py` train QCBM using qiskit
    - `setup.py` setup the whole simulation from config including pretraining
    - `utils.py` general utility functions and plotting

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

## Testing and code coverage
```sh
uv run pytest src/tests
uv run pytest --cov=src
```

## Updating dependencies

Add or remove packages with `uv add <package>` / `uv remove <package>` (use `--dev` for dev-only tools). Upgrade everything within the constraints in `pyproject.toml` with `uv lock --upgrade && uv sync`.

## Reference

[1] Mathis Makarski, Jumpei Kato, Yuki Sato, Naoki Yamamoto, [Circuit Design based on Feature Similarity for Quantum Generative Modeling](https://doi.org/10.48550/arXiv.2503.11983), arXiv preprint (2025)