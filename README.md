# Circuit Design based on Dataset Topology for Quantum Generative Modeling

## First setup

Install dependencies or open in devcontainer

```sh
pip install -r requirements.txt
```

Download JGB dataset from: [https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/index.htm](https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/index.htm) and insert in this project `data/jgbcme_all.csv`

## Project Structure

- `.devcontainer/` for running the devcontainer
- `data/` need to create this folder by yourself when adding data files
- `results/`
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
python -m src
```
Results will be saved in `outputs/`

## Multirun
Running multiple simulations sequentially based on same config file but changing config parameters e.g. the extension method
```sh
python -m src --multirun extension=none,metric_based,all_to_all
```
Important for running simulations with random extension is changing the random_seed, which is somehow fixed when using multirun from hydra
```sh
python -m src --multirun random_seed=42,43,44,45,46
```
Results will be saved in `multirun/`

## Testing and code coverage
```sh
pytest src/tests
pytest --cov=src
```

## Reference

Mathis Makarski, Jumpei Kato, Yuki Sato, Naoki Yamamoto, [Circuit Design based on Feature Similarity for Quantum Generative Modeling](https://doi.org/10.48550/arXiv.2503.11983), arXiv preprint (2025)