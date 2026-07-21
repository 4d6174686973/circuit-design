"""Generative-model benchmarking for the QCBM.

Pure, read-only metric functions plus classical baselines and a wandb-driven best-model selection
step. Metrics consume the {bitstring: count} dictionary format used throughout the codebase and
reuse helpers from src.utils / src.cost so the held-out MMD matches the training kernel exactly.
"""

import os

import numpy as np
import pandas as pd
import scipy.stats as ss

from src.utils import sample_info, array_to_str, get_features_for_quasi_dist, get_nested
from src.cost import cost_mmd


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def to_prob_dict(counts: dict) -> dict:
    """Normalize a {bitstring: count} dict into {bitstring: probability}."""
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def weighted_quantile(values: np.ndarray, probs: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Quantiles of a discrete weighted distribution (values with probabilities probs)."""
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(probs)[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return np.interp(q, cdf, v)


# --------------------------------------------------------------------------------------------------
# distribution-distance metrics (both datasets)
# --------------------------------------------------------------------------------------------------
def mmd(samples: dict, target: dict, sigmas=np.array([1.0])) -> float:
    """Maximum Mean Discrepancy between model samples and target (same kernel as training)."""
    return float(cost_mmd(target, samples, np.asarray(sigmas)))


def kl_divergence(samples: dict, target: dict, eps: float = 1e-8) -> float:
    """KL(target || model) on normalized histograms. Reported metric (KL is unstable as a loss)."""
    P = to_prob_dict(target)
    Q = to_prob_dict(samples)
    return float(sum(p * np.log(p / max(eps, Q.get(k, 0.0))) for k, p in P.items()))


def total_variation_distance(samples: dict, target: dict) -> float:
    """Total variation distance in [0, 1] between model and target histograms."""
    P = to_prob_dict(target)
    Q = to_prob_dict(samples)
    keys = set(P) | set(Q)
    return float(0.5 * sum(abs(P.get(k, 0.0) - Q.get(k, 0.0)) for k in keys))


def classical_fidelity(samples: dict, target: dict) -> float:
    """Bhattacharyya / statistical fidelity in [0, 1]."""
    P = to_prob_dict(target)
    Q = to_prob_dict(samples)
    keys = set(P) | set(Q)
    return float(sum(np.sqrt(P.get(k, 0.0) * Q.get(k, 0.0)) for k in keys) ** 2)


def negative_log_likelihood(samples: dict, target_samples: np.ndarray, eps: float = 1e-8) -> float:
    """Mean NLL of held-out data points under the model distribution."""
    Q = to_prob_dict(samples)
    bitstrings = array_to_str(target_samples)
    return float(np.mean([-np.log(max(eps, Q.get(b, 0.0))) for b in bitstrings]))


# --------------------------------------------------------------------------------------------------
# BAS-specific metrics (discrete finite support)
# --------------------------------------------------------------------------------------------------
def mode_coverage(samples: dict, valid_patterns: np.ndarray) -> float:
    """Fraction of valid BAS patterns that receive nonzero model mass (mode-collapse detector)."""
    valid = set(array_to_str(valid_patterns))
    seen = set(samples.keys()) & valid
    return float(len(seen) / len(valid))


def spurious_mass(samples: dict, valid_patterns: np.ndarray) -> float:
    """Total model probability placed on bitstrings that are not valid bars/stripes patterns."""
    valid = set(array_to_str(valid_patterns))
    Q = to_prob_dict(samples)
    return float(sum(p for k, p in Q.items() if k not in valid))


# --------------------------------------------------------------------------------------------------
# JGB-specific metrics (binarized continuous, per feature)
# --------------------------------------------------------------------------------------------------
def reconstruct_features(samples: dict, bits_per_feature: int, num_features: int,
                         x_min, x_max) -> list:
    """Reconstruct per-feature real-valued marginals from a model-sample dict.

    Returns a list of (values, probs) per feature, using the train-fitted x_min/x_max bounds so the
    inversion matches the (leakage-safe) encoding used for training/eval.
    """
    feat_dicts = get_features_for_quasi_dist(samples, bits_per_feature, num_features)
    denom = 2 ** bits_per_feature - 1
    out = []
    for n, fdict in enumerate(feat_dicts):
        vals, probs = [], []
        for bitstring, p in fdict.items():
            integer = int(bitstring, 2)
            real = x_min[n] + integer * (x_max[n] - x_min[n]) / denom
            vals.append(real)
            probs.append(p)
        out.append((np.array(vals), np.array(probs)))
    return out


def qq_model_vs_data(model_vals, model_probs, data_vals, n_q: int = 100):
    """Quantile pairs (data, model) for a model-vs-data QQ plot."""
    q = np.linspace(0.01, 0.99, n_q)
    return np.quantile(data_vals, q), weighted_quantile(model_vals, model_probs, q)


def qq_model_vs_normal(model_vals, model_probs, n_q: int = 100):
    """Quantile pairs (normal, model): is the model marginal Gaussian-like?"""
    q = np.linspace(0.01, 0.99, n_q)
    mean = np.average(model_vals, weights=model_probs)
    std = np.sqrt(np.average((model_vals - mean) ** 2, weights=model_probs))
    return ss.norm.ppf(q, loc=mean, scale=std), weighted_quantile(model_vals, model_probs, q)


def qq_data_vs_normal(data_vals, n_q: int = 100):
    """Quantile pairs (normal, data): baseline — is the data itself Gaussian?"""
    q = np.linspace(0.01, 0.99, n_q)
    mean, std = np.mean(data_vals), np.std(data_vals)
    return ss.norm.ppf(q, loc=mean, scale=std), np.quantile(data_vals, q)


def wasserstein_per_feature(model_vals, model_probs, data_vals) -> float:
    """1-Wasserstein distance between a model marginal and the empirical data marginal."""
    return float(ss.wasserstein_distance(model_vals, data_vals,
                                         u_weights=model_probs, v_weights=None))


# --------------------------------------------------------------------------------------------------
# classical baselines
# --------------------------------------------------------------------------------------------------
def independent_bits_baseline(target: dict, n_bits: int, n_samples: int, seed: int = 0) -> dict:
    """Product-of-marginals model: each bit sampled independently from its target frequency.

    A QCBM must beat this on MMD/KL/TV to claim it learned correlations between bits.
    """
    T, T_probs = sample_info(target)
    marg = np.average(T, axis=0, weights=T_probs)  # P(bit_j = 1)
    rng = np.random.default_rng(seed)
    samples = (rng.random((n_samples, n_bits)) < marg).astype(int)
    from collections import Counter
    return dict(Counter(array_to_str(samples)))


def gaussian_baseline_jgb(decimal_train: np.ndarray, bits_per_feature: int, n_features: int,
                          n_samples: int, x_min, x_max, seed: int = 0) -> dict:
    """Independent per-tenor Gaussian fitted on train decimals, re-binarized with train bounds."""
    from src.utils import real_to_binary
    from collections import Counter
    rng = np.random.default_rng(seed)
    mean = decimal_train.mean(axis=0)
    std = decimal_train.std(axis=0)
    samples_real = rng.normal(mean, std, size=(n_samples, n_features))
    binary, _ = real_to_binary(samples_real, bits_per_feature, x_min, x_max, clip=True)
    return dict(Counter(array_to_str(binary)))


# --------------------------------------------------------------------------------------------------
# top-level evaluation
# --------------------------------------------------------------------------------------------------
def evaluate(samples: dict, splits: dict, dataset_kind: str, sigmas=np.array([1.0]),
             valid_patterns: np.ndarray = None) -> dict:
    """Run the appropriate metric bundle over each named split and return a flat metric dict.

    Args:
        samples: model sample counts {bitstring: count}
        splits: {split_name: target_count_dict}, e.g. {"val": ..., "test": ...}
        dataset_kind: "BAS" or "JGB"
        valid_patterns: enumerated valid BAS patterns (required for BAS coverage/spurious mass)
    """
    metrics = {}
    for name, target in splits.items():
        if not target:
            continue
        metrics[f"{name}/mmd"] = mmd(samples, target, sigmas)
        metrics[f"{name}/kl"] = kl_divergence(samples, target)
        metrics[f"{name}/tv"] = total_variation_distance(samples, target)
        metrics[f"{name}/fidelity"] = classical_fidelity(samples, target)
    if dataset_kind == "BAS" and valid_patterns is not None:
        metrics["coverage"] = mode_coverage(samples, valid_patterns)
        metrics["spurious_mass"] = spurious_mass(samples, valid_patterns)
    return metrics


# --------------------------------------------------------------------------------------------------
# best-model selection across a wandb group
# --------------------------------------------------------------------------------------------------
def select_best_run(runs: list, metric: str = "best_mmd_val"):
    """Return the run with the minimum summary metric (lowest validation MMD by default)."""
    scored = [(r, r.summary.get(metric, np.inf)) for r in runs]
    scored = [(r, s) for r, s in scored if s is not None and np.isfinite(s)]
    if not scored:
        return None
    return min(scored, key=lambda rs: rs[1])[0]


def load_checkpoint(run, root: str = "./artifacts"):
    """Download a run's model artifact and load (circuit, best_params).

    Skips the download entirely -- including the network round-trip to fetch/verify the artifact
    manifest -- if both expected files are already present locally under root/run.id. Safe here
    because each run logs exactly one model artifact, written once at the end of training (see
    src/setup.py::setup_and_train_qcbm), so there is no newer version a stale local copy could miss.
    """
    from qiskit import qpy
    local_dir = f"{root}/{run.id}"
    circuit_path = f"{local_dir}/circuit.qpy"
    params_path = f"{local_dir}/best_params.npy"

    if os.path.exists(circuit_path) and os.path.exists(params_path):
        print(f"[benchmark]     [{run.id}] checkpoint already downloaded, reusing {local_dir}")
    else:
        art = None
        for a in run.logged_artifacts():
            if a.type == "model":
                art = a
                break
        if art is None:
            raise FileNotFoundError(f"No model artifact for run {run.id}")
        art.download(root=local_dir)

    with open(circuit_path, "rb") as f:
        circuit = qpy.load(f)[0]
    best_params = np.load(params_path)
    return circuit, best_params


def sample_model(circuit, params: np.ndarray, n_shots: int, seed: int = 0) -> dict:
    """Sample a bound circuit via Aer's parameter_binds fast path -> {bitstring: count}."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector", runtime_parameter_bind_enable=True,
                       max_parallel_experiments=0, seed_simulator=seed)
    binds = {p: np.array([params[p.index]]) for p in circuit.parameters}
    counts = sim.run(circuit, parameter_binds=[binds], shots=n_shots).result().get_counts()
    return counts if isinstance(counts, dict) else counts[0]


def _test_split_for_config(config: dict):
    """Reconstruct the held-out test target + dataset context from a run's logged config."""
    from src.data import BAS, JGB, DataLoader
    data_cfg = config["data"]
    if data_cfg["dataset"] == "BAS":
        dataset = BAS(data_cfg["width"], data_cfg["height"])
    else:
        dataset = JGB(data_cfg["N_qubits"], data_cfg["N_features"])
    dl = DataLoader(dataset)
    _, X_val, X_test, _, c_val, c_test = dl.train_val_test_split(
        data_cfg["train_split"], data_cfg["val_split"],
        seed=get_nested(config, "sweep.random_seed"),
        bas_split_mode=data_cfg.get("bas_split_mode", "full_support"))
    valid_patterns = dataset.binary if data_cfg["dataset"] == "BAS" else None
    return {"val": c_val, "test": c_test}, valid_patterns


def benchmark_sweep(sweep_id: str, entity: str, project: str, group_by: str = "circuit.extension",
                    n_shots: int = 10000, metric: str = "best_mmd_val") -> pd.DataFrame:
    """Benchmark the best model per group of a sweep.

    For each group (value of `group_by`, a dot-separated path into the run's nested config, e.g.
    "circuit.extension"), select the seed-run with the lowest validation MMD, load its best
    checkpoint, sample it, and evaluate the full metric suite on the held-out test split. Returns a
    tidy table with one row per group.
    """
    import wandb
    api = wandb.Api()
    entity = entity or api.default_entity  # unresolved None would literally build ".../None/..."
    runs = api.sweep(f"{entity}/{project}/{sweep_id}").runs

    groups = {}
    for r in runs:
        key = get_nested(r.config, group_by)
        groups.setdefault(key, []).append(r)

    rows = []
    for key, group_runs in groups.items():
        print(f"[benchmark]     [{key}] {len(group_runs)} run(s) -> selecting best by {metric}...")
        best = select_best_run(group_runs, metric)
        if best is None:
            print(f"[benchmark]     [{key}] no run with a finite {metric}, skipping.")
            continue
        print(f"[benchmark]     [{key}] best run {best.id} ({metric}={best.summary.get(metric)}); "
              f"downloading checkpoint...")
        circuit, params = load_checkpoint(best)
        splits, valid_patterns = _test_split_for_config(best.config)
        print(f"[benchmark]     [{key}] sampling {n_shots} shots and evaluating metrics...")
        samples = sample_model(circuit, params, n_shots, seed=get_nested(best.config, "sweep.random_seed", 0))
        row = {group_by: key, "run_id": best.id, "run_name": best.name,
               "best_mmd_val": best.summary.get(metric)}
        row.update(evaluate(samples, splits, best.config["data"]["dataset"],
                            sigmas=np.array(get_nested(best.config, "qcbm.sigmas", [1.0])),
                            valid_patterns=valid_patterns))
        rows.append(row)
    return pd.DataFrame(rows)
