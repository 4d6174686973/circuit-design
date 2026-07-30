"""Generative-model benchmarking for the QCBM.

Pure, read-only metric functions plus classical baselines and a wandb-driven best-model selection
step. Metrics consume the {bitstring: count} dictionary format used throughout the codebase and
reuse helpers from src.utils / src.cost so the held-out MMD matches the training kernel exactly.
"""

import os
import shutil

import numpy as np
import pandas as pd
import scipy.stats as ss
from omegaconf import OmegaConf

from src.utils import sample_info, array_to_str, get_features_for_quasi_dist
from src.cost import cost_mmd
from src.config_schema import from_run_config


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
def qbas_metrics(samples: dict, valid_patterns: np.ndarray) -> dict:
    """Precision/recall/F1 against the FULL valid BAS pattern space (not just the unseen subset).

    Standard discrete BAS benchmark protocol (Benedetti, Garcia-Pintos, Perdomo-Ortiz et al.,
    "A generative modeling approach for benchmarking and training shallow quantum circuits", npj
    Quantum Inf. 5, 45, 2019): precision is the fraction of generated samples (with multiplicity)
    that land on a valid bars-and-stripes pattern, recall is the fraction of the enumerated valid
    patterns hit at least once, and the qBAS score is their harmonic mean (F1). Unlike
    generalization_metrics (bench_val/*), this scores against the WHOLE valid space and never
    references the training set.
    """
    valid = set(array_to_str(valid_patterns))
    nan = float("nan")
    Q = sum(samples.values())
    if Q == 0 or not valid:
        return {"bench_BAS/precision": nan, "bench_BAS/recall": nan, "bench_BAS/qbas": nan}

    q_valid = 0
    hit = set()
    for bitstring, c in samples.items():
        if bitstring in valid:
            q_valid += c
            hit.add(bitstring)

    precision = q_valid / Q
    recall = len(hit) / len(valid)
    qbas = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else nan
    return {"bench_BAS/precision": float(precision), "bench_BAS/recall": float(recall),
            "bench_BAS/qbas": float(qbas)}


# --------------------------------------------------------------------------------------------------
# generalization metrics (validity-based) -- Gili et al.
#
# Established in:
#   * D. Gili, M. Mauri, A. Perdomo-Ortiz, "Do Quantum Circuit Born Machines Generalize?",
#     Quantum Sci. Technol. 8, 035021 (2023). arXiv:2207.13645.
#   * K. Gili, M. Hibat-Allah, M. Mauri, C. Ballance, A. Perdomo-Ortiz, "Generalization Metrics for
#     Practical Quantum Advantage in Generative Models", Phys. Rev. Applied 21, 044032 (2024).
#     arXiv:2201.08770.
#
# These measure a model trained on a STRICT SUBSET of the valid solution space by how well it
# generates unseen-yet-valid samples (true generalization, not memorization). For BAS,
# valid_patterns is the enumerated bars/stripes support (well-defined only under the `holdout`
# split; under `full_support`, train == valid space, so there is no unseen valid space and every
# metric is NaN). For JGB there is no separate notion of "valid" bitstring --
# every bit pattern decodes to some real value under the fixed-point encoding -- so valid_patterns
# is left None and the full 2^n bitstring hypercube is used as the valid space instead (n = the
# bitstring length, inferred from the samples; never materialized, only its size is used).
# --------------------------------------------------------------------------------------------------
def _to_bitstring_set(patterns) -> set:
    """Normalize a pattern container to a set of bitstrings.

    Accepts a {bitstring: count} dict (uses its keys), a 2-D binary ndarray (one row per pattern),
    or an iterable of bitstrings -- so callers can pass either the Counter splits or the raw
    enumerated-pattern arrays used elsewhere in this module.
    """
    if patterns is None:
        return set()
    if isinstance(patterns, dict):
        return set(patterns.keys())
    arr = np.asarray(patterns)
    if arr.dtype.kind in ("U", "S", "O"):   # already bitstrings
        return set(np.atleast_1d(arr).tolist())
    return set(array_to_str(np.atleast_2d(arr)))


def generalization_metrics(samples: dict, train_patterns, valid_patterns=None) -> dict:
    """Validity-based generalization metrics for constraint-satisfaction generative models.

    Following Gili, Mauri & Perdomo-Ortiz (arXiv:2207.13645) and Gili et al. (Phys. Rev. Applied 21,
    044032, 2024; arXiv:2201.08770). Quantifies generation of unseen-yet-valid samples from a model
    trained on a strict subset of the valid space -- i.e. generalization rather than memorization.

    Let Q be the number of queries (shots, counted WITH multiplicity), G_new the queries outside the
    training set, G_sol the queries that are valid AND outside the training set, g_sol the number of
    UNIQUE unseen-valid bitstrings generated, S the valid solution space and T the training-set size:

        exploration = |G_new| / Q                             fraction of samples that are novel
        fidelity    = |G_sol| / |G_new|                       precision: novel samples that are valid
        rate        = |G_sol| / Q      (= exploration*fidelity)   efficiency of useful generation
        coverage    = g_sol / (|S| - T)                       recall of the unseen valid space

    Counts are with multiplicity except `coverage` (unique bitstrings). Returns keys prefixed
    "bench_val/" (Gili et al. generalization suite; distinct from the bench_BAS/* precision-recall
    pair and the bench_dist/* distribution distances, which include a separate `fidelity` --
    Bhattacharyya distribution fidelity, not this precision measure). Metrics are NaN when their
    denominator is empty (Q=0, no novel samples, or no unseen valid space under BAS `full_support`).

    valid_patterns=None (JGB): every bitstring is valid, so S is the full 2^n hypercube (n = the
    bitstring length) rather than an enumerated finite set -- `fidelity` is then trivially 1.0
    (nothing generated can be invalid) and `rate` collapses to `exploration`; the non-trivial signal
    for JGB is `exploration`/`coverage`, i.e. how much of the encoding's unseen resolution the model
    reaches beyond the finite training sample.
    """
    train_set = _to_bitstring_set(train_patterns)
    nan = float("nan")
    Q = sum(samples.values())
    if Q == 0:
        return {"bench_val/exploration": nan, "bench_val/fidelity": nan,
                "bench_val/rate": nan, "bench_val/coverage": nan}

    if valid_patterns is None:
        n_bits = len(next(iter(samples)))
        unseen_size = 2 ** n_bits - len(train_set)

        def is_valid(_bitstring):
            return True
    else:
        valid_set = _to_bitstring_set(valid_patterns)
        unseen_size = len(valid_set - train_set)

        def is_valid(bitstring):
            return bitstring in valid_set

    q_new = q_sol = 0
    unique_sol = set()
    for bitstring, c in samples.items():
        valid = is_valid(bitstring)
        is_new = bitstring not in train_set
        if is_new:
            q_new += c
        if valid and is_new:
            q_sol += c
            unique_sol.add(bitstring)

    return {
        "bench_val/exploration": float(q_new / Q),
        "bench_val/fidelity": float(q_sol / q_new) if q_new > 0 else nan,
        "bench_val/rate": float(q_sol / Q),
        "bench_val/coverage": float(len(unique_sol) / unseen_size) if unseen_size else nan,
    }


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
             valid_patterns: np.ndarray = None, train_patterns=None) -> dict:
    """Run the appropriate metric bundle over each named split and return a flat metric dict.

    Args:
        samples: model sample counts {bitstring: count}
        splits: {split_name: target_count_dict}, e.g. {"val": ..., "test": ...}
        dataset_kind: "BAS" or "JGB"
        valid_patterns: enumerated valid BAS patterns (BAS only -- used by both bench_val and
            bench_BAS below); None for JGB, where bench_val instead treats every bitstring as
            valid (see generalization_metrics).
        train_patterns: seen (training-split) patterns as a Counter/ndarray; when supplied, the
            Gili et al. bench_val/* generalization metrics are computed for BOTH dataset kinds
            (BAS against the enumerated valid space, JGB against the full bitstring hypercube).

    Returns bench_dist/<split>/{mmd,kl,tv,fidelity} for every non-empty split, bench_val/* (all
    datasets, whenever train_patterns is given), and -- BAS only -- bench_BAS/{precision,recall,qbas}.
    """
    metrics = {}
    for name, target in splits.items():
        if not target:
            continue
        metrics[f"bench_dist/{name}/mmd"] = mmd(samples, target, sigmas)
        metrics[f"bench_dist/{name}/kl"] = kl_divergence(samples, target)
        metrics[f"bench_dist/{name}/tv"] = total_variation_distance(samples, target)
        metrics[f"bench_dist/{name}/fidelity"] = classical_fidelity(samples, target)
    if train_patterns is not None:
        metrics.update(generalization_metrics(samples, train_patterns, valid_patterns))
    if dataset_kind == "BAS" and valid_patterns is not None:
        metrics.update(qbas_metrics(samples, valid_patterns))
    return metrics


# --------------------------------------------------------------------------------------------------
# wandb fetch caches
#
# Every figure/table in a plotting pass is derived from the SAME sweep, so the run list and the
# per-run parsed config are fetched/parsed once per process and reused. Caches are keyed on
# identities that are immutable for a finished run (sweep path, run id) and are only ever additive,
# so the only thing they can miss is a run that started/finished mid-pass -- call reset_wandb_cache()
# if you need to re-read a live sweep in a long-running session (e.g. a notebook).
# --------------------------------------------------------------------------------------------------
_API = None                # wandb.Api() (its construction + default_entity lookup hit the network)
_SWEEP_RUNS_CACHE = {}     # (entity, project, sweep_id) -> [Run]
_CFG_CACHE = {}            # run.id -> parsed config (from_run_config)


def wandb_api():
    """The process-wide wandb.Api() instance (constructing one per call re-does auth/lookup work)."""
    global _API
    if _API is None:
        import wandb
        _API = wandb.Api()
    return _API


def sweep_runs(sweep_id: str, entity: str, project: str) -> list:
    """Materialized run list of a sweep -- ONE wandb query per (entity, project, sweep) per process.

    `Sweep.runs` is already a materialized list server-side, but re-requesting it per figure is
    the single most expensive redundant fetch in a plotting pass, so it is cached here.
    """
    api = wandb_api()
    entity = entity or api.default_entity  # unresolved None would literally build ".../None/..."
    key = (entity, project, sweep_id)
    if key not in _SWEEP_RUNS_CACHE:
        _SWEEP_RUNS_CACHE[key] = list(api.sweep(f"{entity}/{project}/{sweep_id}").runs)
    return _SWEEP_RUNS_CACHE[key]


def run_config(run):
    """Parsed (schema-validated) config of a run, cached by run id.

    Raises whatever from_run_config raises for a config predating the current schema; callers that
    want to skip such runs catch it (see _group_runs, plotting.fetch_runs).
    """
    if run.id not in _CFG_CACHE:
        _CFG_CACHE[run.id] = from_run_config(run.config)
    return _CFG_CACHE[run.id]


def reset_wandb_cache():
    """Drop the cached api/run-list/config state (use when re-reading a sweep that is still running)."""
    global _API
    _API = None
    _SWEEP_RUNS_CACHE.clear()
    _CFG_CACHE.clear()


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


def load_checkpoint(run, root: str = "./artifacts", which: str = "best"):
    """Download a run's model artifact and load (circuit, params).

    `which` selects "best" (validation-selected, default) or "final" (last training iteration) --
    see QCBM.save, which persists both checkpoints (best_params.npy / final_params.npy) alongside
    the circuit in every run's model artifact.

    Skips the download entirely -- including the network round-trip to fetch/verify the artifact
    manifest -- if both expected files are already present locally under root/run.id. Safe here
    because each run logs exactly one model artifact, written once at the end of training (see
    src/setup.py::setup_and_train_qcbm), so there is no newer version a stale local copy could miss.

    If a run has no model artifact -- artifact upload disabled (logging.wandb_log_artifacts) or its
    upload was refused, e.g. rate-limited -- this falls back to the training run's own output
    directory, whose absolute path every run records in its summary as `save_dir`. That only works
    when the benchmark runs on a machine that can see that path.
    """
    from qiskit import qpy
    if which not in ("best", "final"):
        raise ValueError(f"which must be 'best' or 'final', got {which!r}")
    local_dir = f"{root}/{run.id}"
    circuit_path = f"{local_dir}/circuit.qpy"
    params_path = f"{local_dir}/{which}_params.npy"

    if os.path.exists(circuit_path) and os.path.exists(params_path):
        print(f"[benchmark]     [{run.id}] checkpoint already downloaded, reusing {local_dir}")
    else:
        art = None
        for a in run.logged_artifacts():
            if a.type == "model":
                art = a
                break
        if art is not None:
            art.download(root=local_dir)
        else:
            save_dir = run.summary.get("save_dir")
            local = {"circuit.qpy": circuit_path, f"{which}_params.npy": params_path}
            if not save_dir or not all(os.path.exists(f"{save_dir}/{f}") for f in local):
                raise FileNotFoundError(
                    f"No model artifact for run {run.id} and no readable save_dir "
                    f"({save_dir!r}) to fall back on")
            print(f"[benchmark]     [{run.id}] no model artifact, loading from {save_dir}")
            os.makedirs(local_dir, exist_ok=True)
            for name, dest in local.items():
                shutil.copyfile(f"{save_dir}/{name}", dest)

    with open(circuit_path, "rb") as f:
        circuit = qpy.load(f)[0]
    params = np.load(params_path)
    return circuit, params


def sample_model(circuit, params: np.ndarray, n_shots: int, seed: int = 0) -> dict:
    """Sample a bound circuit via Aer's parameter_binds fast path -> {bitstring: count}."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector", runtime_parameter_bind_enable=True,
                       max_parallel_experiments=0, seed_simulator=seed)
    binds = {p: np.array([params[p.index]]) for p in circuit.parameters}
    counts = sim.run(circuit, parameter_binds=[binds], shots=n_shots).result().get_counts()
    return counts if isinstance(counts, dict) else counts[0]


def _test_split_for_config(cfg) -> tuple:
    """Reconstruct the held-out targets + dataset context from a run's config.

    Returns (splits, valid_patterns, train_counts): the val/test target dicts used for the
    distribution-distance metrics, the enumerated valid patterns (BAS only, else None), and the
    training-split counts (used for the generalization metrics' seen-set).
    """
    from src.data import BAS, JGB, DataLoader
    if cfg.data.dataset == "BAS":
        dataset = BAS(cfg.data.width, cfg.data.height)
    else:
        dataset = JGB(cfg.data.N_qubits, cfg.data.N_features)
    dl = DataLoader(dataset)
    # Reconstruct the SAME split training used: keyed on initial_random_seed (see
    # setup.compute_split), not the per-run random_seed, so the held-out val/test targets match the
    # run's training split (critical for BAS holdout, where the seed selects the partition).
    _, X_val, X_test, c_train, c_val, c_test = dl.train_val_test_split(
        cfg.data.train_split, cfg.data.val_split,
        seed=cfg.sweep.initial_random_seed, bas_split_mode=cfg.data.bas_split_mode)
    valid_patterns = dataset.binary if cfg.data.dataset == "BAS" else None
    return {"val": c_val, "test": c_test}, valid_patterns, c_train


_CONNECTIONS_CACHE = {}  # config fingerprint -> new-connection count


def extension_new_connection_count(cfg):
    """Number of NEW two-qubit connections a run's extension adds on top of the linear baseline.

    Training runs don't log this, so it is reconstructed from the config -- mirroring
    setup.setup_circuit_extensions' topology selection exactly (same helpers, same
    `linear_topology(range(N_qubits))` baseline, same set-difference as extend_circuit) but without
    building a circuit, so the count matches what actually trained. Returns None for an unrecognized
    extension rather than guessing.

    `random` needs no RNG here: setup sizes it to exactly match metric_based's NEW-connection count,
    so that count IS the answer. Cached per distinct dataset/extension config, since the dataset
    reconstruction (needed only by the data-driven topologies) is the expensive part.
    """
    from src.setup import setup_dataloader, compute_split, _metric_based_connections
    from src.extension import (linear_topology, all_to_all_topology, nearest_neighbor_topology,
                               chow_liu_topology)
    from src.utils import mutual_info_matrix

    extension = cfg.circuit.extension
    key = (extension, cfg.data.dataset, cfg.data.N_qubits, cfg.data.width, cfg.data.height,
           cfg.circuit.extension_metric, cfg.circuit.threshold_rule, cfg.circuit.threshold,
           cfg.data.train_split, cfg.data.val_split, cfg.data.bas_split_mode,
           cfg.sweep.initial_random_seed)
    if key in _CONNECTIONS_CACHE:
        return _CONNECTIONS_CACHE[key]

    init_connections = linear_topology(list(range(cfg.data.N_qubits)))
    if extension == "none":
        extension_connections = []
    elif extension == "all_to_all":
        extension_connections = all_to_all_topology(cfg.data.N_qubits)
    elif extension == "nearest_neighbor":
        extension_connections = nearest_neighbor_topology(cfg.data.width, cfg.data.height)
    elif extension in ("metric_based", "random", "chow_liu"):
        # the data-driven topologies need the same train split training used (compute_split keys on
        # initial_random_seed, so every run of the sweep sees the identical split/topology)
        X_train, *_ = compute_split(cfg, setup_dataloader(cfg))
        if extension == "chow_liu":
            extension_connections = chow_liu_topology(mutual_info_matrix(np.asarray(X_train)))
        else:
            extension_connections, _ = _metric_based_connections(
                X_train, cfg.circuit.extension_metric, cfg.circuit.threshold_rule,
                cfg.circuit.threshold)
    else:
        return None

    n = len(set(extension_connections) - set(init_connections))
    _CONNECTIONS_CACHE[key] = n
    return n


def _run_setup_facts(run) -> dict:
    """Per-run architecture/cost facts for the setup table: parameter count, added connections, and
    the total measurement/iteration cost actually spent. Summary keys are best-effort (a run predating
    a logging change simply reports None, which bootstrap_group_metrics renders as NaN)."""
    try:
        n_connections = extension_new_connection_count(run_config(run))
    except Exception as e:  # a config we can't reconstruct must not sink the whole benchmark
        print(f"[benchmark]     [{run.id}] could not derive connection count: {e!r}")
        n_connections = None
    return {"num_parameters": run.summary.get("train/num_parameters"),
            "n_connections": n_connections,
            "total_measurements": run.summary.get("total_measurements"),
            "iterations_run": run.summary.get("iterations_run")}


def _evaluate_run(run, n_shots: int, which: str = "best") -> dict:
    """Load one run's checkpoint, sample it, and evaluate the full held-out test metric suite.

    `which` selects "best" (validation-selected, default) or "final" (last training iteration) --
    see load_checkpoint.

    Returns the flat metric dict from `evaluate` (bench_dist/test/{mmd,kl,tv,fidelity}, bench_val/*,
    plus BAS-only bench_BAS/{precision,recall,qbas}). Shared by benchmark_sweep (best-per-group) and
    benchmark_all_runs (every run) so a run is scored identically regardless of caller.
    """
    circuit, params = load_checkpoint(run, which=which)
    cfg = run_config(run)
    splits, valid_patterns, train_counts = _test_split_for_config(cfg)
    samples = sample_model(circuit, params, n_shots, seed=cfg.sweep.random_seed)
    return evaluate(samples, splits, cfg.data.dataset, sigmas=np.array(cfg.qcbm.sigmas),
                    valid_patterns=valid_patterns, train_patterns=train_counts)


def _group_runs(sweep_id: str, entity: str, project: str, group_by: str) -> dict:
    """{group_key: [runs]} for a sweep, skipping runs whose config predates the current schema.

    Uses the process-wide sweep_runs/run_config caches, so calling this repeatedly (or alongside
    plotting.fetch_runs) costs exactly one wandb sweep query per sweep.
    """
    groups = {}
    for r in sweep_runs(sweep_id, entity, project):
        try:
            key = OmegaConf.select(run_config(r), group_by)
        except Exception as e:
            print(f"[benchmark]     skipping run {r.id}: config incompatible with current schema ({e})")
            continue
        groups.setdefault(key, []).append(r)
    return groups


def benchmark_sweep(sweep_id: str, entity: str, project: str, group_by: str = "circuit.extension",
                    n_shots: int = 10000, metric: str = "best_mmd_val",
                    groups: dict = None, which: str = "best") -> pd.DataFrame:
    """Benchmark the best model per group of a sweep (point estimate, one row per group).

    For each group (value of `group_by`, a dot-separated path into the run's config, e.g.
    "circuit.extension"), select the seed-run with the lowest validation MMD, load its checkpoint
    (`which`: "best" validation-selected iteration, or "final" -- see load_checkpoint), sample it,
    and evaluate the full metric suite on the held-out test split. For a bootstrap over ALL runs
    (mean +/- std across seeds) use benchmark_all_runs instead.

    `groups` optionally supplies an already-fetched {group_key: [runs]} mapping (e.g. from
    plotting.fetch_runs), skipping the sweep query entirely.
    """
    groups = groups if groups is not None else _group_runs(sweep_id, entity, project, group_by)
    rows = []
    for key, group_runs in groups.items():
        print(f"[benchmark]     [{key}] {len(group_runs)} run(s) -> selecting best by {metric}...")
        best = select_best_run(group_runs, metric)
        if best is None:
            print(f"[benchmark]     [{key}] no run with a finite {metric}, skipping.")
            continue
        print(f"[benchmark]     [{key}] best run {best.id} ({metric}={best.summary.get(metric)}); "
              f"evaluating checkpoint...")
        row = {group_by: key, "run_id": best.id, "run_name": best.name,
               "best_mmd_val": best.summary.get(metric), **_run_setup_facts(best)}
        row.update(_evaluate_run(best, n_shots, which=which))
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_all_runs(sweep_id: str, entity: str, project: str, group_by: str = "circuit.extension",
                       n_shots: int = 10000, metric: str = "best_mmd_val",
                       groups: dict = None, which: str = "best") -> pd.DataFrame:
    """Evaluate EVERY run's checkpoint on the held-out test split (not just the per-group best).

    `which` selects "best" (validation-selected, default) or "final" (last training iteration) for
    EVERY run's checkpoint -- see load_checkpoint.

    Returns a tidy table with ONE ROW PER RUN (group_by, run_id, run_name, best_mmd_val,
    num_parameters, + the full held-out metric suite), so downstream code
    (plotting.bootstrap_group_metrics) can bootstrap the metrics across the runs of each group --
    the mean +/- across-seed standard error. Each run is scored identically to benchmark_sweep (same
    `which` checkpoint, same n_shots), so the per-group best row of this table matches
    benchmark_sweep's point estimate.

    Runs with an incompatible config or no usable checkpoint are skipped with a warning rather than
    aborting the whole benchmark.

    `groups` optionally supplies an already-fetched {group_key: [runs]} mapping (e.g. from
    plotting.fetch_runs), skipping the sweep query entirely.
    """
    groups = groups if groups is not None else _group_runs(sweep_id, entity, project, group_by)
    rows = []
    for key, group_runs in groups.items():
        print(f"[benchmark]     [{key}] evaluating {len(group_runs)} run(s) @ {n_shots} shots...")
        for i, r in enumerate(group_runs):
            try:
                metrics = _evaluate_run(r, n_shots, which=which)
            except Exception as e:  # one bad run must not sink the whole group
                print(f"[benchmark]     [{key}] skipping run {r.id}: {e!r}")
                continue
            row = {group_by: key, "run_id": r.id, "run_name": r.name,
                   "best_mmd_val": r.summary.get(metric), **_run_setup_facts(r)}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)
