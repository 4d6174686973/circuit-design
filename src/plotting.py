"""wandb-driven plotting for QCBM sweeps.

Replaces the hardcoded-path `results v1/plot_figures.ipynb`: figures that depend on training runs
are built by pulling run histories from a wandb sweep (grouped by any config dimension), and MMD is
plotted against cumulative circuit *measurements* rather than iteration index. Run-independent
dataset/topology figures are ported from the notebook. All figures are saved as PDF (and PNG).

Typical use:
    from src.plotting import generate_all_figures
    generate_all_figures(sweep_id="<sweep>", entity="<you>", project="qcbm-circuit-design",
                         dataset_cfg={"dataset": "BAS", "width": 3, "height": 3, "N_qubits": 9})
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.extension import (add_su4_gate, linear_topology, nearest_neighbor_topology,
                           all_to_all_topology, metric_based_topology)
from src.data import BAS, JGB, DataLoader, init_qubit_order_bas
from src.utils import varInfoMat, get_features_for_quasi_dist, array_to_str
from src import benchmark as bm


# --------------------------------------------------------------------------------------------------
# styling helpers
# --------------------------------------------------------------------------------------------------
def use_science_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass


# canonical extension legend labels (matches the v1 notebook)
_EXTENSION_LABELS = {
    "none": "linear",
    "linear": "linear",
    "nearest_neighbor": "nearest-neighbor",
    "random": "random",
    "metric_based": "metric-based",
    "all_to_all": "all-to-all",
}


def default_colors(legend_keys) -> dict:
    """Color map preserving the v1 convention for extensions; sequential for numeric sweeps."""
    keys = list(legend_keys)
    # numeric sweep dimension -> sequential colormap ordered by value
    try:
        numeric = sorted(keys, key=lambda k: float(k))
        blues = plt.cm.viridis(np.linspace(0.15, 0.9, len(numeric)))
        return {k: blues[i] for i, k in enumerate(numeric)}
    except (TypeError, ValueError):
        pass
    named = {"metric-based": "darkorange", "metric_based": "darkorange",
             "nearest-neighbor": "slategray", "nearest_neighbor": "slategray"}
    blues = plt.cm.Blues(np.linspace(0.3, 1, max(len(keys), 2)))[::-1]
    colors = {}
    bi = 0
    for k in keys:
        label = _EXTENSION_LABELS.get(k, k)
        if label in named:
            colors[k] = named[label]
        else:
            colors[k] = blues[bi % len(blues)]
            bi += 1
    return colors


def _save(fig, plots_dir, filename):
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(f"{plots_dir}/{filename}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"{plots_dir}/{filename}.png", bbox_inches="tight", transparent=False, dpi=300)


# --------------------------------------------------------------------------------------------------
# wandb fetch + aggregation
# --------------------------------------------------------------------------------------------------
def fetch_runs(sweep_id: str, entity: str, project: str, group_by: str = "extension",
               filters: dict = None) -> dict:
    """Return {legend_key: [runs]} for a real wandb Sweep, grouped by a config dimension.

    filters is an optional {config_key: value} dict to pin non-legend swept params (facet slicing),
    applied client-side since Sweep.runs is a materialized list, not a server-side query.
    """
    import wandb
    api = wandb.Api()
    runs = list(api.sweep(f"{entity}/{project}/{sweep_id}").runs)
    if filters:
        runs = [r for r in runs if all(r.config.get(k) == v for k, v in filters.items())]
    grouped = {}
    for r in runs:
        grouped.setdefault(r.config.get(group_by), []).append(r)
    return grouped


def fetch_history(run, metric: str = "mmd_train") -> pd.DataFrame:
    """Per-run dataframe with cumulative_measurements and the requested metric (NaN rows dropped)."""
    keys = ["cumulative_measurements", metric]
    df = run.history(keys=keys, pandas=True)
    if df is None or df.empty or metric not in df:
        # fallback: reconstruct measurements from config if not logged
        n = run.config.get("iterations", 0)
        P = run.summary.get("num_parameters") or run.config.get("num_parameters", 0)
        shots = run.config.get("N_shots", 0)
        per = (2 * P + 1) * shots
        df = pd.DataFrame({"cumulative_measurements": np.arange(1, n + 1) * per,
                           metric: [np.nan] * n})
    return df.dropna(subset=[metric]).sort_values("cumulative_measurements")


def aggregate_over_measurements(histories: list, metric: str, mode: str = "medperc",
                                n_grid: int = 400, window: int = 1) -> dict:
    """Interpolate each run onto a common measurement grid, then aggregate across seeds.

    Different legend keys have different measurements/iteration, so runs are not index-aligned; we
    interpolate onto a shared x-grid before mean/std or median/percentile aggregation.
    """
    curves = [h for h in histories if len(h) > 0]
    if not curves:
        return None
    lo = max(c["cumulative_measurements"].min() for c in curves)
    hi = min(c["cumulative_measurements"].max() for c in curves)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, n_grid)
    stacked = np.vstack([
        np.interp(grid, c["cumulative_measurements"].values, c[metric].values) for c in curves
    ])
    if mode == "meanstd":
        line = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        lower, upper = line - std, line + std
    else:  # medperc
        line = np.median(stacked, axis=0)
        lower = np.percentile(stacked, 10, axis=0)
        upper = np.percentile(stacked, 90, axis=0)
    if window > 1:
        smooth = lambda a: pd.Series(a).rolling(window, min_periods=1).mean().values
        line, lower, upper = smooth(line), smooth(lower), smooth(upper)
    return {"x": grid, "line": line, "lower": lower, "upper": upper}


def plot_mmd_vs_measurements(runs_by_key: dict, metric: str = "mmd_train", mode: str = "medperc",
                             window: int = 1, colors: dict = None, filename: str = "MMD_measurements",
                             plots_dir: str = "plots", save: bool = True):
    """MMD (or any logged metric) vs cumulative measurements, aggregated across seeds per group."""
    colors = colors or default_colors(runs_by_key.keys())
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for key, runs in runs_by_key.items():
        hists = [fetch_history(r, metric) for r in runs]
        agg = aggregate_over_measurements(hists, metric, mode, window=window)
        if agg is None:
            continue
        label = _EXTENSION_LABELS.get(key, str(key))
        ax.plot(agg["x"], agg["line"], label=label, color=colors.get(key))
        ax.fill_between(agg["x"], agg["lower"], agg["upper"], alpha=0.3, lw=0.0, color=colors.get(key))
    ax.set_xlabel("Measurements")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


# --------------------------------------------------------------------------------------------------
# run-independent dataset / topology figures (ported from plot_figures.ipynb)
# --------------------------------------------------------------------------------------------------
def plot_su4_gate(plots_dir: str = "plots", save: bool = True):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    qc = QuantumCircuit(2)
    params = ParameterVector("theta", 15)
    add_su4_gate(qc, 0, 1, params)
    fig = qc.draw(output="mpl")
    if save:
        _save(fig, plots_dir, "SU4_gate")
    return fig


def plot_bas_images(width=3, height=3, plots_dir: str = "plots", save: bool = True):
    bas = BAS(width, height)
    imgs = bas.binary
    fig, axes = plt.subplots(1, len(imgs), figsize=(len(imgs), 2))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(imgs[i].reshape(width, height), norm=plt.Normalize(0, 1), cmap="gray")
        for _, spine in ax.spines.items():
            spine.set_visible(True); spine.set_color("black"); spine.set_linewidth(1)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, "BAS_images")
    return fig


def plot_bas_extension(width=3, height=3, threshold=0.5, plots_dir: str = "plots", save: bool = True):
    import seaborn as sns
    from scipy.spatial import distance
    X = BAS(width, height).binary
    hamming = distance.cdist(X.T, X.T, "hamming")
    dim = hamming.shape[0]
    dist_filter = np.zeros_like(hamming)
    dist_filter[hamming < threshold] = 1.0
    dist_filter -= np.eye(dim)
    fig, axs = plt.subplots(1, 2, figsize=(6, 2.6))
    sns.heatmap(hamming, cmap="Blues", ax=axs[0], vmin=0.0, vmax=1.0)
    axs[0].set_title("a) Hamming distance")
    sns.heatmap(dist_filter, cmap="Blues", ax=axs[1])
    axs[1].set_title("b) Circuit Extension")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, "BAS_extension")
    return fig


def _draw_topology(ax, n_qubits, edges_base, edges_ext, title):
    edges_new = sorted(set(edges_ext) - set(edges_base))
    G = nx.Graph(); G.add_nodes_from(range(n_qubits))
    G.add_edges_from(edges_base); G.add_edges_from(edges_ext)
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="white", edgecolors="black", node_size=200, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_base, edge_color="black", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_new, edge_color="cornflowerblue", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
    ax.set_title(title); ax.set_aspect("equal"); ax.set_frame_on(False)
    ax.set_xticks([]); ax.set_yticks([])


def plot_bas_topology(width=3, height=3, threshold=0.5, plots_dir: str = "plots", save: bool = True):
    from scipy.spatial import distance
    n_qubits = width * height
    X = BAS(width, height).binary
    hamming = distance.cdist(X.T, X.T, "hamming")
    edges_lin = linear_topology(init_qubit_order_bas["3x3"]) if (width, height) == (3, 3) \
        else linear_topology(list(range(n_qubits)))
    extensions = {
        "a) Linear": edges_lin,
        "b) Nearest-Neighbor": nearest_neighbor_topology(width, height),
        "c) Metric-Based": metric_based_topology(hamming, threshold),
        "d) All-to-All": all_to_all_topology(n_qubits),
    }
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    for ax, (name, edges) in zip(axs.flatten(), extensions.items()):
        _draw_topology(ax, n_qubits, edges_lin, edges, name)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, "BAS_topology")
    return fig


def plot_jgb_raw_data(N_qubits=12, N_features=3, plots_dir: str = "plots", save: bool = True):
    jgb = JGB(N_qubits, N_features)
    df0 = jgb.raw
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(df0.columns)))[::-1]
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for i, c in enumerate(df0.columns):
        ax.plot(df0[c], label=f"{c[:-1]}-year Rate", color=colors[i])
    ax.set_xlabel("Date"); ax.set_ylabel("Interest Rate [%]"); ax.legend()
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, "JGB_raw_data")
    return fig


def plot_jgb_binary_histograms(N_qubits=12, N_features=3, plots_dir: str = "plots", save: bool = True):
    from collections import Counter
    from qiskit.visualization import plot_histogram
    from matplotlib.colors import to_hex
    jgb = JGB(N_qubits, N_features)
    bits_per_feature = N_qubits // N_features
    target_dict = Counter(array_to_str(jgb.binary))
    feat_dicts = get_features_for_quasi_dist(target_dict, bits_per_feature, N_features)
    labels = ([f"{t}" for t in ["5-year", "10-year", "20-year"]] if N_features == 3
              else ["2-year", "5-year", "10-year", "20-year"])
    colors = [to_hex(c) for c in plt.cm.Blues(np.linspace(0.4, 1, N_features))[::-1]]
    fig, axs = plt.subplots(N_features, 1, figsize=(5, 5))
    for i, ax in enumerate(np.atleast_1d(axs)):
        plot_histogram(feat_dicts[i], ax=ax, bar_labels=False, color=colors[i])
        ax.set_title(labels[i])
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, "JGB_binary_histograms")
    return fig


def plot_jgb_threshold(N_qubits=12, N_features=3, plots_dir: str = "plots", save: bool = True):
    jgb = JGB(N_qubits, N_features); dl = DataLoader(jgb)
    X, *_ = dl.train_val_test_split(0.7, 0.15)
    num_steps = 100
    steps = np.linspace(1 / num_steps, 1, num_steps)
    num_connections = np.zeros_like(steps)
    varinfo = varInfoMat(pd.DataFrame(X), norm=True).values
    dim = varinfo.shape[0]
    for i, threshold in enumerate(steps):
        dist_filter = np.zeros_like(varinfo)
        dist_filter[varinfo < threshold] = 1.0
        dist_filter -= np.eye(dim)
        num_connections[i] = np.sum(dist_filter) / 2
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(steps, num_connections, color=plt.cm.Blues(0.8))
    ax.set_xlabel("Threshold"); ax.set_ylabel("Number of Connections")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, "JGB_threshold")
    return fig


def plot_jgb_extension(N_qubits=12, N_features=3, threshold=0.95, plots_dir: str = "plots", save: bool = True):
    import seaborn as sns
    jgb = JGB(N_qubits, N_features); dl = DataLoader(jgb)
    X, *_ = dl.train_val_test_split(0.7, 0.15)
    varinfo = varInfoMat(pd.DataFrame(X), norm=True).values
    dim = varinfo.shape[0]
    dist_filter = np.zeros_like(varinfo)
    dist_filter[varinfo < threshold] = 1.0
    dist_filter -= np.eye(dim)
    fig, axs = plt.subplots(1, 2, figsize=(6, 2.6))
    sns.heatmap(varinfo, ax=axs[0], cmap="Blues", vmin=0, vmax=1)
    axs[0].set_title("a) Variation of Information")
    sns.heatmap(dist_filter, ax=axs[1], cmap="Blues", vmin=0, vmax=1)
    axs[1].set_title("b) Circuit Extension")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, "JGB_extension")
    return fig


def plot_jgb_topology(N_qubits=12, N_features=3, threshold=0.95, plots_dir: str = "plots", save: bool = True):
    jgb = JGB(N_qubits, N_features); dl = DataLoader(jgb)
    X, *_ = dl.train_val_test_split(0.7, 0.15)
    varinfo = varInfoMat(pd.DataFrame(X), norm=True)
    edges_lin = [(i, i + 1) for i in range(N_qubits - 1)]
    edges_ext = metric_based_topology(varinfo.values, threshold)
    fig, axs = plt.subplots(1, 2, figsize=(6, 2.9))
    _draw_topology(axs[0], N_qubits, edges_lin, edges_lin, "a) Linear")
    _draw_topology(axs[1], N_qubits, edges_lin, edges_ext, "b) Extended")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, "JGB_topology")
    return fig


# --------------------------------------------------------------------------------------------------
# benchmark figures (best model per group)
# --------------------------------------------------------------------------------------------------
def plot_metric_bars(bench_df: pd.DataFrame, metric_col: str = "test/mmd", group_by: str = "extension",
                     plots_dir: str = "plots", filename: str = None, save: bool = True):
    """Bar chart of a benchmark metric across groups (best model per group)."""
    if bench_df is None or bench_df.empty or metric_col not in bench_df:
        return None
    filename = filename or f"benchmark_{metric_col.replace('/', '_')}"
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = [_EXTENSION_LABELS.get(k, str(k)) for k in bench_df[group_by]]
    colors = default_colors(bench_df[group_by])
    ax.bar(labels, bench_df[metric_col], color=[colors.get(k) for k in bench_df[group_by]])
    ax.set_ylabel(metric_col); ax.set_xlabel(group_by)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


def plot_qq_grid(sweep_id: str, entity: str, project: str, group_by: str = "extension",
                 n_shots: int = 10000, n_q: int = 100, plots_dir: str = "plots", save: bool = True):
    """Per-group QQ plots (model vs data, model vs normal, data vs normal) for JGB best models."""
    import wandb
    api = wandb.Api()
    grouped = fetch_runs(sweep_id, entity, project, group_by)
    figs = {}
    for key, runs in grouped.items():
        best = bm.select_best_run(runs)
        if best is None or best.config.get("dataset") != "JGB":
            continue
        circuit, params = bm.load_checkpoint(best)
        samples = bm.sample_model(circuit, params, n_shots, seed=best.config.get("random_seed", 0))
        splits, _ = bm._test_split_for_config(best.config)
        cfg = best.config
        jgb = JGB(cfg["N_qubits"], cfg["N_features"]); dl = DataLoader(jgb)
        dl.train_val_test_split(cfg["train_split"], cfg["val_split"])
        xmin, xmax = dl.conv_min_max
        bpf = jgb.bits_per_feature
        feats = bm.reconstruct_features(samples, bpf, cfg["N_features"], xmin, xmax)
        data = jgb.decimal.values
        fig, axs = plt.subplots(1, cfg["N_features"], figsize=(3 * cfg["N_features"], 3))
        for i, ax in enumerate(np.atleast_1d(axs)):
            mv, mp = feats[i]
            dx, my = bm.qq_model_vs_data(mv, mp, data[:, i], n_q)
            nx_, ny = bm.qq_model_vs_normal(mv, mp, n_q)
            ndx, ndy = bm.qq_data_vs_normal(data[:, i], n_q)
            ax.plot(dx, my, ".", ms=3, label="model vs data")
            ax.plot(nx_, ny, ".", ms=3, label="model vs normal")
            ax.plot(ndx, ndy, ".", ms=3, label="data vs normal")
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, "k--", lw=0.6)
            ax.set_title(f"feature {i}")
        axs.flatten()[0].legend(fontsize=6) if hasattr(axs, "flatten") else axs.legend(fontsize=6)
        plt.tight_layout()
        label = _EXTENSION_LABELS.get(key, str(key)).replace("/", "_")
        if save:
            _save(fig, plots_dir, f"JGB_QQ_{label}")
        figs[key] = fig
    return figs


# --------------------------------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------------------------------
def generate_all_figures(sweep_id: str, entity: str, project: str, dataset_cfg: dict,
                         group_by: str = "extension", metrics=("mmd_train", "mmd_test"),
                         plots_dir: str = "plots", science_style: bool = True):
    """Generate the full figure set for a sweep: static dataset/topology + MMD-vs-measurements +
    best-model benchmark (metric bars, and QQ grids for JGB). Saves PDFs to plots_dir."""
    if science_style:
        use_science_style()
    dataset = dataset_cfg.get("dataset", "BAS")

    # 1) run-independent figures
    plot_su4_gate(plots_dir)
    if dataset == "BAS":
        w, h = dataset_cfg.get("width", 3), dataset_cfg.get("height", 3)
        plot_bas_images(w, h, plots_dir=plots_dir)
        plot_bas_extension(w, h, dataset_cfg.get("extension_threshhold", 0.5), plots_dir=plots_dir)
        plot_bas_topology(w, h, dataset_cfg.get("extension_threshhold", 0.5), plots_dir=plots_dir)
    else:
        nq, nf = dataset_cfg.get("N_qubits", 12), dataset_cfg.get("N_features", 3)
        plot_jgb_raw_data(nq, nf, plots_dir=plots_dir)
        plot_jgb_binary_histograms(nq, nf, plots_dir=plots_dir)
        plot_jgb_threshold(nq, nf, plots_dir=plots_dir)
        plot_jgb_extension(nq, nf, dataset_cfg.get("extension_threshhold", 0.95), plots_dir=plots_dir)
        plot_jgb_topology(nq, nf, dataset_cfg.get("extension_threshhold", 0.95), plots_dir=plots_dir)

    # 2) MMD-vs-measurements over all seeds
    grouped = fetch_runs(sweep_id, entity, project, group_by)
    for metric in metrics:
        plot_mmd_vs_measurements(grouped, metric=metric, filename=f"{metric}_measurements",
                                 plots_dir=plots_dir)

    # 3) best-model benchmark figures
    bench_df = bm.benchmark_sweep(sweep_id, entity, project, group_by)
    for col in ["test/mmd", "test/tv", "test/fidelity"]:
        plot_metric_bars(bench_df, col, group_by, plots_dir=plots_dir)
    if dataset == "JGB":
        plot_qq_grid(sweep_id, entity, project, group_by, plots_dir=plots_dir)
    return bench_df
