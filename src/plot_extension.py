"""Static, training-independent figures investigating the thresholding rule and circuit extension.

Everything here depends only on the dataset + Hydra config -- never on a trained/wandb run -- so it
only needs to be regenerated when the data or extension settings change, unlike the per-sweep
figures in src.plotting (MMD curves, benchmark tables, QQ grids). The SU(4) gate diagram is more
general still: it doesn't depend on the dataset at all, so it is written once to a shared
`general/` folder rather than duplicated per dataset.

The metric_based extension threshold is no longer hand-tuned: it is auto-selected via
cfg.circuit.threshold_rule (see extension.select_threshold), either at the knee of the
connections-vs-threshold curve or at the bond-percolation threshold. This module computes that same
threshold -- via the same helpers setup.setup_circuit_extensions uses -- and marks it on the
threshold curve, the extension heatmap, and the topology network, so what's plotted always matches
what a real training run would build.

Typical use:
    uv run python -m src.plot_extension                      # both datasets -> plots/BAS_3x3/, plots/JGB_12Q3F/
    uv run python -m src.plot_extension --dataset BAS
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from hydra import initialize, compose

import src.config_schema  # noqa: F401 -- registers Config with Hydra's ConfigStore for compose()
from src.extension import (linear_topology, nearest_neighbor_topology, all_to_all_topology,
                           metric_based_topology, chow_liu_topology, add_su4_gate,
                           connection_threshold_curve, select_threshold)
from src.utils import mutual_info_matrix, feature_distance_matrix, get_features_for_quasi_dist, array_to_str
from src.data import BAS, JGB
from src.setup import setup_dataloader, compute_split
from src.plotting import _save, use_science_style, OKABE_ITO

_METRIC_LABELS = {"hamming": "Hamming distance", "varinfo": "Variation of Information"}


# --------------------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------------------
def load_cfg(dataset: str):
    """Compose the Hydra config for one dataset, exactly as a real training run would see it
    (src/conf/config.yaml + src/conf/dataset/<dataset>.yaml)."""
    with initialize(version_base=None, config_path="conf"):
        return compose(config_name="config", overrides=[f"dataset={dataset}"])


def dataset_folder(cfg) -> str:
    if cfg.data.dataset == "BAS":
        return f"BAS_{cfg.data.width}x{cfg.data.height}"
    return f"JGB_{cfg.data.N_qubits}Q{cfg.data.N_features}F"


# --------------------------------------------------------------------------------------------------
# general (dataset-independent, generated once)
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


# --------------------------------------------------------------------------------------------------
# dataset preprocessing figures (unreordered raw data -- these are about the dataset itself, not
# the circuit's qubit graph, so they must NOT go through the qubit-reordering used for training)
# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
# thresholding / extension figures (reordered train split -- must match the qubit indices setup.py
# actually builds the circuit on)
# --------------------------------------------------------------------------------------------------
def plot_threshold_curve(thresholds: np.ndarray, counts: np.ndarray, threshold: float,
                         rule: str = "knee", plots_dir: str = "plots",
                         filename: str = "threshold_curve", save: bool = True):
    """Connections-vs-threshold curve for the metric_based extension, with the auto-selected
    threshold marked (vertical line + annotation of the threshold and the resulting connection
    count)."""
    threshold_idx = int(np.argmin(np.abs(thresholds - threshold)))
    threshold_count = int(counts[threshold_idx])
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(thresholds, counts, color=plt.cm.Blues(0.8))
    ax.axvline(threshold, color=OKABE_ITO["vermillion"], ls="--", lw=1)
    ax.annotate(f"{rule} @ {threshold:.3f}\n({threshold_count} conn.)", xy=(threshold, threshold_count),
               xytext=(8, 4), textcoords="offset points", fontsize=7, color=OKABE_ITO["vermillion"])
    ax.set_xlabel("Threshold"); ax.set_ylabel("Number of Connections")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


def plot_extension_heatmap(distmat: np.ndarray, threshold: float, rule: str = "knee",
                           plots_dir: str = "plots", metric_label: str = "Distance",
                           filename: str = "extension_heatmap", save: bool = True):
    """Distance-matrix heatmap next to the thresholded circuit-extension mask."""
    import seaborn as sns
    dim = distmat.shape[0]
    dist_filter = np.zeros_like(distmat)
    dist_filter[distmat < threshold] = 1.0
    dist_filter -= np.eye(dim)
    fig, axs = plt.subplots(1, 2, figsize=(6, 2.6))
    sns.heatmap(distmat, cmap="Blues", ax=axs[0], vmin=0.0, vmax=1.0)
    axs[0].set_title(f"a) {metric_label}")
    sns.heatmap(dist_filter, cmap="Blues", ax=axs[1])
    axs[1].set_title(f"b) Circuit Extension ({rule}={threshold:.3f})")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


def _draw_topology(ax, n_qubits, edges_base, edges_ext, title):
    edges_new = sorted(set(edges_ext) - set(edges_base))
    G = nx.Graph(); G.add_nodes_from(range(n_qubits))
    G.add_edges_from(edges_base); G.add_edges_from(edges_ext)
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="white", edgecolors="black", node_size=200, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_base, edge_color="black", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_new, edge_color=OKABE_ITO["blue"], ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
    ax.set_title(title); ax.set_aspect("equal"); ax.set_frame_on(False)
    ax.set_xticks([]); ax.set_yticks([])


def plot_topology_panel(cfg, X_train: np.ndarray, distmat: np.ndarray, threshold: float,
                        rule: str = "knee", plots_dir: str = "plots", save: bool = True):
    """Topology networks over the same qubit indices the trained circuit actually uses (X_train is
    the reordered split from compute_split, so linear/metric-based/chow-liu edges here are directly
    comparable to setup.setup_circuit_extensions' init_connections/extension_connections)."""
    dataset = cfg.data.dataset
    n_qubits = cfg.data.N_qubits
    edges_lin = linear_topology(list(range(n_qubits)))
    edges_metric = metric_based_topology(distmat, threshold)
    edges_tree = chow_liu_topology(mutual_info_matrix(np.asarray(X_train)))

    if dataset == "BAS":
        width, height = cfg.data.width, cfg.data.height
        panels = {
            "a) Linear": edges_lin,
            "b) Nearest-Neighbor": nearest_neighbor_topology(width, height),
            f"c) Metric-Based ({rule}={threshold:.2f})": edges_metric,
            "d) Chow-Liu": edges_tree,
            "e) All-to-All": all_to_all_topology(n_qubits),
        }
    else:
        panels = {
            "a) Linear": edges_lin,
            f"b) Metric-Based ({rule}={threshold:.2f})": edges_metric,
            "c) Chow-Liu": edges_tree,
        }

    fig, axs = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
    for ax, (name, edges) in zip(np.atleast_1d(axs), panels.items()):
        _draw_topology(ax, n_qubits, edges_lin, edges, name)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, f"{dataset}_topology")
    return fig


# --------------------------------------------------------------------------------------------------
# per-dataset driver
# --------------------------------------------------------------------------------------------------
def generate_extension_figures(cfg, plots_dir: str = "plots") -> float:
    """Generate every dataset-only figure for one dataset config: preprocessing, threshold curve,
    extension heatmap, and topology networks, all sharing one auto-selected threshold (rule per
    cfg.circuit.threshold_rule). Returns the selected threshold."""
    dataset = cfg.data.dataset
    rule = cfg.circuit.threshold_rule
    folder = os.path.join(plots_dir, dataset_folder(cfg))

    dl = setup_dataloader(cfg)
    X_train, *_ = compute_split(cfg, dl)  # exact reordered train split setup.py trains the circuit on

    distmat = feature_distance_matrix(X_train, cfg.circuit.extension_metric)
    thresholds, counts = connection_threshold_curve(distmat)
    threshold = select_threshold(distmat, rule)
    threshold_idx = int(np.argmin(np.abs(thresholds - threshold)))
    print(f"[plot_extension] {dataset_folder(cfg)}: {rule} threshold = {threshold:.4f} "
         f"({int(counts[threshold_idx])} connections) -> {folder}/")

    if dataset == "BAS":
        plot_bas_images(cfg.data.width, cfg.data.height, plots_dir=folder)
    else:
        plot_jgb_raw_data(cfg.data.N_qubits, cfg.data.N_features, plots_dir=folder)
        plot_jgb_binary_histograms(cfg.data.N_qubits, cfg.data.N_features, plots_dir=folder)

    plot_threshold_curve(thresholds, counts, threshold, rule=rule, plots_dir=folder)
    plot_extension_heatmap(distmat, threshold, rule=rule, plots_dir=folder,
                           metric_label=_METRIC_LABELS[cfg.circuit.extension_metric])
    plot_topology_panel(cfg, X_train, distmat, threshold, rule=rule, plots_dir=folder)

    return threshold


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m src.plot_extension",
        description="Generate every training-independent figure investigating the thresholding "
                    "rule and circuit extension (SU(4) gate, preprocessing, threshold curve with "
                    "the auto-selected threshold, extension heatmaps, topology networks), driven by the "
                    "Hydra config, into per-dataset folders (e.g. BAS_3x3/, JGB_12Q3F/).")
    parser.add_argument("--dataset", choices=["BAS", "JGB", "both"], default="both",
                        help="Which dataset config(s) to render (default: both).")
    parser.add_argument("--plots-dir", default="plots", help="Output directory root.")
    parser.add_argument("--no-science-style", action="store_true",
                        help="Skip the scienceplots styling (use matplotlib defaults).")
    args = parser.parse_args(argv)

    if not args.no_science_style:
        use_science_style()

    plot_su4_gate(os.path.join(args.plots_dir, "general"))

    datasets = ["BAS", "JGB"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        generate_extension_figures(load_cfg(ds), args.plots_dir)

    print(f"Figures written under {args.plots_dir}/")


if __name__ == "__main__":
    main()
