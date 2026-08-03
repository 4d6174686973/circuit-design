"""Static, training-independent figures investigating the thresholding rule and circuit extension.

Everything here depends only on the dataset + Hydra config -- never on a trained/wandb run -- so it
only needs to be regenerated when the data or extension settings change, unlike the per-sweep
figures in src.plotting (MMD curves, benchmark tables, the QQ figure). The SU(4) gate diagram is more
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
from matplotlib.patches import Patch
import networkx as nx

from hydra import initialize, compose

import src.config_schema  # noqa: F401 -- registers Config with Hydra's ConfigStore for compose()
from src.extension import (linear_topology, all_to_all_topology,
                           metric_based_topology, chow_liu_topology, add_su4_gate,
                           connection_threshold_curve, select_threshold, knee_threshold,
                           percolation_threshold)
from src.utils import mutual_info_matrix, feature_distance_matrix, get_features_for_quasi_dist, array_to_str
from src.data import BAS, JGB, DataLoader, init_qubit_order_bas
from src.setup import setup_dataloader, compute_split
from src.plotting import _save, use_science_style, OKABE_ITO, SEQUENTIAL_CMAP, categorical_colors

_METRIC_LABELS = {"hamming": "Hamming distance", "varinfo": "Variation of Information"}

# shared train/val/test color coding for the BAS-image and JGB-raw-data split figures
_SPLIT_ORDER = ["train", "val", "test"]
_SPLIT_BASE_COLORS = {"train": OKABE_ITO["blue"], "val": OKABE_ITO["orange"], "test": OKABE_ITO["bluish_green"]}
_SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}


def _split_color(label: str) -> str:
    """Color for a split label; combined labels (e.g. "train+val+test", from BAS full_support mode,
    where every sample belongs to all three splits at once) get a neutral color of their own."""
    return _SPLIT_BASE_COLORS.get(label, OKABE_ITO["black"])


def _split_sort_key(label: str) -> int:
    return _SPLIT_ORDER.index(label) if label in _SPLIT_ORDER else len(_SPLIT_ORDER)


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
    # the quantizer changes the encoding and therefore every JGB figure; suffix only when it is
    # non-default so existing JGB_<n>Q<m>F plot paths keep working
    suffix = "" if cfg.data.quantizer == "minmax" else f"_{cfg.data.quantizer}"
    return f"JGB_{cfg.data.N_qubits}Q{cfg.data.N_features}F{suffix}"


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
def _bas_split_labels(bas: BAS, train_split: float, val_split: float, seed: int,
                      bas_split_mode: str) -> list:
    """Per-pattern split membership label ("train", "val", "test", or a combined
    "train+val+test" under bas_split_mode="full_support", where every pattern is in every split).

    Delegates to DataLoader.train_val_test_split itself (reorder=False, so rows stay in bas.binary's
    original order/columns) rather than re-deriving the seeded permutation, so this can't drift from
    the actual split logic in src.data. Patterns are unique 0/1 vectors, so membership is recovered by
    matching each row of bas.binary against the returned split arrays.
    """
    X_train, X_val, X_test, *_ = DataLoader(bas).train_val_test_split(
        train_split, val_split, reorder=False, seed=seed, bas_split_mode=bas_split_mode)
    split_sets = {"train": {tuple(r) for r in X_train}, "val": {tuple(r) for r in X_val},
                 "test": {tuple(r) for r in X_test}}
    labels = []
    for row in bas.binary:
        row_t = tuple(row)
        labels.append("+".join(s for s in _SPLIT_ORDER if row_t in split_sets[s]))
    return labels


def plot_bas_images(width=3, height=3, train_split=0.5, val_split=0.25, seed=None,
                    bas_split_mode="full_support", plots_dir: str = "plots", save: bool = True):
    """All enumerated BAS patterns, bordered by train/val/test split membership (see
    _bas_split_labels; under bas_split_mode="full_support" every pattern is in every split, so all
    borders share one neutral color)."""
    from collections import Counter
    bas = BAS(width, height)
    imgs = bas.binary
    labels = _bas_split_labels(bas, train_split, val_split, seed, bas_split_mode)
    label_counts = Counter(labels)
    n = len(imgs)

    # wrap into a roughly square grid instead of one long row -- a single row stays readable for a
    # handful of patterns but keeps stretching the figure wider as the enumerated support grows.
    # BAS 3x3 (n=14) is laid out as a fixed 2x7 block rather than the generic near-square grid.
    if width == 3 and height == 3:
        ncols, nrows = 7, 2
    else:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(imgs[i].reshape(width, height), norm=plt.Normalize(0, 1), cmap="gray")
        color = _split_color(labels[i])
        for _, spine in ax.spines.items():
            spine.set_visible(True); spine.set_color(color); spine.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(facecolor="none", edgecolor=_split_color(l), linewidth=2,
                    label=f"{' + '.join(_SPLIT_LABELS.get(s, s) for s in l.split('+'))} "
                          f"({label_counts[l]})")
              for l in sorted(set(labels), key=_split_sort_key)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=18,
              bbox_to_anchor=(0.5, 1.0 + 0.4 / nrows), frameon=False)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, "BAS_images")
    return fig


def plot_jgb_raw_data(N_qubits=12, N_features=3, train_split=0.5, val_split=0.25,
                      plots_dir: str = "plots", save: bool = True, quantizer: str = "minmax"):
    """Raw JGB yield series, shaded by the chronological train/val/test split (see
    DataLoader._split_jgb): train = earliest block, val = next, test = most recent."""
    jgb = JGB(N_qubits, N_features, quantizer)
    df0 = jgb.raw
    colors = categorical_colors(len(df0.columns))
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for i, c in enumerate(df0.columns):
        ax.plot(df0[c], label=f"{c[:-1]}-year Rate", color=colors[i])

    # boundaries mirror _split_jgb exactly, but computed on jgb.decimal (the day-over-day diff),
    # which drops the first raw row -- so decimal index i sits at raw index i+1; using N = len(decimal)
    # as the position into df0's index accounts for that +1 offset automatically.
    N = len(jgb.decimal)
    N_train = int(N * train_split)
    N_val = int(N * val_split)
    split_counts = {"train": N_train, "val": N_val, "test": N - N_train - N_val}
    bounds = [0, N_train, N_train + N_val, N]
    for start, end, split in zip(bounds[:-1], bounds[1:], _SPLIT_ORDER):
        ax.axvspan(df0.index[start], df0.index[end], color=_split_color(split), alpha=0.12, lw=0)

    legend_box_style = dict(frameon=True, fancybox=True, framealpha=0.9, edgecolor="0.3",
                            facecolor="white")
    # both legends sit outside the axes (to the right), stacked -- the JGB series can peak near
    # either top corner depending on the date range, so no in-plot corner is reliably clear, and
    # keeping the two legends together (rather than one in/one out) reads as one consistent design
    line_legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
                            **legend_box_style)
    ax.add_artist(line_legend)
    split_handles = [Patch(facecolor=_split_color(s), alpha=0.3,
                          label=f"{s} ({100 * split_counts[s] / N:.0f}%)") for s in _SPLIT_ORDER]
    split_legend = ax.legend(handles=split_handles, loc="upper left", bbox_to_anchor=(1.02, 0.55),
                             fontsize=7, title="Split", **legend_box_style)

    ax.set_xlabel("Date"); ax.set_ylabel("Interest Rate [%]")
    plt.tight_layout()
    if save:
        # both legends sit outside the axes -- see line_legend/split_legend above -- so savefig's
        # tight-bbox pass needs both listed explicitly (see _save) or the orphaned line_legend gets
        # clipped at the figure's right edge instead of padded like split_legend (the axes' current
        # legend, auto-included).
        _save(fig, plots_dir, "JGB_raw_data", extra_artists=[line_legend, split_legend])
    return fig


def plot_jgb_binary_histograms(N_qubits=12, N_features=3, plots_dir: str = "plots",
                               save: bool = True, quantizer: str = "minmax"):
    from collections import Counter
    from qiskit.visualization import plot_histogram
    jgb = JGB(N_qubits, N_features, quantizer)
    bits_per_feature = N_qubits // N_features
    target_dict = Counter(array_to_str(jgb.binary))
    feat_dicts = get_features_for_quasi_dist(target_dict, bits_per_feature, N_features)
    # derive from the columns JGB actually selected, so this stays correct for any feature subset
    labels = [f"{c[:-1]}-year" for c in jgb.raw.columns]
    colors = categorical_colors(N_features)
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
def plot_threshold_curve(curves: dict, selected_rule: str = "knee", selected_metric: str = None,
                         plots_dir: str = "plots", filename: str = "threshold_curve", save: bool = True):
    """Connections-vs-threshold curves for the metric_based extension, one subplot per distance
    metric (e.g. hamming and varinfo), so both are visible side by side regardless of which metric
    the dataset config actually uses. `curves` maps metric name -> (thresholds, counts, knee,
    percolation). Both the knee and bond-percolation thresholds are always marked in each subplot;
    the one actually selected by cfg.circuit.threshold_rule is highlighted. The subplot for
    `selected_metric` (the metric the dataset config actually uses) is marked in its title."""
    metrics = list(curves.keys())
    fig, axs = plt.subplots(1, len(metrics), figsize=(3 * len(metrics), 2), squeeze=False)
    axs = axs[0]
    for ax, metric in zip(axs, metrics):
        thresholds, counts, knee, percolation = curves[metric]
        rule_thresholds = {"knee": knee, "percolation": percolation}
        ax.plot(thresholds, counts, color=OKABE_ITO["blue"])
        for rule, color in (("knee", OKABE_ITO["vermillion"]), ("percolation", OKABE_ITO["orange"])):
            threshold = rule_thresholds[rule]
            threshold_idx = int(np.argmin(np.abs(thresholds - threshold)))
            threshold_count = int(counts[threshold_idx])
            is_selected = rule == selected_rule
            label = f"{rule} @ {threshold:.3f} ({threshold_count} conn.)"
            if is_selected:
                label += " [selected]"
            ax.axvline(threshold, color=color, ls="--" if is_selected else ":",
                      lw=1.2 if is_selected else 1, label=label)
        # let matplotlib place the legend clear of the data instead of hand-picking annotation
        # offsets, which overlap whenever the two thresholds happen to land close together
        ax.legend(fontsize=6, loc="best", frameon=True, framealpha=0.85)
        title = _METRIC_LABELS.get(metric, metric)
        if metric == selected_metric:
            title += " [selected]"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Threshold"); ax.set_ylabel("Number of Connections")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


_THRESHOLD_RULE_COLORS = {"knee": OKABE_ITO["vermillion"], "percolation": OKABE_ITO["orange"]}


def plot_selected_threshold_curve(curves: dict, selected_rule: str, selected_metric: str,
                                  plots_dir: str = "plots", filename: str = "threshold_curve_selected",
                                  save: bool = True):
    """Connections-vs-threshold curve for ONLY the metric/rule combination the dataset config
    actually uses (cfg.circuit.extension_metric / cfg.circuit.threshold_rule) -- a single-subplot,
    single-marker companion to plot_threshold_curve's full comparison-across-metrics-and-rules
    figure, for callers that just want "what did this run actually select" without the other
    metric/rule shown alongside it. `curves` is the same {metric: (thresholds, counts, knee,
    percolation)} mapping generate_extension_figures already builds for plot_threshold_curve."""
    thresholds, counts, knee, percolation = curves[selected_metric]
    value = knee if selected_rule == "knee" else percolation
    color = _THRESHOLD_RULE_COLORS[selected_rule]
    threshold_idx = int(np.argmin(np.abs(thresholds - value)))
    threshold_count = int(counts[threshold_idx])

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    ax.plot(thresholds, counts, color=OKABE_ITO["blue"])
    ax.axvline(value, color=color, ls="--", lw=1.2,
              label=f"{selected_rule} @ {value:.3f} ({threshold_count} conn.)")
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.85)
    ax.set_title(_METRIC_LABELS.get(selected_metric, selected_metric), fontsize=9)
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
    sns.heatmap(distmat, cmap=SEQUENTIAL_CMAP, ax=axs[0], vmin=0.0, vmax=1.0)
    axs[0].set_title(f"a) {metric_label}")
    sns.heatmap(dist_filter, cmap=SEQUENTIAL_CMAP, ax=axs[1])
    axs[1].set_title(f"b) Circuit Extension ({rule}={threshold:.3f})")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


# each extension (non-linear) edge becomes one add_su4_gate call (src.extension) -- a generic
# two-qubit unitary with exactly 15 real parameters (dim U(4) mod global phase), always, regardless
# of data. The linear/MPS-derived edges are NOT SU4 gates -- they come from mps2circuit's
# TwoQubitBasisDecomposer(CXGate(), euler_basis='U'), whose per-bond gate/parameter count depends on
# how entangled that specific trained MPS bond is (verified against real training logs: anywhere from
# ~2.4 CX/bond up to the generic max of 3). That count isn't knowable without actually training the
# MPS on this data, which this training-independent module deliberately never does -- so only the
# extension's added parameters (exact and data-independent) are reported here, not a circuit total.
SU4_PARAMS_PER_EDGE = 15


def _draw_topology(ax, n_qubits, edges_base, edges_ext, title):
    # nodes sit around the circle in index order (0, 1, ... n-1) in every panel, so a node's place on
    # the page is fixed and only the edges differ between panels -- for BAS 3x3 that means the MPS
    # chain's permuted hops (2-5, 5-4, 4-3, 3-6) are visible as chords rather than hidden by moving
    # the nodes to make the chain a neat ring.
    edges_new = sorted(set(edges_ext) - set(edges_base))
    G = nx.Graph(); G.add_nodes_from(range(n_qubits))
    G.add_edges_from(edges_base); G.add_edges_from(edges_ext)
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="white", edgecolors="black", node_size=200, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_base, edge_color="black", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_new, edge_color=OKABE_ITO["blue"], ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
    ax.set_title(f"{title}\n(+{len(edges_new) * SU4_PARAMS_PER_EDGE} params)")
    ax.set_aspect("equal"); ax.set_frame_on(False)
    ax.set_xticks([]); ax.set_yticks([])


def _node_labels(cfg) -> list:
    """The index labels these panels draw, as a qubit -> label map.

    PLOTTING ONLY -- nothing here feeds a circuit; the simulation's qubit indices and connections are
    untouched. The labels are the dataset's own feature indices (for BAS: grid-pixel positions), which
    is what a reader of these figures interprets the node numbers as. The circuit runs on qubits that
    DataLoader.reorder_features has permuted those features onto -- qubit k carries feature
    init_qubit_order_bas[k] -- so the map below is exactly that permutation, and identity for every
    config reorder_features leaves alone (JGB, and BAS grids with no entry).
    """
    if cfg.data.dataset == "BAS":
        return list(init_qubit_order_bas.get(f"{cfg.data.width}x{cfg.data.height}",
                                             range(cfg.data.N_qubits)))
    return list(range(cfg.data.N_qubits))


def _to_labels(connections: list, labels: list) -> list:
    """Rewrite qubit-indexed connections onto the panels' node labels (see _node_labels)."""
    return sorted(set(tuple(sorted((int(labels[i]), int(labels[j])))) for i, j in connections))


def plot_topology_panel(cfg, X_train: np.ndarray, distmat: np.ndarray, threshold: float,
                        rule: str = "knee", plots_dir: str = "plots", save: bool = True):
    """Topology networks for the plotted extension methods, drawn on the dataset's own feature indices
    (BAS: grid-pixel positions -- see _node_labels), showing exactly the graphs
    setup.setup_circuit_extensions builds. The nearest-neighbor extension is deliberately NOT plotted
    (it is excluded from every figure -- see plotting.EXCLUDED_EXTENSIONS); setup/extension still
    implement it, so a run can be configured with it, it just no longer appears in the figure set.

    The MPS chain is setup's linear_topology(range(N_qubits)) over data reorder_features has already
    permuted, so on these labels it is the snake path 0-1-2-5-4-3-6-7-8 for BAS 3x3, not 0-...-8: the
    same physical circuit, read on the grid. EVERY extension graph is relabelled the same way, by
    _to_labels -- whatever setup hands to extend_circuit is a list of QUBIT pairs. Relabelling chain
    and extension together is a bijection, so each panel's NEW-edge count is exactly the "New
    connections" its run logs: 4 metric-based / 5 Chow-Liu / 28 all-to-all for BAS 3x3 (verified
    against run gqr78zjr)."""
    dataset = cfg.data.dataset
    n_qubits = cfg.data.N_qubits
    labels = _node_labels(cfg)
    edges_lin = linear_topology(labels)
    edges_metric = _to_labels(metric_based_topology(distmat, threshold), labels)
    edges_tree = _to_labels(chow_liu_topology(mutual_info_matrix(np.asarray(X_train))), labels)

    panels = {
        "a) Linear": edges_lin,
        f"b) Metric-Based ({rule}={threshold:.2f})": edges_metric,
        "c) Chow-Liu": edges_tree,
        "d) All-to-All": all_to_all_topology(n_qubits),
    }

    fig, axs = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
    for ax, (name, edges) in zip(np.atleast_1d(axs), panels.items()):
        _draw_topology(ax, n_qubits, edges_lin, edges, name)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, f"{dataset}_topology")
    return fig


def _connection_levels(thresholds: np.ndarray, counts: np.ndarray) -> list:
    """Collapse the (thresholds, counts) step function from connection_threshold_curve into
    (count, threshold_lo, threshold_hi) runs -- one per distinct connection count, spanning the
    threshold interval over which that count holds."""
    levels = []
    start = 0
    for i in range(1, len(counts) + 1):
        if i == len(counts) or counts[i] != counts[start]:
            levels.append((int(counts[start]), float(thresholds[start]), float(thresholds[i - 1])))
            start = i
    return levels


# --------------------------------------------------------------------------------------------------
# per-dataset driver
# --------------------------------------------------------------------------------------------------
def generate_extension_figures(cfg, plots_dir: str = "plots") -> float:
    """Generate every dataset-only figure for one dataset config: preprocessing, threshold curves
    (both the full comparison-across-metrics-and-rules figure and the selected-only companion),
    extension heatmap, and topology networks, all sharing one auto-selected threshold (rule per
    cfg.circuit.threshold_rule). Returns the selected threshold."""
    dataset = cfg.data.dataset
    rule = cfg.circuit.threshold_rule
    folder = os.path.join(plots_dir, dataset_folder(cfg))

    dl = setup_dataloader(cfg)
    X_train, *_ = compute_split(cfg, dl)  # exact reordered train split setup.py trains the circuit on

    distmat = feature_distance_matrix(X_train, cfg.circuit.extension_metric)
    threshold = select_threshold(distmat, rule)

    # threshold curves are computed for every distance metric (not just the one the dataset config
    # actually uses) so plot_threshold_curve can show them side by side for comparison
    curves = {}
    for metric in _METRIC_LABELS:
        metric_distmat = distmat if metric == cfg.circuit.extension_metric else \
            feature_distance_matrix(X_train, metric)
        metric_thresholds, metric_counts = connection_threshold_curve(metric_distmat)
        curves[metric] = (metric_thresholds, metric_counts, knee_threshold(metric_distmat),
                          percolation_threshold(metric_distmat))

    thresholds, counts, *_ = curves[cfg.circuit.extension_metric]
    threshold_idx = int(np.argmin(np.abs(thresholds - threshold)))
    print(f"[plot_extension] {dataset_folder(cfg)}: {rule} threshold = {threshold:.4f} "
         f"({int(counts[threshold_idx])} connections) -> {folder}/")

    # connection counts for every knee/percolation marker actually drawn on the threshold curve
    # figure (both metrics, both rules), so the printed numbers always match what's plotted
    for metric, (metric_thresholds, metric_counts, knee, percolation) in curves.items():
        active = " [active]" if metric == cfg.circuit.extension_metric else ""
        for label, value in (("knee", knee), ("percolation", percolation)):
            idx = int(np.argmin(np.abs(metric_thresholds - value)))
            print(f"[plot_extension]   {metric}{active} {label} @ {value:.4f} "
                 f"({int(metric_counts[idx])} connections)")
        print(f"[plot_extension]   {metric}{active} connection levels:")
        for count, lo, hi in _connection_levels(metric_thresholds, metric_counts):
            print(f"[plot_extension]     {count} connections for threshold in [{lo:.4f}, {hi:.4f}]")

    if dataset == "BAS":
        plot_bas_images(cfg.data.width, cfg.data.height, cfg.data.train_split, cfg.data.val_split,
                        seed=cfg.sweep.initial_random_seed, bas_split_mode=cfg.data.bas_split_mode,
                        plots_dir=folder)
    else:
        plot_jgb_raw_data(cfg.data.N_qubits, cfg.data.N_features, cfg.data.train_split,
                          cfg.data.val_split, plots_dir=folder, quantizer=cfg.data.quantizer)
        plot_jgb_binary_histograms(cfg.data.N_qubits, cfg.data.N_features, plots_dir=folder,
                                   quantizer=cfg.data.quantizer)

    plot_threshold_curve(curves, selected_rule=rule, selected_metric=cfg.circuit.extension_metric,
                        plots_dir=folder)
    plot_selected_threshold_curve(curves, selected_rule=rule, selected_metric=cfg.circuit.extension_metric,
                                  plots_dir=folder)
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
