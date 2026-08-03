"""Static, training-independent figures investigating the thresholding rule and circuit extension.

Everything here depends only on the dataset + Hydra config, never on a trained/wandb run, so it only
needs regenerating when the data or extension settings change (unlike src.plotting). The SU(4) gate
diagram doesn't depend on the dataset at all, so it goes to a shared `general/` folder.

The metric_based threshold is auto-selected per cfg.circuit.threshold_rule (extension.select_threshold),
computed here with the same helpers setup.setup_circuit_extensions uses and marked on the threshold
curve, heatmap and topology network, so what's plotted matches what a real run would build.

    uv run python -m src.plot_extension                      # both datasets
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
from src.plotting import (_save, use_science_style, OKABE_ITO, SEQUENTIAL_CMAP, categorical_colors,
                          feature_label)

_METRIC_LABELS = {"hamming": "Hamming distance", "varinfo": "Variation of Information"}

# knee/percolation are RULES, not extensions, so they keep their own colors rather than borrowing
# plotting._ROLE_COLORS (which is keyed on extension names)
_THRESHOLD_RULE_COLORS = {"knee": OKABE_ITO["vermillion"], "percolation": OKABE_ITO["orange"]}

# shared train/val/test color coding for the BAS-image and JGB-raw-data split figures
_SPLIT_ORDER = ["train", "val", "test"]
_SPLIT_BASE_COLORS = {"train": OKABE_ITO["blue"], "val": OKABE_ITO["orange"], "test": OKABE_ITO["bluish_green"]}
_SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
_SPLIT_ALPHA = 0.15   # one value for both the plotted shading and its legend swatch


def _split_color(label: str) -> str:
    """Color for a split label; combined labels (e.g. "train+val+test", from BAS full_support mode,
    where every sample is in all three splits) get a neutral color of their own."""
    return _SPLIT_BASE_COLORS.get(label, OKABE_ITO["black"])


def _split_sort_key(label: str) -> int:
    return _SPLIT_ORDER.index(label) if label in _SPLIT_ORDER else len(_SPLIT_ORDER)


# --------------------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------------------
def load_cfg(dataset: str):
    """Compose one dataset's Hydra config exactly as a real training run would see it."""
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
# dataset preprocessing figures (unreordered raw data -- these are about the dataset itself, not the
# circuit's qubit graph, so they must NOT go through the qubit-reordering used for training)
# --------------------------------------------------------------------------------------------------
def _bas_split_labels(bas: BAS, train_split: float, val_split: float, seed: int,
                      bas_split_mode: str) -> list:
    """Per-pattern split membership ("train"/"val"/"test", or a combined "train+val+test" under
    bas_split_mode="full_support" where every pattern is in every split).

    Delegates to DataLoader.train_val_test_split (reorder=False, so rows stay in bas.binary's order)
    rather than re-deriving the seeded permutation, so this can't drift from src.data. Patterns are
    unique 0/1 vectors, so membership is recovered by matching each row against the split arrays.
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
    """All enumerated BAS patterns, bordered by split membership (see _bas_split_labels)."""
    from collections import Counter
    bas = BAS(width, height)
    imgs = bas.binary
    labels = _bas_split_labels(bas, train_split, val_split, seed, bas_split_mode)
    label_counts = Counter(labels)
    n = len(imgs)

    # wrap into a roughly square grid instead of one long row, which keeps stretching the figure wider
    # as the enumerated support grows. BAS 3x3 (n=14) gets a fixed 2x7 block.
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
        # (width, height): BAS.lines_stripes lays a pattern out as `width` blocks of `height` entries
        ax.imshow(imgs[i].reshape(width, height), norm=plt.Normalize(0, 1), cmap="gray")
        color = _split_color(labels[i])
        for _, spine in ax.spines.items():
            spine.set_visible(True); spine.set_color(color); spine.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(facecolor="none", edgecolor=_split_color(l), linewidth=2,
                    label=f"{' + '.join(_SPLIT_LABELS.get(s, s) for s in l.split('+'))} "
                          f"({label_counts[l]})")
              for l in sorted(set(labels), key=_split_sort_key)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8,
              bbox_to_anchor=(0.5, 1.0 + 0.22 / nrows), frameon=False)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, "BAS_images")
    return fig


def plot_jgb_raw_data(N_qubits=12, N_features=3, train_split=0.5, val_split=0.25,
                      plots_dir: str = "plots", save: bool = True, quantizer: str = "minmax"):
    """Raw JGB yield series, shaded by the chronological split (DataLoader._split_jgb): train =
    earliest block, val = next, test = most recent."""
    jgb = JGB(N_qubits, N_features, quantizer)
    df0 = jgb.raw
    colors = categorical_colors(len(df0.columns))
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for i, c in enumerate(df0.columns):
        ax.plot(df0[c], label=feature_label(c), color=colors[i])

    # boundaries mirror _split_jgb exactly, but computed on jgb.decimal (the day-over-day diff), which
    # drops the first raw row -- so decimal index i sits at raw index i+1; using N = len(decimal) as
    # the position into df0's index accounts for that +1 offset automatically
    N = len(jgb.decimal)
    N_train = int(N * train_split)
    N_val = int(N * val_split)
    split_counts = {"train": N_train, "val": N_val, "test": N - N_train - N_val}
    bounds = [0, N_train, N_train + N_val, N]
    for start, end, split in zip(bounds[:-1], bounds[1:], _SPLIT_ORDER):
        ax.axvspan(df0.index[start], df0.index[end], color=_split_color(split), alpha=_SPLIT_ALPHA,
                   lw=0)

    # one legend, inside the axes, lines and splits side by side (ncol=2 fills column-major -- lines
    # in column 1, splits in column 2, see matplotlib's legend guide) rather than two stacked boxes:
    # the only clear span on this axis is the top of the pre-2013 train era (never exceeds ~2.6%
    # here), a band tall enough for one 3-row legend but not two
    split_handles = [Patch(facecolor=_split_color(s), alpha=_SPLIT_ALPHA,
                          label=f"{_SPLIT_LABELS[s]} ({100 * split_counts[s] / N:.0f}%)")
                     for s in _SPLIT_ORDER]
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(line_handles + split_handles, line_labels + [h.get_label() for h in split_handles],
             ncol=2, loc="upper left", fontsize=9, frameon=True, fancybox=True, framealpha=0.9,
             edgecolor="0.3", facecolor="white")

    # larger than the suite's 8pt default: this figure stands alone (not one panel among several,
    # like most of the rest of the suite), so it can afford the room and reads better at print scale
    ax.set_xlabel("Date", fontsize=11); ax.set_ylabel("Interest Rate [%]", fontsize=11)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, "JGB_raw_data")
    return fig


def plot_jgb_binary_histograms(N_qubits=12, N_features=3, plots_dir: str = "plots",
                               save: bool = True, quantizer: str = "minmax"):
    """Per-feature quantized marginals of the encoded JGB data, one panel per feature."""
    from collections import Counter
    from qiskit.visualization import plot_histogram
    jgb = JGB(N_qubits, N_features, quantizer)
    bits_per_feature = jgb.bits_per_feature
    target_dict = Counter(array_to_str(jgb.binary))
    feat_dicts = get_features_for_quasi_dist(target_dict, bits_per_feature, N_features)
    # plot_histogram renders only the keys it is given, so pad every panel out to the full alphabet:
    # otherwise a feature missing a level shifts its bars and the stacked panels, which invite exactly
    # that comparison, are not on a common axis
    alphabet = [format(k, f"0{bits_per_feature}b") for k in range(2 ** bits_per_feature)]
    feat_dicts = [{k: d.get(k, 0.0) for k in alphabet} for d in feat_dicts]
    labels = [feature_label(c) for c in jgb.raw.columns]
    colors = categorical_colors(N_features)
    fig, axs = plt.subplots(N_features, 1, figsize=(6, 6), sharex=True)
    axs = np.atleast_1d(axs)
    for i, ax in enumerate(axs):
        plot_histogram(feat_dicts[i], ax=ax, bar_labels=False, color=colors[i])
        ax.set_title(labels[i], fontsize=9)
        ax.set_ylabel("Quasi-probability", fontsize=8)
        ax.tick_params(labelsize=6)
        if i < len(axs) - 1:
            ax.set_xticklabels([])
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
    """Connections-vs-threshold curves, one subplot per distance metric, so both are visible side by
    side regardless of which one the dataset config uses. `curves` maps metric -> (thresholds, counts,
    knee, percolation). Both rules are always marked; the one cfg.circuit.threshold_rule selects is
    highlighted, and `selected_metric` is marked in its subplot title."""
    metrics = list(curves.keys())
    fig, axs = plt.subplots(1, len(metrics), figsize=(3 * len(metrics), 2), squeeze=False)
    axs = axs[0]
    for ax, metric in zip(axs, metrics):
        thresholds, counts, knee, percolation = curves[metric]
        rule_thresholds = {"knee": knee, "percolation": percolation}
        ax.plot(thresholds, counts, color=OKABE_ITO["blue"])
        for rule, color in _THRESHOLD_RULE_COLORS.items():
            threshold = rule_thresholds[rule]
            threshold_idx = int(np.argmin(np.abs(thresholds - threshold)))
            threshold_count = int(counts[threshold_idx])
            is_selected = rule == selected_rule
            label = f"{rule} @ {threshold:.3f} ({threshold_count} conn.)"
            if is_selected:
                label += " [selected]"
            ax.axvline(threshold, color=color, ls="--" if is_selected else ":",
                      lw=1.2 if is_selected else 1, label=label)
        # let matplotlib place the legend clear of the data instead of hand-picking annotation offsets,
        # which overlap whenever the two thresholds land close together
        ax.legend(fontsize=6, loc="best", frameon=True, framealpha=0.85)
        title = _METRIC_LABELS.get(metric, metric)
        if metric == selected_metric:
            title += " [selected]"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Threshold")
    axs[0].set_ylabel("Number of Connections")   # shared scale, so label it once
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


def plot_selected_threshold_curve(curves: dict, selected_rule: str, selected_metric: str,
                                  plots_dir: str = "plots", filename: str = "threshold_curve_selected",
                                  save: bool = True):
    """Connections-vs-threshold curve for ONLY the metric/rule the dataset config actually uses -- a
    single-subplot companion to plot_threshold_curve's full comparison, for callers who just want
    "what did this run select". `curves` is the same mapping generate_extension_figures builds."""
    thresholds, counts, knee, percolation = curves[selected_metric]
    value = {"knee": knee, "percolation": percolation}[selected_rule]
    color = _THRESHOLD_RULE_COLORS[selected_rule]
    threshold_idx = int(np.argmin(np.abs(thresholds - value)))
    threshold_count = int(counts[threshold_idx])

    fig, ax = plt.subplots(1, 1, figsize=(3, 1.7))
    ax.plot(thresholds, counts, color=OKABE_ITO["blue"])
    ax.axvline(value, color=color, ls="--", lw=1.2,
              label=f"{selected_rule} @ {value:.3f} ({threshold_count} conn.)")
    ax.legend(fontsize=6, loc="best", frameon=True, framealpha=0.85)
    ax.set_xlabel("Threshold", fontsize=7); ax.set_ylabel("Connections", fontsize=7)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


def plot_extension_heatmap(distmat: np.ndarray, threshold: float, rule: str = "knee",
                           plots_dir: str = "plots", metric_label: str = "Distance",
                           filename: str = "extension_heatmap", save: bool = True):
    """Distance-matrix heatmap next to the thresholded circuit-extension mask. Both panels are pinned
    to 0..1 (the metrics' own range, and the mask is 0/1) so their colorbars are comparable."""
    import seaborn as sns
    dim = distmat.shape[0]
    dist_filter = np.zeros_like(distmat)
    dist_filter[distmat < threshold] = 1.0
    dist_filter -= np.eye(dim)
    fig, axs = plt.subplots(1, 2, figsize=(6, 2.6))
    sns.heatmap(distmat, cmap=SEQUENTIAL_CMAP, ax=axs[0], vmin=0.0, vmax=1.0)
    axs[0].set_title(f"a) {metric_label}")
    sns.heatmap(dist_filter, cmap=SEQUENTIAL_CMAP, ax=axs[1], vmin=0.0, vmax=1.0)
    axs[1].set_title(f"b) Circuit Extension ({rule}={threshold:.3f})")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


# node/label/title sizes for the topology panels. Larger than the rest of the suite's 7-9pt on
# purpose: these panels carry only a handful of glyphs on a 3-inch square, so at the shared size the
# qubit indices were unreadable at print scale. NODE_SIZE is an AREA in pt^2 and must track FONT_SIZE
# or a two-digit label (JGB, 12 qubits) overflows its circle.
_TOPO_FONT_SIZE = 9
_TOPO_TITLE_SIZE = 12
_TOPO_NODE_SIZE = 250
_TOPO_LINEWIDTH = 1.2   # node border + edges; thinner than the earlier 1.4, which read as too bold


def _draw_topology(ax, n_qubits, edges_base, edges_ext, title):
    # nodes sit around the circle in index order in every panel, so a node's place on the page is
    # fixed and only the edges differ -- for BAS 3x3 the MPS chain's permuted hops (2-5, 5-4, 4-3,
    # 3-6) are then visible as chords rather than hidden by moving the nodes to make a neat ring
    edges_new = sorted(set(edges_ext) - set(edges_base))
    G = nx.Graph(); G.add_nodes_from(range(n_qubits))
    G.add_edges_from(edges_base); G.add_edges_from(edges_ext)
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="white", edgecolors="black",
                           node_size=_TOPO_NODE_SIZE, linewidths=_TOPO_LINEWIDTH, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_base, edge_color="black",
                           width=_TOPO_LINEWIDTH, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges_new, edge_color=OKABE_ITO["blue"],
                           width=_TOPO_LINEWIDTH, ax=ax)
    # va="center_baseline", not the default "center": digit glyphs have no descender, so centering on
    # the font's full bounding box (which reserves descender space) sits every label visibly high --
    # center_baseline centers on the glyphs actually drawn instead
    nx.draw_networkx_labels(G, pos, font_size=_TOPO_FONT_SIZE, font_weight="bold",
                            verticalalignment="center_baseline", ax=ax)
    ax.set_title(title, fontsize=_TOPO_TITLE_SIZE)
    ax.set_aspect("equal"); ax.set_frame_on(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.margins(0.14)   # room for the enlarged nodes, which circular_layout doesn't account for


def _node_labels(cfg) -> list:
    """The index labels these panels draw, as a qubit -> label map.

    PLOTTING ONLY -- nothing here feeds a circuit. The labels are the dataset's own feature indices
    (BAS: grid-pixel positions), which is how a reader interprets the node numbers. Qubit k carries
    feature init_qubit_order_bas[k] after DataLoader.reorder_features, so this map is exactly that
    permutation, and identity wherever reorder_features is a no-op (JGB, BAS grids with no entry).
    """
    if cfg.data.dataset == "BAS":
        return list(init_qubit_order_bas.get(f"{cfg.data.width}x{cfg.data.height}",
                                             range(cfg.data.N_qubits)))
    return list(range(cfg.data.N_qubits))


def _to_labels(connections: list, labels: list) -> list:
    """Rewrite qubit-indexed connections onto the panels' node labels (see _node_labels)."""
    return sorted(set(tuple(sorted((int(labels[i]), int(labels[j])))) for i, j in connections))


def plot_topology_panel(cfg, X_train: np.ndarray, distmat: np.ndarray, threshold: float,
                        plots_dir: str = "plots", save: bool = True):
    """Topology networks for the plotted extension methods, on the dataset's own feature indices (see
    _node_labels), showing exactly the graphs setup.setup_circuit_extensions builds.

    Two extensions are deliberately absent. nearest_neighbor is excluded from every figure
    (plotting.EXCLUDED_EXTENSIONS). `random` is excluded because it is the only extension the config
    does NOT determine: setup seeds it per run, so a sweep has one different graph per seed and no
    single panel represents it -- the MMD curves and benchmark tables, which aggregate over all seeds,
    are where it belongs.

    The MPS chain is setup's linear_topology(range(N_qubits)) over the features reorder_features has
    already permuted, so on these labels it is the snake path 0-1-2-5-4-3-6-7-8 for BAS 3x3 -- the same
    circuit, read on the grid. Relabelling chain and extension together via _to_labels is a bijection,
    so each panel's NEW-edge count is exactly the "New connections" its run logs: 4 metric-based /
    5 chow-liu / 28 all-to-all for BAS 3x3 (verified against run gqr78zjr).
    """
    dataset = cfg.data.dataset
    n_qubits = cfg.data.N_qubits
    labels = _node_labels(cfg)
    edges_lin = linear_topology(labels)
    edges_metric = _to_labels(metric_based_topology(distmat, threshold), labels)
    edges_tree = _to_labels(chow_liu_topology(mutual_info_matrix(np.asarray(X_train))), labels)

    panels = {
        "a) Linear": edges_lin,
        "b) Metric-Based": edges_metric,
        "c) Chow-Liu": edges_tree,
        "d) All-to-All": all_to_all_topology(n_qubits),
    }

    fig, axs = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.4))
    for ax, (name, edges) in zip(np.atleast_1d(axs), panels.items()):
        _draw_topology(ax, n_qubits, edges_lin, edges, name)
    plt.tight_layout(w_pad=0.3)
    if save:
        _save(fig, plots_dir, f"{dataset}_topology")
    return fig


def _connection_levels(thresholds: np.ndarray, counts: np.ndarray) -> list:
    """Collapse the (thresholds, counts) step function into (count, threshold_lo, threshold_hi) runs --
    one per distinct connection count, spanning the interval over which it holds."""
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
    """Every dataset-only figure for one dataset config: preprocessing, threshold curves (the full
    comparison and the selected-only companion), heatmap, topology networks -- all sharing one
    auto-selected threshold (rule per cfg.circuit.threshold_rule). Returns that threshold."""
    dataset = cfg.data.dataset
    rule = cfg.circuit.threshold_rule
    metric = cfg.circuit.extension_metric
    folder = os.path.join(plots_dir, dataset_folder(cfg))

    dl = setup_dataloader(cfg)
    X_train, *_ = compute_split(cfg, dl)  # exact reordered train split setup.py trains the circuit on

    distmat = feature_distance_matrix(X_train, metric)
    threshold = select_threshold(distmat, rule)

    # curves for every known metric plus the config's own (even if it isn't in _METRIC_LABELS), so
    # plot_threshold_curve can show them side by side and curves[metric] always exists
    curves = {}
    for m in dict.fromkeys([*_METRIC_LABELS, metric]):
        m_distmat = distmat if m == metric else feature_distance_matrix(X_train, m)
        m_thresholds, m_counts = connection_threshold_curve(m_distmat)
        curves[m] = (m_thresholds, m_counts, knee_threshold(m_distmat),
                     percolation_threshold(m_distmat))

    thresholds, counts, *_ = curves[metric]
    threshold_idx = int(np.argmin(np.abs(thresholds - threshold)))
    print(f"[plot_extension] {dataset_folder(cfg)}: {rule} threshold = {threshold:.4f} "
         f"({int(counts[threshold_idx])} connections) -> {folder}/")

    # connection counts for every marker actually drawn on the threshold-curve figure, so the printed
    # numbers always match what's plotted
    for m, (m_thresholds, m_counts, knee, percolation) in curves.items():
        active = " [active]" if m == metric else ""
        for label, value in (("knee", knee), ("percolation", percolation)):
            idx = int(np.argmin(np.abs(m_thresholds - value)))
            print(f"[plot_extension]   {m}{active} {label} @ {value:.4f} "
                 f"({int(m_counts[idx])} connections)")
        print(f"[plot_extension]   {m}{active} connection levels:")
        for count, lo, hi in _connection_levels(m_thresholds, m_counts):
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

    plot_threshold_curve(curves, selected_rule=rule, selected_metric=metric, plots_dir=folder)
    plot_selected_threshold_curve(curves, selected_rule=rule, selected_metric=metric,
                                  plots_dir=folder)
    plot_extension_heatmap(distmat, threshold, rule=rule, plots_dir=folder,
                           metric_label=_METRIC_LABELS.get(metric, metric))
    plot_topology_panel(cfg, X_train, distmat, threshold, plots_dir=folder)

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
