"""wandb-driven plotting for QCBM sweeps.

Training-dependent figures, built by pulling run histories from a wandb sweep (grouped by any config
dimension), with MMD plotted against cumulative circuit *measurements* rather than iteration index.
Saved as PDF + PNG. Static dataset/topology/threshold figures live in src.plot_extension.

    from src.plotting import generate_all_figures
    generate_all_figures(sweep_id="<sweep>", entity="<you>", project="qcbm-circuit-design")

The dataset is read off the runs' own configs (detect_dataset), not passed in: it names the output
directory and gates the dataset-specific figures, so a caller default could mislabel a figure set.
"""

import os
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from src.data import JGB, DataLoader
from src.utils import bootstrap_mean_std
from src import benchmark as bm


# --------------------------------------------------------------------------------------------------
# styling
# --------------------------------------------------------------------------------------------------
def use_science_style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass


# canonical extension legend labels
_EXTENSION_LABELS = {
    "none": "Linear",
    "linear": "Linear",
    "random": "Random",
    "metric_based": "Metric-Based",
    "chow_liu": "Chow-Liu",
    "all_to_all": "All-to-All",
    "nearest_neighbor": "Nearest-Neighbor",   # excluded below, so normally never plotted
}

# extensions kept out of EVERY figure: fetch_runs drops their runs, so no curve, table or QQ panel is
# built for them, and plot_extension omits their topology panel. A PLOTTING decision only --
# src.extension/src.setup still implement them and a run can still be configured with one.
EXCLUDED_EXTENSIONS = {"nearest_neighbor"}

# label of the shared pre-training reference line (see _draw_mmd_curves)
_BASELINE_LABEL = "MPS baseline"

# Okabe & Ito (2008) colorblind-safe categorical palette (Wong, Nature Methods 8, 441, 2011).
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# One fixed color AND dash pattern per extension, keyed on the canonical label, so an extension looks
# the same in EVERY figure regardless of which/how-many others are present. Both maps must stay
# exhaustive over _EXTENSION_LABELS: the fallbacks assign by position, so a partially-pinned map lets
# a missing group silently shift everything unpinned.
_ROLE_COLORS = {
    "linear": OKABE_ITO["blue"],
    "chow-liu": OKABE_ITO["bluish_green"],
    "metric-based": OKABE_ITO["orange"],
    "random": OKABE_ITO["vermillion"],
    "all-to-all": OKABE_ITO["reddish_purple"],
    "nearest-neighbor": OKABE_ITO["sky_blue"],
}
# Pinned explicitly because the scienceplots "ieee" style cycles linestyle alongside color in a
# 4-entry prop_cycle: an unspecified ls is taken from the axes cycler, i.e. from DRAW ORDER, so
# dropping one group re-dashed every later one and a 5-group figure reused a pattern.
_ROLE_LINESTYLES = {
    "linear": "-",                                   # solid
    "metric-based": (0, (4, 1.5)),                   # dash
    "chow-liu": (0, (4, 1.5, 1, 1.5)),               # dash-dot
    "random": (0, (1, 1.5)),                         # dotted
    "all-to-all": (0, (7, 1.5)),                     # long dash
    "nearest-neighbor": (0, (4, 1.5, 1, 1.5, 1, 1.5)),   # dash-dot-dot
}
_BASELINE_LINESTYLE = (0, (1, 4))                    # sparse dotted, distinct from `random`

# fallbacks for categorical keys that are NOT a known extension (a newly added one, or a sweep grouped
# by some other config key). Yellow is last: it is a fill color, not a line color -- as _CB_CYCLE[1]
# it rendered the second JGB series near-invisible on white.
_CB_CYCLE = [OKABE_ITO["sky_blue"], OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"],
             OKABE_ITO["reddish_purple"], OKABE_ITO["orange"], OKABE_ITO["blue"],
             OKABE_ITO["yellow"]]
_LS_CYCLE = list(_ROLE_LINESTYLES.values())

# shared sequential colormap for all continuous/2-D fields: cividis is perceptually uniform AND
# colorblind-safe, so it pairs with the categorical palette above. Imported by src.plot_extension.
SEQUENTIAL_CMAP = "cividis"


def categorical_colors(n: int) -> list:
    """`n` distinct colorblind-safe colors for series that aren't keyed on an extension (e.g. the
    per-feature JGB lines/histograms in src.plot_extension)."""
    return [_CB_CYCLE[i % len(_CB_CYCLE)] for i in range(n)]


def feature_label(column: str) -> str:
    """A JGB bond-tenor column ("5Y") as a figure label, shared by every JGB figure so the raw-data,
    histogram and QQ figures name a feature identically."""
    return f"{column[:-1]}-year Rate"


def _numeric_order(keys):
    """`keys` sorted numerically, or None if they aren't all numbers (a numeric sweep dimension gets
    the sequential colormap ordered by value instead of categorical styling)."""
    try:
        return sorted(keys, key=lambda k: float(k))
    except (TypeError, ValueError):
        return None


def _by_role(keys, roles: dict, cycle: list, default):
    """Pin every known extension to its `roles` entry; assign the rest from `cycle` in sorted order,
    skipping values a role in this same call already claimed. Order-independent either way, so the
    mapping for a key never depends on which other keys are present."""
    pinned = {k: roles[_EXTENSION_LABELS.get(k, k)] for k in keys
              if _EXTENSION_LABELS.get(k, k) in roles}
    spare = [c for c in cycle if c not in set(pinned.values())] or cycle
    for i, k in enumerate(sorted((k for k in keys if k not in pinned), key=str)):
        pinned[k] = spare[i % len(spare)]
    return pinned


def default_colors(legend_keys) -> dict:
    """{key: color}: pinned _ROLE_COLORS per extension, _CB_CYCLE for unknown keys, SEQUENTIAL_CMAP
    for a numeric sweep dimension."""
    keys = list(legend_keys)
    numeric = _numeric_order(keys)
    if numeric is not None:
        shades = plt.get_cmap(SEQUENTIAL_CMAP)(np.linspace(0.15, 0.9, len(numeric)))
        return {k: shades[i] for i, k in enumerate(numeric)}
    return _by_role(keys, _ROLE_COLORS, _CB_CYCLE, "-")


def default_linestyles(legend_keys) -> dict:
    """{key: linestyle}, the dash-pattern counterpart of default_colors. All solid for a numeric
    sweep, where the colormap already encodes the ordering."""
    keys = list(legend_keys)
    if _numeric_order(keys) is not None:
        return {k: "-" for k in keys}
    return _by_role(keys, _ROLE_LINESTYLES, _LS_CYCLE, "-")


def _save(fig, plots_dir, filename, extra_artists=None):
    """`extra_artists` is savefig's bbox_extra_artists: the tight-bbox pass auto-discovers figure
    legends and the axes' CURRENT legend, but not one added via ax.add_artist (see
    plot_extension.plot_jgb_raw_data), which then gets clipped instead of padded."""
    path = f"{plots_dir}/{filename}"
    os.makedirs(os.path.dirname(path), exist_ok=True)  # filename may itself contain "/"
    fig.savefig(f"{path}.pdf", bbox_inches="tight", bbox_extra_artists=extra_artists, transparent=True)
    fig.savefig(f"{path}.png", bbox_inches="tight", bbox_extra_artists=extra_artists, transparent=False,
               dpi=300)


# --------------------------------------------------------------------------------------------------
# wandb fetch + aggregation
# --------------------------------------------------------------------------------------------------
def fetch_runs(sweep_id: str, entity: str, project: str, group_by: str = "circuit.extension",
               filters: dict = None) -> dict:
    """{legend_key: [runs]} for a wandb Sweep, grouped by a config dimension.

    group_by/filters keys are dot-separated config paths. `filters` pins non-legend swept params
    (facet slicing), applied client-side since Sweep.runs is a materialized list; the sweep query and
    config parsing go through benchmark's caches, so re-grouping costs no extra request.

    Runs whose config predates the current schema are skipped with a warning, and runs of an
    EXCLUDED_EXTENSIONS extension are dropped regardless of `group_by`.
    """
    runs = []
    for r in bm.sweep_runs(sweep_id, entity, project):
        try:
            runs.append((r, bm.run_config(r)))
        except Exception as e:
            print(f"[plotting]     skipping run {r.id}: config incompatible with current schema ({e})")
    n_before = len(runs)
    runs = [(r, cfg) for r, cfg in runs
            if OmegaConf.select(cfg, "circuit.extension") not in EXCLUDED_EXTENSIONS]
    if len(runs) < n_before:
        print(f"[plotting]     excluding {n_before - len(runs)} run(s) of "
              f"{', '.join(sorted(EXCLUDED_EXTENSIONS))} (not plotted)")
    if filters:
        runs = [(r, cfg) for r, cfg in runs
                if all(OmegaConf.select(cfg, k) == v for k, v in filters.items())]
    grouped = {}
    for r, cfg in runs:
        grouped.setdefault(OmegaConf.select(cfg, group_by), []).append(r)
    return grouped


def detect_dataset(runs_by_key: dict, expected: str = None) -> str:
    """The dataset kind ("BAS"/"JGB") a sweep's runs were actually trained on, per their configs.

    No metric depends on it -- each run is scored against its own dataset -- so a wrong kind mislabels
    the output directory rather than corrupting a number. `expected` is a cross-check only: on
    disagreement the runs win. Raises ValueError if no config could be read at all.
    """
    kinds = {}
    for runs in runs_by_key.values():
        for r in runs:
            try:
                kinds.setdefault(bm.run_config(r).data.dataset, []).append(r.id)
            except Exception:
                continue  # already reported by fetch_runs
    if not kinds:
        raise ValueError("could not read the dataset kind from any run config of this sweep")
    dataset = max(kinds, key=lambda k: len(kinds[k]))
    if len(kinds) > 1:
        counts = ", ".join(f"{k}: {len(v)}" for k, v in sorted(kinds.items()))
        print(f"[plotting]     WARNING: sweep mixes dataset kinds ({counts}); labelling as the "
              f"majority kind {dataset}. The metric tables mix both -- plot the sweeps separately.")
    if expected is not None and expected != dataset:
        print(f"[plotting]     WARNING: requested dataset {expected} disagrees with the sweep's runs "
              f"({dataset}); using {dataset}.")
    return dataset


# run.id -> (frozenset of metrics the cached request asked for, raw history dataframe)
_HISTORY_CACHE = {}

# rows per history fetch, well above wandb's default of 500: cumulative_measurements is logged EVERY
# iteration and a measurement-budget run can exceed 500, at which point the sampled-history endpoint
# silently subsamples and the curve is drawn from a thinned, unevenly spaced subset.
_HISTORY_SAMPLES = 10_000


def _raw_history(run, metrics: tuple) -> pd.DataFrame:
    """Cached raw history for `run` covering ALL of `metrics` in ONE wandb request, so a plotting pass
    costs one history fetch per run however many metrics consume it. A later request for metrics
    outside the cached set refetches the union.

    Safe to batch because train/mmd_* and the bench_*/ suite are logged on the same eval steps (see
    QCBM.stochastic_gradient_descent, one payload), so no metric loses rows by sharing a request.
    """
    want = frozenset(metrics)
    cached = _HISTORY_CACHE.get(run.id)
    if cached is not None and want <= cached[0]:
        return cached[1]
    if cached is not None:
        want = want | cached[0]
    df = run.history(keys=["train/cumulative_measurements", *sorted(want)], pandas=True,
                     samples=_HISTORY_SAMPLES)
    if df is None:
        df = pd.DataFrame()
    _HISTORY_CACHE[run.id] = (want, df)
    return df


def prefetch_histories(runs_by_key: dict, metrics) -> None:
    """Warm the cache with one request per run covering every metric to be plotted, so the per-figure
    fetch_history calls are pure cache hits. Purely an optimization."""
    metrics = tuple(metrics)
    for runs in runs_by_key.values():
        for r in runs:
            _raw_history(r, metrics)


def reset_history_cache() -> None:
    """Drop cached run histories (use when re-plotting a sweep that is still running)."""
    _HISTORY_CACHE.clear()


def fetch_history(run, metric: str = "train/mmd_train") -> pd.DataFrame:
    """Per-run dataframe with train/cumulative_measurements and `metric` (NaN rows dropped), read off
    the shared cache. Empty (but correctly shaped) when the run never logged `metric`."""
    df = _raw_history(run, (metric,))
    if df.empty or metric not in df:
        return pd.DataFrame(columns=["train/cumulative_measurements", metric])
    return df.dropna(subset=[metric]).sort_values("train/cumulative_measurements")


def aggregate_over_measurements(histories: list, metric: str, mode: str = "bootstrap",
                                n_grid: int = 400, window: int = 1, n_boot: int = 1000,
                                boot_seed: int = 0, log_x: bool = False) -> dict:
    """Aggregate the runs of one group into a mean curve + spread band across seeds.

    Seed-runs normally share an IDENTICAL measurement axis, so the aggregation happens at those native
    positions with no interpolation and no resampling. `n_grid` only matters for the fallback below.

    `mode`: "bootstrap" (default) resamples the runs with replacement n_boot times per x -- `line` is
    the mean of the bootstrap means, the band +/- their std (the across-seed SE). "meanstd" is a plain
    mean +/- std; "medperc" the median with a 10/90 percentile band.

    Fallback: runs that genuinely disagree on x (one stopped early) are interpolated onto a shared
    grid over the window where all have data, log-spaced when `log_x`. A last resort -- it distorts a
    log-x plot both ways: MMD is logged every qcbm.eval_every iterations, so the first gap spans ~1
    decade and np.interp renders it as a plateau-then-knee artifact, while at the high end a log grid
    collapses several real samples into one cell.

    `log_x` also drops the step-0 row logged at cumulative_measurements=0: log(0) is undefined and
    padding down to a positive floor spends decades of width on a flat line carrying one number.
    Callers draw that value as a horizontal reference line instead (_draw_mmd_curves).
    """
    curves = [h for h in histories if len(h) > 0]
    if not curves:
        return None
    if log_x:
        curves = [c[c["train/cumulative_measurements"] > 0] for c in curves]  # drop x=0 (see above)
        curves = [c for c in curves if len(c) > 0]
        if not curves:
            return None

    x_ref = curves[0]["train/cumulative_measurements"].values
    aligned = all(len(c) == len(x_ref)
                  and np.array_equal(c["train/cumulative_measurements"].values, x_ref)
                  for c in curves[1:])
    if aligned and len(x_ref) > 1 and np.isfinite(x_ref).all():
        grid = x_ref
        stacked = np.vstack([c[metric].values for c in curves])
    else:
        # the shared window is where EVERY run has data, so no curve is extrapolated
        lo = max(c["train/cumulative_measurements"].min() for c in curves)
        hi = min(c["train/cumulative_measurements"].max() for c in curves)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        grid = np.geomspace(lo, hi, n_grid) if log_x else np.linspace(lo, hi, n_grid)
        stacked = np.vstack([
            np.interp(grid, c["train/cumulative_measurements"].values, c[metric].values)
            for c in curves
        ])
    if mode == "bootstrap":
        line, std = bootstrap_mean_std(stacked, n_boot=n_boot, seed=boot_seed)  # across-run SE
        lower, upper = line - std, line + std
    elif mode == "meanstd":
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
    return {"x": grid, "line": line, "lower": lower, "upper": upper, "n_runs": len(curves)}


def _draw_mmd_curves(ax, runs_by_key: dict, metric: str, mode: str, window: int, n_boot: int,
                     colors: dict, log_axes: bool, linestyles: dict = None) -> None:
    """One aggregated curve + bootstrap-SE band per group, plus the pre-training baseline as a
    horizontal reference line. Shared by plot_mmd_vs_measurements and plot_train_val_mmd.

    The curves span only real measurements (see aggregate_over_measurements), so step 0 becomes that
    line rather than a point on them. One line, neutral grey: step 0 is the SHARED unextended circuit
    at the sweep's fixed initial_random_seed, bit-identical across every run and connectivity
    (QCBM._log_baseline_step), so it belongs to the sweep and not to any group.
    """
    linestyles = linestyles if linestyles is not None else default_linestyles(runs_by_key.keys())
    baselines = []
    for key, runs in runs_by_key.items():
        hists = [fetch_history(r, metric) for r in runs]
        for h in hists:
            at_zero = h.loc[h["train/cumulative_measurements"] == 0, metric]
            if len(at_zero):
                baselines.append(float(at_zero.iloc[0]))
        agg = aggregate_over_measurements(hists, metric, mode, window=window, n_boot=n_boot,
                                          log_x=log_axes)
        if agg is None:
            continue
        label = _EXTENSION_LABELS.get(key, str(key))
        ax.plot(agg["x"], agg["line"], label=label, color=colors.get(key),
                ls=linestyles.get(key, "-"))
        ax.fill_between(agg["x"], agg["lower"], agg["upper"], alpha=0.3, lw=0.0, color=colors.get(key))
    # runs predating the baseline-logging change have no step-0 row -> no line rather than a guess
    if baselines:
        ax.axhline(float(np.mean(baselines)), ls=_BASELINE_LINESTYLE, lw=0.9, color="0.4", zorder=0,
                   label=_BASELINE_LABEL)
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")


def _group_param_count(runs) -> float:
    """A group's parameter count, for the legend/series ordering. A group predating parameter logging
    sorts last instead of crashing the comparison."""
    counts = [r.summary.get("train/num_parameters") for r in runs]
    counts = [c for c in counts if c is not None]
    return counts[0] if counts else np.inf


def _ordered_legend(ax, axes, runs_by_key: dict, **kwargs):
    """One legend on `ax` merging the handles of every axes in `axes` (so a group with data in only
    one panel still appears), ordered by parameter count.

    Ascending in what the extension ADDS, matching how the benchmark tables are sorted; the MPS
    baseline has none of the added SU(4) gates so it always leads. Shared by every MMD figure --
    hand-rolling it per figure had the standalone and side-by-side legends in different orders.
    """
    handles_by_label = {}
    for a in axes:
        for handle, label in zip(*a.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)
    if not handles_by_label:
        return None
    params = {}
    for key, runs in runs_by_key.items():
        params.setdefault(_EXTENSION_LABELS.get(key, str(key)), _group_param_count(runs))
    order = sorted(handles_by_label,
                   key=lambda l: (-1, 0) if l == _BASELINE_LABEL else (0, params.get(l, np.inf)))
    return ax.legend([handles_by_label[l] for l in order], order, **kwargs)


def plot_mmd_vs_measurements(runs_by_key: dict, metric: str = "train/mmd_train", mode: str = "bootstrap",
                             window: int = 1, n_boot: int = 1000, colors: dict = None,
                             filename: str = "MMD_measurements", plots_dir: str = "plots",
                             save: bool = True, log_axes: bool = False, title: str = None,
                             ylabel: str = None):
    """MMD (or any logged metric) vs cumulative measurements, aggregated across seeds per group.

    Defaults to the bootstrap aggregation; the band is the bootstrap std of the mean. `log_axes`
    plots log-log, which drops the step-0 row and shows it as a reference line instead (see
    aggregate_over_measurements). `title`/`ylabel` override the metric-name-derived defaults.
    """
    colors = colors or default_colors(runs_by_key.keys())
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    _draw_mmd_curves(ax, runs_by_key, metric, mode, window, n_boot, colors, log_axes)
    ax.set_xlabel("Measurements")
    ax.set_ylabel(ylabel or metric.replace("_", " ").upper())
    if title:
        ax.set_title(title)
    _ordered_legend(ax, [ax], runs_by_key, loc="best", fontsize=7, frameon=True)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


def plot_train_val_mmd(runs_by_key: dict, mode: str = "bootstrap", window: int = 1, n_boot: int = 1000,
                       colors: dict = None, filename: str = "train_val_mmd_measurements",
                       plots_dir: str = "plots", save: bool = True, log_axes: bool = True):
    """Train and val MMD vs cumulative measurements as two side-by-side panels sharing one legend.

    Each panel keeps its own y-axis: train and val MMD can differ enough in scale that one shared
    range would flatten one curve. Aggregation and log-axis handling are identical to
    plot_mmd_vs_measurements (same _draw_mmd_curves helper) per panel.
    """
    colors = colors or default_colors(runs_by_key.keys())
    linestyles = default_linestyles(runs_by_key.keys())
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    for ax, metric, split in ((axs[0], "train/mmd_train", "Training"),
                              (axs[1], "train/mmd_val", "Validation")):
        _draw_mmd_curves(ax, runs_by_key, metric, mode, window, n_boot, colors, log_axes,
                         linestyles=linestyles)
        ax.set_xlabel("Measurements")
        ax.set_title(split)
        ax.set_ylabel("MMD")
    _ordered_legend(axs[0], axs, runs_by_key, loc="best", fontsize=7, frameon=True)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


# --------------------------------------------------------------------------------------------------
# benchmark tables (bootstrap across all runs per group)
# --------------------------------------------------------------------------------------------------
# setup/cost facts (benchmark._run_setup_facts): deterministic per group -- same architecture and
# budget for every seed -- so they get a plain mean, no bootstrap SE, and lead the column order.
_SETUP_COLUMNS = ("num_parameters", "n_connections", "total_measurements", "iterations_run")


def bootstrap_group_metrics(per_run_df: pd.DataFrame, group_by: str = "circuit.extension",
                            metric_cols: list = None, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Bootstrap held-out benchmark metrics across the runs of each group.

    `per_run_df` is benchmark.benchmark_all_runs' tidy one-row-per-run table. Per group and metric,
    resample the runs with replacement n_boot times and report the mean and the std of the bootstrap
    means (the across-seed SE); non-finite per-run values are dropped first. Returns one row per group
    with columns <m> and <m>_std plus n_runs -- except _SETUP_COLUMNS, which are constant per group so
    they get a plain mean, no `_std`, and lead the order.
    """
    if per_run_df is None or per_run_df.empty:
        return None
    if metric_cols is None:
        skip = {group_by, "run_id", "run_name"}
        numeric_cols = [c for c in per_run_df.columns
                       if c not in skip and np.issubdtype(per_run_df[c].dropna().dtype, np.number)]
        setup_cols = [c for c in _SETUP_COLUMNS if c in numeric_cols]
        metric_cols = setup_cols + [c for c in numeric_cols if c not in setup_cols]
    rows = []
    for key, sub in per_run_df.groupby(group_by, sort=False):
        row = {group_by: key, "n_runs": len(sub)}
        for c in metric_cols:
            vals = sub[c].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if c in _SETUP_COLUMNS:
                row[c] = float(vals.mean()) if vals.size else np.nan
            elif vals.size == 0:
                row[c] = row[f"{c}_std"] = np.nan
            else:
                mean, std = bootstrap_mean_std(vals, n_boot=n_boot, seed=seed)
                row[c], row[f"{c}_std"] = float(mean), float(std)
        rows.append(row)
    return pd.DataFrame(rows)


def save_metric_table(bench_df: pd.DataFrame, plots_dir: str = "plots",
                      filename: str = "benchmark_table", save: bool = True) -> pd.DataFrame:
    """Write the full benchmark table (every group/metric/_std/n_runs column) as CSV. Works for both
    the bootstrap table and the point-estimate one. No figure: a full-precision CSV is for downstream
    analysis, not at-a-glance reading."""
    if bench_df is None or bench_df.empty:
        return None
    if save:
        os.makedirs(plots_dir, exist_ok=True)
        bench_df.to_csv(f"{plots_dir}/{filename}.csv", index=False)
    return bench_df


# --------------------------------------------------------------------------------------------------
# rendered benchmark tables (the CSV above stays the source of truth; these are readable cuts of it)
# --------------------------------------------------------------------------------------------------
# Which direction is "better" per metric, for the bold-best marking: distances/divergences/NLLs are
# minimized, scores/fidelities/coverages/rates maximized. A column absent from this map is never
# bolded. bench_dist/* is scored against the FULL dataset (train+val+test), see benchmark.evaluate.
_LOWER_IS_BETTER = {
    "selection_metric": True,
    "bench_dist/mmd": True,
    "bench_dist/kl": True,
    "bench_dist/tv": True,
    "bench_dist/nll": True,
    "bench_dist/fidelity": False,
    "bench_val/coverage": False,
    "bench_val/fidelity": False,
    "bench_val/rate": False,
    "bench_val/exploration": False,
    "bench_BAS/precision": False,
    "bench_BAS/recall": False,
    "bench_BAS/qbas": False,
}

# columns rendered as plain counts (no ± std, no decimals)
_COUNT_COLUMNS = {"n_runs", "num_parameters", "n_connections", "total_measurements", "iterations_run"}

# (filename, title, show_std, ((column, header), ...)) per rendered table. The setup table shows no
# ± std and marks no "best": its columns are experiment DESIGN facts, identical across a group's seeds
# (std 0), and bolding a best there would flag the fewest-parameter linear circuit as the winner.
_TABLE_SPECS = (
    ("benchmark_table_setup", "Setup", False,
     (("n_runs", "runs"),
      ("num_parameters", "params"),
      ("n_connections", "added conn."),
      ("total_measurements", "measurements"),
      ("iterations_run", "iterations"))),
    ("benchmark_table_validity", "Validity metrics", True,
     (("bench_val/coverage", "coverage"),
      ("bench_val/fidelity", "fidelity"),
      ("bench_val/rate", "rate"),
      ("bench_val/exploration", "exploration"))),
    ("benchmark_table_distribution", "Distribution distance metrics", True,
     (("bench_dist/mmd", "MMD"),
      ("bench_dist/kl", "KL"),
      ("bench_dist/tv", "TV"),
      ("bench_dist/nll", "NLL"),
      ("bench_dist/fidelity", "fidelity"))),
    ("benchmark_table_bas", "BAS metrics", True,
     (("bench_BAS/precision", "precision"),
      ("bench_BAS/recall", "recall"),
      ("bench_BAS/qbas", "qBAS"))),
)


def _fmt_count(value) -> str:
    """A count cell: thousands-separated integer, em dash when missing."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{int(round(value)):,}"


def _fmt_metric(mean, std, precision: int, show_std: bool) -> str:
    """A metric cell: "mean ± std" when a finite companion std is present and wanted, else the bare
    mean; em dash for missing/NaN."""
    if mean is None or not np.isfinite(mean):
        return "—"
    if show_std and std is not None and np.isfinite(std):
        return f"{mean:.{precision}f} ± {std:.{precision}f}"
    return f"{mean:.{precision}f}"


def _best_row_per_column(bench_df: pd.DataFrame, columns) -> dict:
    """{column index: row index of the best value}, per each column's _LOWER_IS_BETTER direction.
    Columns with no defined direction, and all-NaN columns, are absent (nothing gets bolded)."""
    best = {}
    for j, (col, _label) in enumerate(columns):
        if col not in _LOWER_IS_BETTER:
            continue
        vals = pd.to_numeric(bench_df[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            continue
        best[j] = int(np.nanargmin(vals) if _LOWER_IS_BETTER[col] else np.nanargmax(vals))
    return best


def plot_benchmark_tables(bench_df: pd.DataFrame, group_by: str = "circuit.extension",
                          plots_dir: str = "plots", save: bool = True, precision: int = 4) -> dict:
    """Render the benchmark table as readable per-family figures (see _TABLE_SPECS).

    Each metric table shows "mean ± bootstrap SE" and bolds the best value per column, per that
    metric's direction. Tables whose columns are all absent are skipped (e.g. BAS on a JGB sweep).
    Rows are sorted by parameter count, so groups line up across all four tables and with the MMD
    legends. Returns {filename: figure}.
    """
    if bench_df is None or bench_df.empty:
        return {}
    if "num_parameters" in bench_df.columns:
        bench_df = bench_df.sort_values("num_parameters", na_position="last").reset_index(drop=True)
    figs = {}
    for filename, title, show_std, columns in _TABLE_SPECS:
        columns = [(c, lab) for c, lab in columns if c in bench_df]
        if not columns:
            continue
        fig = _render_benchmark_table(bench_df, group_by, columns, show_std, title, filename,
                                     plots_dir, save, precision)
        if fig is not None:
            figs[filename] = fig
    return figs


def _render_benchmark_table(bench_df, group_by, columns, show_std, title, filename, plots_dir,
                            save, precision):
    """Render one benchmark table figure; see plot_benchmark_tables."""
    row_labels = [_EXTENSION_LABELS.get(k, str(k)) for k in bench_df[group_by]]
    cell_text = [[_fmt_count(row.get(col, np.nan)) if col in _COUNT_COLUMNS
                  else _fmt_metric(row.get(col, np.nan), row.get(f"{col}_std", np.nan),
                                   precision, show_std)
                  for col, _label in columns]
                 for _, row in bench_df.iterrows()]
    best = _best_row_per_column(bench_df, columns)

    n_rows, n_cols = len(cell_text), len(columns)
    col_width = 1.55 if show_std else 1.15  # "mean ± std" cells need more room than counts
    # Height the figure to the table, not the reverse: matplotlib sizes rows as a FRACTION of the axes
    # (~fontsize/72 * 1.2, then scaled), so a taller figure leaves the table floating in whitespace.
    # Solving that relation for the height fills the canvas at any row count.
    font_size, y_scale = 8, 1.5
    row_height = font_size / 72 * 1.2 * y_scale
    fig, ax = plt.subplots(figsize=(col_width * (n_cols + 1) + 0.8,
                                    (n_rows + 1) * row_height + 0.55))  # + title/margin
    ax.axis("off")
    table = ax.table(cellText=cell_text, rowLabels=row_labels,
                     colLabels=[label for _col, label in columns], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, y_scale)
    for j, i in best.items():
        table[(i + 1, j)].get_text().set_fontweight("bold")  # +1: row 0 holds the column headers
    ax.set_title(title, pad=12)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


# --------------------------------------------------------------------------------------------------
# JGB QQ figures
# --------------------------------------------------------------------------------------------------
# Shared data/encoding context for a sweep's QQ figures: every run trains on the same dataset, so the
# marginals (and the axes they set) are built once from the first run's config. A run whose ENCODING
# differs would be read against a reference it never saw, so callers compare `encoding` and drop
# mismatches rather than silently rescaling.
_JGBRef = namedtuple("_JGBRef", "encoding data n_features bits_per_feature quantizer feature_names")

# larger than the suite's 7-8pt default: these are read closely (marginal shape, floor separation),
# and each panel has room to spare at 3in square
_QQ_TITLE_SIZE = 12
_QQ_LABEL_SIZE = 11
_QQ_TICK_SIZE = 9
_QQ_LEGEND_SIZE = 9


def _jgb_reference(cfg) -> _JGBRef:
    jgb = JGB(cfg.data.N_qubits, cfg.data.N_features, cfg.data.quantizer)
    dl = DataLoader(jgb)
    dl.train_val_test_split(cfg.data.train_split, cfg.data.val_split)  # train-fitted quantizer
    return _JGBRef((cfg.data.N_qubits, cfg.data.N_features, cfg.data.quantizer),
                   jgb.decimal.values, cfg.data.N_features, jgb.bits_per_feature, dl.quantizer,
                   list(jgb.raw.columns))  # bond tenors, e.g. "5Y"/"10Y", in feature order


def _run_qq_quantiles(run, ref: _JGBRef, n_shots: int, which: str, n_q: int) -> list:
    """One run's per-feature model quantiles on the shared data-quantile grid (x is the same for every
    run and reference curve, so only the y-values are returned)."""
    cfg = bm.run_config(run)
    circuit, params = bm.load_checkpoint(run, which=which)
    samples = bm.sample_model(circuit, params, n_shots, seed=cfg.sweep.random_seed)
    feats = bm.reconstruct_features(samples, ref.bits_per_feature, ref.n_features,
                                    quantizer=ref.quantizer)
    return [bm.qq_model_vs_data(*feats[i], ref.data[:, i], n_q)[1] for i in range(ref.n_features)]


def _draw_qq_references(ax, ref: _JGBRef, i: int, n_q: int) -> list:
    """One panel's two reference curves, on the same lattice and estimator as the model curves: the
    floor -- the data round-tripped through the encoding, so zero model error -- and a Gaussian
    baseline. Diagonal-to-floor is quantization, floor-to-model is model error. The Gaussian is NOT a
    floor: a good model can beat it. Returns the arrays so the caller can include them in the range.
    """
    fdx, fdy = bm.qq_quantized_data_vs_data(ref.quantizer, i, ref.data[:, i], n_q)
    ax.plot(fdx, fdy, "--", lw=1.1, color=OKABE_ITO["black"], zorder=4, label="Quantization Floor")
    gdx, gdy = bm.qq_quantized_gaussian_vs_data(ref.quantizer, i, ref.data[:, i], n_q)
    ax.plot(gdx, gdy, ":", lw=1.1, color="0.45", zorder=3, label="Quantized Gaussian")
    return [fdx, fdy, gdx, gdy]


def _finish_qq_panel(ax, span: list, title: str) -> None:
    """Square equal-aspect axes spanning every plotted series, plus the y=x guide."""
    lo = min(float(np.min(a)) for a in span)
    hi = max(float(np.max(a)) for a in span)
    pad = 0.03 * (hi - lo)
    ax.plot([lo, hi], [lo, hi], "-", lw=0.6, color="0.6", zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")   # a QQ plot is only read against y=x, which must appear at 45 degrees
    ax.set_title(title, fontsize=_QQ_TITLE_SIZE)
    ax.set_xlabel("Data Quantile", fontsize=_QQ_LABEL_SIZE)
    ax.tick_params(labelsize=_QQ_TICK_SIZE)


def _qq_legend(fig, axs, save: bool, plots_dir: str, filename: str):
    """One shared legend below the panels: with a series per group plus the references it no longer
    fits inside a panel without covering the curves it describes."""
    handles, labels = axs[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                        ncol=min(len(labels), 7), fontsize=_QQ_LEGEND_SIZE, frameon=False)
    axs[0].set_ylabel("Model Quantile", fontsize=_QQ_LABEL_SIZE)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename, extra_artists=(legend,))
    return legend


def plot_qq_vs_data(sweep_id: str, entity: str, project: str, group_by: str = "circuit.extension",
                    n_shots: int = 10000, n_q: int = 100, plots_dir: str = "plots", save: bool = True,
                    runs_by_key: dict = None, which: str = "final",
                    select_metric: str = "train/mmd_train", filename: str = "JGB_QQ"):
    """One JGB QQ figure comparing EVERY group's model against the data, one panel per feature.

    All series share one reference -- the empirical data quantiles on x -- so groups are comparable
    and y=x reads as "matches the data". Two more curves share the panel (see _draw_qq_references):
    the quantization floor, which no model can beat, and a quantized-Gaussian baseline, which a good
    model can.

    One model per group: the run with the lowest final-iteration `select_metric`, with which="final"
    so the checkpoint plotted is the one that value was measured on. train/mmd_train is deliberate --
    best_mmd_val lands at initialization for most runs (best_iter == 1 for 63 of 80 on the sweep this
    was built against) and final-iteration validation barely discriminates (~1.03x between group
    medians vs ~1.13x between seeds), while training MMD separates by ~10x. Never pass a test-split
    metric: selecting by it leaks the split the figure is read against.

    So this picks a REPRESENTATIVE model for judging marginal shape, not evidence of generalization;
    the bench_dist/* tables remain the quantitative claim and across-seed spread is not drawn here
    (see plot_qq_bootstrap_vs_data). Returns the figure, or None if no group had a usable checkpoint.
    """
    grouped = runs_by_key if runs_by_key is not None else fetch_runs(sweep_id, entity, project, group_by)
    colors = default_colors(list(grouped))
    # the panels can only be built once the shared data axis is known, so sampling comes first
    series, ref = [], None
    for key, runs in grouped.items():
        best = bm.select_best_run(runs, metric=select_metric)
        if best is None:
            continue
        cfg = bm.run_config(best)
        if cfg.data.dataset != "JGB":
            continue
        if ref is None:
            ref = _jgb_reference(cfg)
        elif (cfg.data.N_qubits, cfg.data.N_features, cfg.data.quantizer) != ref.encoding:
            print(f"[plotting]     [{key}] skipped: JGB encoding differs from {ref.encoding}; "
                  f"not comparable on shared axes.")
            continue
        print(f"[plotting]     [{key}] QQ series from run {best.id} "
              f"({select_metric}={best.summary.get(select_metric)}, which={which})...")
        series.append((key, _group_param_count(runs),
                       _run_qq_quantiles(best, ref, n_shots, which, n_q)))
    if not series:
        return None
    series.sort(key=lambda s: s[1])

    fig, axs = plt.subplots(1, ref.n_features, figsize=(3 * ref.n_features, 3), squeeze=False)
    axs = axs[0]
    for i, ax in enumerate(axs):
        dx = np.quantile(ref.data[:, i], np.linspace(0.01, 0.99, n_q))
        span = [dx]
        for key, _count, quantiles in series:
            ax.plot(dx, quantiles[i], "-", lw=1, color=colors[key],
                    label=_EXTENSION_LABELS.get(key, str(key)))
            span.append(quantiles[i])
        span += _draw_qq_references(ax, ref, i, n_q)
        _finish_qq_panel(ax, span, feature_label(ref.feature_names[i]))
    _qq_legend(fig, axs, save, plots_dir, filename)
    return fig


def plot_qq_bootstrap_vs_data(sweep_id: str, entity: str, project: str,
                              group_by: str = "circuit.extension", n_shots: int = 10000,
                              n_q: int = 100, n_boot: int = 1000, plots_dir: str = "plots",
                              save: bool = True, runs_by_key: dict = None, which: str = "final",
                              residual_row: bool = True, filename: str = "JGB_QQ_bootstrap"):
    """The QQ-vs-data figure with EVERY trained model of each group, bootstrapped across seeds.

    Same panels, references and reading as plot_qq_vs_data, but each group is a band: every seed-run
    is sampled and the seeds are bootstrapped per quantile level into a mean +/- across-seed SE. This
    drops the single-model selection problem entirely and answers what one model cannot -- whether a
    group's departure from the quantization floor is resolved above seed-to-seed scatter.

    `residual_row` adds a second row plotting each band MINUS the floor. On the raw QQ axes both the
    effect and the band are a few percent of the plotted range (the SE ~0.1%, thinner than the line
    over it), so the top row alone cannot show whether bands separate; the residual row rescales y to
    the effect, where zero is the floor, vertical gaps are extension-attributable and band thickness
    is seed noise.

    Caveats: the band is POINTWISE per quantile level, not a simultaneous confidence region; and it
    covers seed variability only -- shot noise is not resampled, being <1% of a bin at these n_shots.
    Costs one checkpoint load + sampling per RUN. Returns None if no group had a usable checkpoint.
    """
    grouped = runs_by_key if runs_by_key is not None else fetch_runs(sweep_id, entity, project, group_by)
    colors = default_colors(list(grouped))
    # (group_key, param_count, per-feature (mean, se), n_runs); as above, the shared data axis has to
    # exist before any curve can be placed on it, so all sampling happens first
    series, ref = [], None
    for key, runs in grouped.items():
        cfgs = [(r, bm.run_config(r)) for r in runs]
        cfgs = [(r, c) for r, c in cfgs if c.data.dataset == "JGB"]
        if not cfgs:
            continue
        if ref is None:
            ref = _jgb_reference(cfgs[0][1])
        cfgs = [(r, c) for r, c in cfgs
                if (c.data.N_qubits, c.data.N_features, c.data.quantizer) == ref.encoding]
        if not cfgs:
            print(f"[plotting]     [{key}] skipped: JGB encoding differs from {ref.encoding}; "
                  f"not comparable on shared axes.")
            continue
        print(f"[plotting]     [{key}] QQ band from {len(cfgs)} run(s) (which={which})...")
        per_run = []
        for run, _cfg in cfgs:
            try:
                per_run.append(_run_qq_quantiles(run, ref, n_shots, which, n_q))
            except Exception as e:   # one unusable checkpoint must not sink the group
                print(f"[plotting]         skipping run {run.id}: {e!r}")
        if not per_run:
            print(f"[plotting]     [{key}] no usable checkpoint, skipping group.")
            continue
        # (n_runs, n_q) per feature -> pointwise bootstrap mean and across-seed SE
        bands = []
        for i in range(ref.n_features):
            stacked = np.array([curves[i] for curves in per_run], dtype=float)
            bands.append(bootstrap_mean_std(stacked, n_boot=n_boot, seed=0))
        series.append((key, _group_param_count(runs), bands, len(per_run)))
    if not series:
        return None
    series.sort(key=lambda s: s[1])
    print(f"[plotting]     bootstrapping {n_boot} resamples per quantile from "
          f"{', '.join(f'{k}: {n}' for k, _c, _b, n in series)} run(s)")

    n_rows = 2 if residual_row else 1
    # the residual row is not equal-aspect (its y is a difference) so it needs less height than the
    # square QQ panels above it
    fig, grid = plt.subplots(n_rows, ref.n_features, squeeze=False,
                             figsize=(3 * ref.n_features, 3 + 2.1 * (n_rows - 1)),
                             gridspec_kw={"height_ratios": [3, 2][:n_rows]})
    axs = grid[0]
    for i, ax in enumerate(axs):
        dx = np.quantile(ref.data[:, i], np.linspace(0.01, 0.99, n_q))
        span = [dx]
        for key, _count, bands, _n in series:
            mean, se = bands[i]
            ax.fill_between(dx, mean - se, mean + se, color=colors[key], alpha=0.25, lw=0, zorder=2)
            ax.plot(dx, mean, "-", lw=1, color=colors[key], zorder=2,
                    label=_EXTENSION_LABELS.get(key, str(key)))
            span += [mean - se, mean + se]
        span += _draw_qq_references(ax, ref, i, n_q)
        _finish_qq_panel(ax, span, feature_label(ref.feature_names[i]))
        if not residual_row:
            continue
        rax = grid[1][i]
        _, floor = bm.qq_quantized_data_vs_data(ref.quantizer, i, ref.data[:, i], n_q)
        for key, _count, bands, _n in series:
            mean, se = bands[i]
            rax.fill_between(dx, mean - se - floor, mean + se - floor, color=colors[key],
                             alpha=0.25, lw=0, zorder=2)
            rax.plot(dx, mean - floor, "-", lw=1, color=colors[key], zorder=2)
        _, gauss = bm.qq_quantized_gaussian_vs_data(ref.quantizer, i, ref.data[:, i], n_q)
        rax.plot(dx, gauss - floor, ":", lw=1.1, color="0.45", zorder=3)
        rax.axhline(0.0, ls="--", lw=1.1, color=OKABE_ITO["black"], zorder=4)
        rax.set_xlim(*ax.get_xlim())
        rax.set_xlabel("Data Quantile", fontsize=_QQ_LABEL_SIZE)
        rax.tick_params(labelsize=_QQ_TICK_SIZE)
    if residual_row:
        grid[1][0].set_ylabel("Model $-$ Floor", fontsize=_QQ_LABEL_SIZE)
        for ax in axs:      # the shared x-axis is labelled on the residual row instead
            ax.set_xlabel("")
    _qq_legend(fig, axs, save, plots_dir, filename)
    return fig


# --------------------------------------------------------------------------------------------------
# threshold-sweep figures (opt-in: for a sweep where circuit.threshold itself was varied, a
# fundamentally different sweep shape from the standard per-extension pipeline above)
# --------------------------------------------------------------------------------------------------
# y-axis label per benchmark metric, one figure per metric. Key order is generation order; a metric in
# bench_df but missing here still gets a figure, labelled with its raw column name.
_BENCH_METRIC_LABELS = {
    "selection_metric": "seed-selection metric (see benchmark.select_best_run)",
    "bench_dist/mmd": "MMD",
    "bench_dist/kl": "KL divergence",
    "bench_dist/tv": "total variation",
    "bench_dist/nll": "negative log-likelihood",
    "bench_dist/fidelity": "classical fidelity",
    "bench_val/coverage": "coverage",
    "bench_val/fidelity": "validity fidelity",
    "bench_val/rate": "rate",
    "bench_val/exploration": "exploration",
    "bench_BAS/precision": "precision",
    "bench_BAS/recall": "recall",
    "bench_BAS/qbas": "qBAS",
}

# {threshold: (label, color, marker)} for the two sweep ends, which coincide with named topologies:
# threshold=0 admits no pair (dist < 0 never holds) so it is the plain linear MPS circuit, threshold=1
# admits every pair and reproduces all-to-all (verified for BAS 4x3: 0 and 55 added connections). NB
# these are the ENDS of a plateau, so neighbouring points can be the same circuit. Colors are from
# _ROLE_COLORS, so "linear" here is the same blue as everywhere else.
_THRESHOLD_ENDPOINTS = {
    0.0: ("linear @ 0.000", _ROLE_COLORS["linear"], "s"),
    1.0: ("all-to-all @ 1.000", _ROLE_COLORS["all-to-all"], "D"),
}


# Metrics the combined grid shows by default, and its panel geometry. Sized for a two-column paper's
# full-width float (a figure*'s \textwidth is ~7in), so the PDF is included at ~1:1 and its fonts land
# on the page at the size they were authored -- scaling four 4x3in single-metric figures down into a
# 2x2 subfigure grid shrank them instead.
_GRID_METRICS = ("bench_dist/mmd", "bench_dist/kl", "bench_val/coverage", "bench_BAS/qbas")
_GRID_FIG_W = 6.9      # total width, ~a two-column figure*'s \textwidth, so it is included at 1:1
_GRID_PANEL_H = 1.6    # per row, excluding the shared legend strip
_GRID_TITLE_SIZE = 9
_GRID_LABEL_SIZE = 9
_GRID_TICK_SIZE = 8
_GRID_LEGEND_SIZE = 8


def _threshold_reference_value(any_run):
    """The (rule, threshold) to mark on a threshold-sweep figure: whichever value
    cfg.circuit.threshold_rule would auto-select -- not both knee and percolation, since a run only
    ever uses one rule.

    Computed from ONE run's dataset/extension_metric (constant across a threshold sweep) via the same
    helpers setup uses. An annotation, not data: an unreconstructable config yields None with a
    warning so the figures are still produced, unmarked.
    """
    from src.setup import setup_dataloader, compute_split
    from src.utils import feature_distance_matrix
    from src.extension import knee_threshold, percolation_threshold

    try:
        cfg = bm.run_config(any_run)
        rule = cfg.circuit.threshold_rule
        X_train, *_ = compute_split(cfg, setup_dataloader(cfg))
        distmat = feature_distance_matrix(X_train, cfg.circuit.extension_metric)
        value = knee_threshold(distmat) if rule == "knee" else percolation_threshold(distmat)
        return rule, value
    except Exception as e:
        print(f"[plotting]     no threshold-rule reference line: {e!r}")
        return None


def plot_metrics_vs_threshold(sweep_id: str, entity: str, project: str,
                              group_by: str = "circuit.threshold", n_shots: int = 10000,
                              n_boot: int = 1000, which: str = "final", metrics=None,
                              plots_dir: str = "plots", save: bool = True,
                              science_style: bool = True, grid_metrics=_GRID_METRICS) -> tuple:
    """One figure per benchmark metric vs the metric_based threshold, for a sweep where
    `circuit.threshold` (not circuit.extension) was the swept dimension -- e.g. to see how an
    auto-selected knee/percolation threshold compares to a hand-swept range.

    Every metric bench_df carries gets a figure (mean +/- bootstrap SE per threshold, with the
    config-selected rule's value marked); `metrics` restricts that set. The sweep is fetched,
    benchmarked and bootstrapped ONCE and reused for every figure; absent or all-NaN metrics are
    skipped. Applies the science style itself (like generate_all_figures) so a library call matches
    the rest of the suite. Saves to plots_dir/<sweep_id>-threshold/. Returns (bench_df, figs).
    """
    if science_style:
        use_science_style()
    plots_dir = os.path.join(plots_dir, f"{sweep_id}-threshold")
    grouped = fetch_runs(sweep_id, entity, project, group_by)
    if not grouped:
        print("[plotting]     no runs found.")
        return None, {}

    per_run = bm.benchmark_all_runs(sweep_id, entity, project, group_by, n_shots=n_shots,
                                    which=which, groups=grouped)
    bench_df = bootstrap_group_metrics(per_run, group_by=group_by, n_boot=n_boot)
    if bench_df is None or bench_df.empty:
        print("[plotting]     no benchmark metrics -- no run had a usable checkpoint?")
        return None, {}

    # the swept dimension is numeric here; a run with threshold=None (auto-selected rather than
    # pinned) can't be placed on the axis, so it is dropped with a warning rather than silently
    bench_df = bench_df.copy()
    bench_df[group_by] = pd.to_numeric(bench_df[group_by], errors="coerce")
    n_dropped = int(bench_df[group_by].isna().sum())
    if n_dropped:
        print(f"[plotting]     dropping {n_dropped} group(s) with a non-numeric {group_by}")
        bench_df = bench_df[bench_df[group_by].notna()]
    bench_df = bench_df.sort_values(group_by).reset_index(drop=True)
    if bench_df.empty:
        print(f"[plotting]     no groups left with a numeric {group_by}.")
        return None, {}

    if metrics is None:
        metrics = [m for m in _BENCH_METRIC_LABELS if m in bench_df]
        # anything benchmarked but not in the label map still gets a figure, under its raw name --
        # every numeric outcome column, excluding the SEs and the setup/cost counts
        metrics += [c for c in bench_df.columns
                    if c not in _BENCH_METRIC_LABELS and not c.endswith("_std")
                    and c not in _COUNT_COLUMNS and c != group_by
                    and pd.api.types.is_numeric_dtype(bench_df[c])]
    reference = _threshold_reference_value(next(iter(grouped.values()))[0])

    figs = {}
    for metric in metrics:
        if metric not in bench_df or not bench_df[metric].notna().any():
            continue
        filename = f"{metric.replace('/', '_')}_vs_threshold"
        figs[filename] = _render_metric_vs_threshold(bench_df, group_by, metric, reference,
                                                    plots_dir, filename, save)
    print(f"[plotting]     {len(figs)} metric-vs-threshold figure(s): "
          f"{', '.join(sorted(figs))}")
    # plus the headline metrics as one combined figure, for including as a single full-width float
    # instead of a LaTeX subfigure grid that repeats the legend per panel
    for cols, name in ((2, "metrics_vs_threshold_grid"), (len(grid_metrics), "metrics_vs_threshold_row")):
        grid = plot_metric_grid_vs_threshold(bench_df, group_by, grid_metrics, reference, ncols=cols,
                                             plots_dir=plots_dir, save=save, filename=name)
        if grid is not None:
            figs[name] = grid
            print(f"[plotting]     combined {cols}-column figure: {name}")
    return bench_df, figs


def _draw_threshold_panel(ax, bench_df, group_by, metric, reference) -> list:
    """Draw one metric-vs-threshold panel onto `ax`; returns [(threshold, handle), ...] for whoever
    builds the legend. Shared by the one-metric-per-figure renderer and the combined grid, so a panel
    is drawn identically either way."""
    sub = bench_df[bench_df[metric].notna()]
    std_col = f"{metric}_std"
    # no additive axis offset ("+8.5e-1" alongside the ticks): it reads as part of the panel title in
    # the grid, and a reader has to add it back by hand to recover an absolute value
    ax.ticklabel_format(axis="y", useOffset=False)

    def _errs(rows):
        """Per-point bootstrap SE, or None when this metric has no finite SE at all (then errorbar
        draws bare markers). NB a metric CAN have a legitimately zero SE -- bench_val/coverage counts
        unique bitstrings over a small unseen-valid set, so all seeds often land on the identical
        value -- which draws a zero-length bar, not a missing one."""
        if std_col not in rows:
            return None
        e = rows[std_col].to_numpy()
        return e if np.isfinite(e).any() else None

    # the endpoint thresholds are known topologies (see _THRESHOLD_ENDPOINTS) -- split them out so
    # each gets its own marker/colour AND its own matching error bar, rather than a recoloured marker
    # sitting on a bar from the main series
    is_endpoint = sub[group_by].apply(
        lambda v: any(np.isclose(v, e) for e in _THRESHOLD_ENDPOINTS))
    main, ends = sub[~is_endpoint], sub[is_endpoint]

    # collected as (threshold, handle) so the legend can be ordered by threshold -- linear (0.0) first,
    # all-to-all (1.0) last -- regardless of the order these artists were drawn in
    legend_entries = []
    if not main.empty:
        # the hand-swept thresholds carry no legend entry of their own -- they're just the sweep's
        # data points, distinguished from the two named-topology endpoints and the reference line
        ax.errorbar(main[group_by], main[metric], yerr=_errs(main), fmt="o", ms=4, ls="none",
                   capsize=2, color="0.35", elinewidth=0.8)
    for value, (label, color, marker) in _THRESHOLD_ENDPOINTS.items():
        row = ends[np.isclose(ends[group_by], value)]
        if row.empty:
            continue
        handle = ax.errorbar(row[group_by], row[metric], yerr=_errs(row), fmt=marker, ms=5, ls="none",
                            capsize=2, color=color, elinewidth=0.8, label=label, zorder=4)
        legend_entries.append((value, handle))
    if reference is not None:
        # "metric-based" IS this threshold -- the sweep member a real metric_based run would actually
        # build -- so the line carries that extension's role colour; the label names the rule that
        # picked it (knee or percolation), the config detail a reader needs to reproduce the value.
        rule, value = reference
        handle = ax.axvline(value, color=_ROLE_COLORS["metric-based"], ls="--", lw=1.2,
                            label=f"{rule} rule @ {value:.3f}")
        legend_entries.append((value, handle))
    # A metric can saturate and come out identical at every threshold (coverage and qBAS both can).
    # Matplotlib then pads the axis to +/-1e-12 and, with the offset switched off above, labels the
    # ticks to 13 decimals -- wide enough to shove a neighbouring panel out of the way. Give the flat
    # case a readable window instead, which also shows plainly that nothing varies.
    vals = sub[metric].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size and np.ptp(vals) <= 1e-9 * max(1.0, abs(float(vals[0]))):
        centre = float(vals[0])
        pad = max(abs(centre) * 0.05, 1e-3)
        ax.set_ylim(centre - pad, centre + pad)
    return legend_entries


def _render_metric_vs_threshold(bench_df, group_by, metric, reference, plots_dir, filename, save):
    """Render one metric-vs-threshold figure; see plot_metrics_vs_threshold."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    legend_entries = _draw_threshold_panel(ax, bench_df, group_by, metric, reference)
    if legend_entries:
        legend_entries.sort(key=lambda e: e[0])
        ax.legend([h for _, h in legend_entries], [h.get_label() for _, h in legend_entries],
                 fontsize=7, frameon=True)
    ax.set_xlabel("Threshold")
    ax.set_ylabel(_BENCH_METRIC_LABELS.get(metric, metric))
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


def plot_metric_grid_vs_threshold(bench_df, group_by, metrics, reference, ncols: int = 2,
                                  fig_width: float = _GRID_FIG_W, panel_h: float = _GRID_PANEL_H,
                                  plots_dir: str = "plots", save: bool = True,
                                  filename: str = "metrics_vs_threshold_grid"):
    """Several metric-vs-threshold panels in ONE figure with ONE shared legend.

    Replaces a LaTeX subfigure grid of the per-metric figures, which repeated the identical
    three-entry legend once per panel along with every axis label. Here the legend is drawn once
    below the panels and the threshold axis is shared (sharex), so only the bottom row carries tick
    labels and the "Threshold" label.

    Each panel is titled "(a) MMD" so the text can reference sub-panels the way subcaptions did, and
    the metric in the title replaces a per-panel y-axis label. `ncols=2` gives a 2x2 block; ncols =
    len(metrics) gives a single flat row, which is much shorter but leaves each panel narrow.

    `fig_width` is the TOTAL width and is what keeps the type readable: the figure is authored at the
    width it will be included at, so \\includegraphics[width=\\textwidth] neither up- nor down-scales
    it. Fixing panel width instead would make an n-panel row n times too wide, and LaTeX would then
    shrink the whole thing -- fonts included -- to fit.

    Returns the figure, or None if none of `metrics` is present and non-NaN in bench_df.
    """
    from string import ascii_lowercase
    from matplotlib.ticker import MaxNLocator

    metrics = [m for m in metrics if m in bench_df and bench_df[m].notna().any()]
    if not metrics:
        return None
    ncols = min(ncols, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axs = plt.subplots(nrows, ncols, sharex=True, squeeze=False,
                            figsize=(fig_width, panel_h * nrows + 0.35))
    flat = axs.ravel()

    # one entry per distinct label across all panels: every panel draws the same three artists, so
    # collecting them into a dict keyed on label is what removes the redundancy
    entries = {}
    for k, (ax, metric) in enumerate(zip(flat, metrics)):
        for value, handle in _draw_threshold_panel(ax, bench_df, group_by, metric, reference):
            entries.setdefault(handle.get_label(), (value, handle))
        ax.set_title(f"({ascii_lowercase[k]}) {_BENCH_METRIC_LABELS.get(metric, metric)}",
                     fontsize=_GRID_TITLE_SIZE)
        ax.tick_params(labelsize=_GRID_TICK_SIZE)
        # cap the tick count per axis: at ncols=4 a panel is under 2in wide, where the default
        # 0.0/0.2/../1.0 x ticks and a 4-significant-digit y axis collide with their neighbours
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # sharex hides tick labels on every row but the last; re-show them on any panel with nothing
        # beneath it, else a partly-filled grid loses its x axis
        if k + ncols >= len(metrics):
            ax.set_xlabel("Threshold", fontsize=_GRID_LABEL_SIZE)
            ax.tick_params(labelbottom=True)
    for ax in flat[len(metrics):]:
        ax.axis("off")

    legend = None
    if entries:
        ordered = sorted(entries.values(), key=lambda e: e[0])
        legend = fig.legend([h for _, h in ordered], [h.get_label() for _, h in ordered],
                            loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(ordered),
                            fontsize=_GRID_LEGEND_SIZE, frameon=False)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename, extra_artists=(legend,) if legend else None)
    return fig


# --------------------------------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------------------------------
# The per-step MMD targets logged by training (qcbm.MMD_KEYS): each gets an "MMD" ylabel and a title
# of just the split name, instead of the generic metric-name-derived labeling. generate_all_figures
# plots the three single splits by default; the union targets only when asked for via --metrics.
_MMD_SPLIT_TITLES = {
    "train/mmd_train": "Training",
    "train/mmd_val": "Validation",
    "train/mmd_test": "Test",
    "train/mmd_train_val": "Training + Validation",
    "train/mmd_train_test": "Training + Test",
    "train/mmd_train_val_test": "Full Dataset",
}


def generate_all_figures(sweep_id: str, entity: str, project: str, dataset_cfg: dict = None,
                         group_by: str = "circuit.extension",
                         metrics=("train/mmd_train", "train/mmd_val", "train/mmd_test"),
                         plots_dir: str = "plots", science_style: bool = True, n_boot: int = 1000,
                         which: str = "final", n_shots: int = 10000):
    """The training-dependent figure set for a sweep: metric-vs-measurements curves (train and val
    combined side by side) + bootstrap benchmark tables, plus the per-feature QQ figures for JGB.

    Curves and benchmark both aggregate ACROSS ALL SEED-RUNS of each group via bootstrap (mean +/-
    across-seed SE, `n_boot` resamples). `which` defaults to "final" rather than "best": the
    val-selected checkpoint is not usable here, where mmd_val selection fires at initialization for
    most runs (see plot_qq_vs_data). Saves to plots_dir/<sweep_id>-<dataset>/, with <dataset> read off
    the run configs (detect_dataset), which also gates the JGB-only QQ figures; `dataset_cfg` is only
    cross-checked against it.

    `n_shots` sets the sampling-noise floor on every reported metric and is the dominant cost of a
    pass (one simulation per run); raise it once the error bars fall below the shot noise.

    Static dataset/topology/threshold figures are NOT generated here -- see src.plot_extension. For a
    circuit.threshold sweep see plot_metrics_vs_threshold.
    """
    if science_style:
        use_science_style()
    print(f"[plotting] sweep={sweep_id} group_by={group_by} which={which} n_shots={n_shots:,}")

    # 1) metric-vs-measurements, bootstrapped over all seeds. wandb is queried exactly twice per
    # pass: once for the sweep's run list (cached in src.benchmark and reused below) and once per run
    # for the history of ALL requested metrics at once.
    print("[plotting] (1/2) fetching runs from wandb...")
    grouped = fetch_runs(sweep_id, entity, project, group_by)
    n_runs = sum(len(v) for v in grouped.values())
    print(f"[plotting]     found {n_runs} runs across {len(grouped)} group(s): "
          f"{', '.join(str(k) for k in grouped)}")
    # the run list is also what the dataset label comes from, so the output directory can't disagree
    # with the sweep it was built from
    dataset = detect_dataset(grouped, expected=(dataset_cfg or {}).get("dataset"))
    plots_dir = os.path.join(plots_dir, f"{sweep_id}-{dataset}")
    print(f"[plotting]     dataset={dataset} -> {plots_dir}/")
    print(f"[plotting]     fetching histories ({len(metrics)} metric(s), one request per run)...")
    prefetch_histories(grouped, metrics)
    # train/val MMD are combined into one side-by-side figure whenever both are present; a caller who
    # narrows `metrics` to just one of the two still gets it standalone via the loop below
    combine_train_val = {"train/mmd_train", "train/mmd_val"}.issubset(metrics)
    for metric in metrics:
        if combine_train_val and metric in ("train/mmd_train", "train/mmd_val"):
            continue
        print(f"[plotting]     plotting {metric} vs. measurements (bootstrap over seeds)...")
        split = _MMD_SPLIT_TITLES.get(metric)
        # metric names may contain "/"; flatten to "_" so the filename doesn't imply a nested
        # directory that was never created (plots_dir is the only dir made)
        plot_mmd_vs_measurements(grouped, metric=metric, n_boot=n_boot,
                                 log_axes=split is not None,
                                 ylabel="MMD" if split else None,
                                 title=split,
                                 filename=f"{metric.replace('/', '_')}_measurements", plots_dir=plots_dir)
    if combine_train_val:
        print("[plotting]     plotting train/val MMD side by side (bootstrap over seeds)...")
        plot_train_val_mmd(grouped, n_boot=n_boot, plots_dir=plots_dir)
    print("[plotting]     done.")

    # 2) benchmark: evaluate every run's checkpoint (per `which`), then bootstrap across seeds.
    # `grouped` is passed through so the benchmark reuses the run list instead of re-querying.
    print("[plotting] (2/2) benchmarking all runs per group (bootstrap over seeds)...")
    per_run = bm.benchmark_all_runs(sweep_id, entity, project, group_by, n_shots=n_shots,
                                    which=which, groups=grouped)
    bench_df = bootstrap_group_metrics(per_run, group_by=group_by, n_boot=n_boot)
    if bench_df is None or bench_df.empty:
        # e.g. no run had a usable checkpoint -> fall back to the best-per-group point estimate
        print("[plotting]     no per-run metrics; falling back to best-per-group point estimate.")
        bench_df = bm.benchmark_sweep(sweep_id, entity, project, group_by, n_shots=n_shots,
                                      which=which, groups=grouped)
    save_metric_table(bench_df, plots_dir=plots_dir)          # full-precision CSV (all columns)
    plot_benchmark_tables(bench_df, group_by=group_by, plots_dir=plots_dir)  # readable per-family cuts
    if dataset == "JGB":
        # one featured model per group, and the across-seed bootstrap band over all of them (the
        # latter carries no seed-selection choice -- see plot_qq_bootstrap_vs_data)
        plot_qq_vs_data(sweep_id, entity, project, group_by, n_shots=n_shots, plots_dir=plots_dir,
                        runs_by_key=grouped, which=which)
        plot_qq_bootstrap_vs_data(sweep_id, entity, project, group_by, n_shots=n_shots,
                                  n_boot=n_boot, plots_dir=plots_dir, runs_by_key=grouped,
                                  which=which)
    print("[plotting]     done.")
    print(f"[plotting] finished -- figures in {plots_dir}/")
    return bench_df


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def _parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m src.plotting",
        description="Regenerate the training-dependent figures (MMD-vs-measurements, benchmark "
                    "tables, JGB QQ) for a wandb sweep. For the static dataset/topology/threshold "
                    "figures, use `python -m src.plot_extension` instead.")
    parser.add_argument("--sweep-id", required=True,
                        help="wandb sweep id, e.g. printed in a run's log line "
                             "'Program started (..., sweep_id=...)', or from the Sweeps tab.")
    parser.add_argument("--project", default="qcbm-circuit-design", help="wandb project name.")
    parser.add_argument("--entity", default=None, help="wandb entity (default: your default entity).")
    parser.add_argument("--dataset", choices=["BAS", "JGB"], default=None,
                        help="Cross-check only. The dataset is read from the sweep's own run configs "
                             "(it names the output directory and gates the dataset-specific figures); "
                             "passing it here just warns if it disagrees with the runs.")
    parser.add_argument("--group-by", default=None,
                        help="Dot-separated config key to use as the plot legend/grouping dimension "
                             "(default: circuit.extension, or circuit.threshold with "
                             "--threshold-sweep; can be any swept key).")
    parser.add_argument("--metrics", nargs="+",
                        default=["train/mmd_train", "train/mmd_val", "train/mmd_test"],
                        help="Logged metrics to plot vs. cumulative measurements.")
    parser.add_argument("--which", choices=["best", "final"], default="final",
                        help="Which checkpoint to benchmark/sample: final (last training iteration, "
                             "default) or best (validation-selected -- not meaningful for this "
                             "pipeline, where mmd_val selection fires at initialization for most "
                             "runs).")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Treat this sweep as a circuit.threshold sweep: plot every benchmark "
                             "metric vs. threshold (one figure each) with the config-selected "
                             "threshold_rule marked, instead of the standard per-extension figure set.")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap resamples for the across-seed mean/SE (curves + benchmark).")
    parser.add_argument("--n-shots", type=int, default=10000,
                        help="Shots used to sample each checkpoint for the benchmark metrics (and "
                             "the JGB QQ figures). Sets the sampling-noise floor on every reported "
                             "metric and is the dominant cost of a pass.")
    parser.add_argument("--plots-dir", default="plots", help="Output directory for the PDFs/PNGs.")
    parser.add_argument("--no-science-style", action="store_true",
                        help="Skip the scienceplots styling (use matplotlib defaults).")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    group_by = args.group_by or ("circuit.threshold" if args.threshold_sweep else "circuit.extension")

    if args.threshold_sweep:
        bench_df, _figs = plot_metrics_vs_threshold(
            sweep_id=args.sweep_id, entity=args.entity, project=args.project,
            group_by=group_by, n_shots=args.n_shots, n_boot=args.n_boot, which=args.which,
            plots_dir=args.plots_dir, science_style=not args.no_science_style,
        )
        print(f"Figures written to {args.plots_dir}/{args.sweep_id}-threshold/")
    else:
        dataset_cfg = {"dataset": args.dataset} if args.dataset else None
        bench_df = generate_all_figures(
            sweep_id=args.sweep_id,
            entity=args.entity,
            project=args.project,
            dataset_cfg=dataset_cfg,
            group_by=group_by,
            metrics=tuple(args.metrics),
            plots_dir=args.plots_dir,
            science_style=not args.no_science_style,
            n_boot=args.n_boot,
            which=args.which,
            n_shots=args.n_shots,
        )
    if bench_df is not None and not bench_df.empty:
        print(bench_df.to_string(index=False))


if __name__ == "__main__":
    main()
