"""wandb-driven plotting for QCBM sweeps.

Replaces the hardcoded-path `results v1/plot_figures.ipynb`: figures that depend on training runs
are built by pulling run histories from a wandb sweep (grouped by any config dimension), and MMD is
plotted against cumulative circuit *measurements* rather than iteration index. All figures are
saved as PDF (and PNG).

Static, training-independent dataset/topology/threshold figures live in src.plot_extension instead
-- they don't need a wandb sweep and only need regenerating when the data or extension settings
change, not on every call here.

Typical use:
    from src.plotting import generate_all_figures
    generate_all_figures(sweep_id="<sweep>", entity="<you>", project="qcbm-circuit-design")

The dataset a sweep was trained on is read off the runs' own logged configs (see detect_dataset), not
passed in -- it decides the output directory name and which dataset-specific figures are generated,
and getting it from the caller meant a default could silently mislabel a whole figure set.
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
    "random": "random",
    "metric_based": "metric-based",
    "chow_liu": "chow-liu",
    "all_to_all": "all-to-all",
    "nearest_neighbor": "nearest-neighbor",   # in EXCLUDED_EXTENSIONS, so normally never plotted
}

# extensions kept out of EVERY figure: fetch_runs drops their runs, so no curve, benchmark row, table
# or QQ panel is ever built for them, and src.plot_extension omits their topology panel. This is a
# PLOTTING decision only -- src.extension/src.setup still implement them and a run can still be
# configured with one; it simply won't be shown.
EXCLUDED_EXTENSIONS = {"nearest_neighbor"}

# Okabe & Ito (2008) colorblind-safe palette -- the de facto standard for categorical color in
# scientific publishing (Wong, "Points of view: Color blindness", Nature Methods 8, 441, 2011).
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

# One fixed color per extension, keyed on the canonical label (_EXTENSION_LABELS), so an extension
# keeps its color in EVERY figure -- across sweeps, and regardless of which/how-many other extensions
# are present. This has to be exhaustive over the extensions, not just a couple of them: the fallback
# below assigns by position, so with only some extensions pinned, a sweep missing one group (or a
# wandb run list in a different order) silently shifted every unpinned extension's color, and the same
# extension then appeared in different colors in two figures of the same paper.
_ROLE_COLORS = {
    "linear": OKABE_ITO["blue"],
    "chow-liu": OKABE_ITO["bluish_green"],
    "metric-based": OKABE_ITO["orange"],
    "random": OKABE_ITO["vermillion"],
    "all-to-all": OKABE_ITO["reddish_purple"],
    "nearest-neighbor": OKABE_ITO["sky_blue"],
}
# fallback draw order for categorical keys that are NOT a known extension (e.g. a newly added one, or
# a sweep grouped by some other config key); ordered so a role color is never handed out twice
_CB_CYCLE = [OKABE_ITO["sky_blue"], OKABE_ITO["yellow"], OKABE_ITO["blue"], OKABE_ITO["vermillion"],
            OKABE_ITO["reddish_purple"], OKABE_ITO["orange"], OKABE_ITO["bluish_green"]]

# shared sequential colormap for all continuous/2-D fields (heatmaps, numeric sweeps): cividis is
# perceptually uniform AND colorblind-safe, so it pairs with the Okabe-Ito categorical palette above.
# Imported by src.plot_extension so its figures use the same palette as this module's.
SEQUENTIAL_CMAP = "cividis"


def categorical_colors(n: int) -> list:
    """`n` distinct colorblind-safe categorical colors from the shared Okabe-Ito cycle (the same
    palette default_colors assigns to extensions), for series that aren't keyed on an extension --
    e.g. per-feature JGB lines/histograms in src.plot_extension."""
    return [_CB_CYCLE[i % len(_CB_CYCLE)] for i in range(n)]


def default_colors(legend_keys) -> dict:
    """Color map preserving the v1 convention for extensions; Okabe-Ito colorblind-safe categorical
    palette for named/extra keys, the shared SEQUENTIAL_CMAP (cividis) for numeric sweeps.

    Every known extension gets its pinned _ROLE_COLORS color, so the mapping for a given key does NOT
    depend on which other keys are present or on the order they arrive in -- two figures of the same
    sweep, and the same extension across sweeps, are guaranteed the same color. Unknown keys fall back
    to _CB_CYCLE, assigned in sorted order (again order-independent) and skipping colors a role in this
    same call already claimed, so the fallback cannot collide with a pinned extension.
    """
    keys = list(legend_keys)
    # numeric sweep dimension -> sequential colormap ordered by value
    try:
        numeric = sorted(keys, key=lambda k: float(k))
        shades = plt.get_cmap(SEQUENTIAL_CMAP)(np.linspace(0.15, 0.9, len(numeric)))
        return {k: shades[i] for i, k in enumerate(numeric)}
    except (TypeError, ValueError):
        pass
    colors = {k: _ROLE_COLORS[_EXTENSION_LABELS.get(k, k)] for k in keys
              if _EXTENSION_LABELS.get(k, k) in _ROLE_COLORS}
    taken = set(colors.values())
    spare = [c for c in _CB_CYCLE if c not in taken] or _CB_CYCLE
    for i, k in enumerate(sorted((k for k in keys if k not in colors), key=str)):
        colors[k] = spare[i % len(spare)]
    return colors


def _save(fig, plots_dir, filename, extra_artists=None):
    """`extra_artists` (e.g. a legend added via ax.add_artist rather than ax.legend, so a second
    legend can coexist on the same axes -- see plot_extension.plot_jgb_raw_data) is passed through as
    bbox_inches="tight"'s bbox_extra_artists: savefig's tight-bbox pass only auto-discovers the axes'
    CURRENT legend (ax.legend_), so an orphaned one is otherwise sized without its own extent and gets
    clipped at the figure edge instead of padded like every other artist."""
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
    """Return {legend_key: [runs]} for a real wandb Sweep, grouped by a config dimension.

    group_by and filters keys are dot-separated paths into the run's config, e.g. "circuit.extension".
    filters is an optional {config_key: value} dict to pin non-legend swept params (facet slicing),
    applied client-side since Sweep.runs is a materialized list, not a server-side query.

    The sweep query and the config parsing both go through benchmark's process-wide caches, so
    re-grouping the same sweep (a different group_by, a different facet filter, or benchmark's own
    _group_runs) costs no additional wandb request.

    Runs whose logged config predates the current schema (e.g. from before a field was renamed or
    regrouped) are skipped with a warning rather than aborting the whole fetch.

    Runs of an EXCLUDED_EXTENSIONS extension are dropped here regardless of `group_by`, so every
    downstream figure (curves, benchmark tables, the QQ figure -- they all reuse this mapping) omits
    them.
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
    """The dataset kind ("BAS"/"JGB") a sweep's runs were actually trained on, per their logged config.

    This is the authoritative source for how a figure set is labelled and which dataset-specific
    figures it gets. The benchmark metrics themselves never depend on it -- each run is scored
    against its own dataset (see benchmark._evaluate_run) -- so a wrong kind doesn't corrupt any
    number, it mislabels the output directory and gates the dataset-specific figures (e.g. the JGB QQ
    grids) on the wrong answer. `expected` is therefore only a cross-check: a disagreement warns and
    the runs win.

    Raises ValueError if no run config could be read at all (nothing to label the figures from).
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

# rows to request per history fetch; comfortably above any realistic run length so wandb returns
# every logged step instead of a subsample (its default is 500 -- see _raw_history)
_HISTORY_SAMPLES = 10_000


def _raw_history(run, metrics: tuple) -> pd.DataFrame:
    """Cached raw history dataframe for `run` covering ALL of `metrics` in a single wandb request.

    Every metric plotted for a run shares one sampled-history request (plus the
    train/cumulative_measurements x-axis) instead of one request per metric, and the result is
    cached per run, so a plotting pass costs one history fetch per run regardless of how many
    metrics/figures consume it. A later request for metrics outside the cached set refetches the
    union, keeping the invariant that the cached frame covers everything asked for so far.

    Safe to batch because the metrics plotted here (train/mmd_* and the bench_*/ suite) are all
    logged on the same eval steps -- see QCBM.stochastic_gradient_descent, which writes them in one
    payload -- so no metric loses rows by sharing a request. Per-metric NaN rows are dropped by
    fetch_history anyway.

    `samples` is raised well above wandb's default of 500: cumulative_measurements is logged on EVERY
    training iteration, and a measurement-budget run of a small circuit can exceed 500 of them (the
    linear group ran 602), at which point the sampled-history endpoint silently subsamples and the
    curve is drawn from a thinned, unevenly spaced subset of what was logged.
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
    """Warm the history cache with one request per run covering every metric to be plotted.

    Called once up front by generate_all_figures so the per-figure fetch_history calls are pure
    cache hits; purely an optimization -- fetch_history works standalone.
    """
    metrics = tuple(metrics)
    for runs in runs_by_key.values():
        for r in runs:
            _raw_history(r, metrics)


def reset_history_cache() -> None:
    """Drop cached run histories (use when re-plotting a sweep that is still running)."""
    _HISTORY_CACHE.clear()


def fetch_history(run, metric: str = "train/mmd_train") -> pd.DataFrame:
    """Per-run dataframe with train/cumulative_measurements and the requested metric (NaN rows dropped).

    Reads the shared per-run history cache (see _raw_history), so requesting several metrics for the
    same run costs a single wandb request in total.
    """
    df = _raw_history(run, (metric,))
    if df.empty or metric not in df:
        cfg = bm.run_config(run)
        n = run.summary.get("iterations_run") or cfg.qcbm.iterations
        P = run.summary.get("train/num_parameters") or run.config.get("num_parameters", 0)
        shots = cfg.qcbm.N_shots
        per = run.summary.get("measurements_per_step") or (2 * P + 1) * shots
        df = pd.DataFrame({"train/cumulative_measurements": np.arange(1, n + 1) * per,
                           metric: [np.nan] * n})
    return df.dropna(subset=[metric]).sort_values("train/cumulative_measurements")


def aggregate_over_measurements(histories: list, metric: str, mode: str = "bootstrap",
                                n_grid: int = 400, window: int = 1, n_boot: int = 1000,
                                boot_seed: int = 0, log_x: bool = False) -> dict:
    """Aggregate the runs of one group into a mean curve + spread band, across seeds.

    Seed-runs of a group normally share an IDENTICAL measurement axis, and then the aggregation
    happens at those native positions with NO interpolation and NO resampling -- every logged point
    is kept, and nothing is invented. `n_grid` only matters for the fallback below.

    `mode`:
      - "bootstrap" (default): at each x, resample the runs with replacement n_boot times; `line` is
        the mean of the bootstrap means and the band is +/- the bootstrap std of the mean (the
        across-seed standard error), via utils.bootstrap_mean_std.
      - "meanstd": plain across-run mean +/- std.
      - "medperc": median with the 10th/90th percentile band.

    Fallback: when the runs genuinely disagree on x (e.g. one stopped early), they are not
    index-aligned, so they are interpolated onto a shared grid over the window where all of them have
    data -- log-spaced when `log_x`, else linear. That path is a last resort because it distorts a
    log-x plot in both directions: MMD is logged only every qcbm.eval_every iterations (10 by
    default), so the first gap alone spans ~1 decade -- 37-50% of the axis at realistic sweep sizes --
    and np.interp's linear-in-x path across it renders as a plateau-then-knee that is pure artifact,
    while at the high end a log grid collapses several real samples into a single cell and throws the
    rest away.

    `log_x` additionally drops the step-0 (pre-training) row every run logs at
    cumulative_measurements=0: log(0) is undefined, and padding the axis down to some small positive
    floor to fake it costs decades of width for a flat line carrying one number (a floor of
    first_step/1000 spent 3 decades on padding vs. 2.3 decades of actual data, crushing the curve into
    the right third of the figure). Callers that want the pre-training value on the plot draw it as a
    horizontal reference line instead (see _draw_mmd_curves).
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
                     colors: dict, log_axes: bool) -> None:
    """Draw one aggregated curve + bootstrap-SE band per group onto `ax`, plus the pre-training
    baseline as a horizontal reference line; shared by plot_mmd_vs_measurements (one metric, one
    panel) and plot_train_val_mmd (two metrics, two panels) so the per-group aggregation/drawing
    logic isn't duplicated between them.

    The curves themselves span only real measurements (see aggregate_over_measurements), so the
    step-0 value is shown as the reference line rather than as a point on the curves. One line, in
    neutral grey: step 0 is the SHARED linear/unextended circuit at the sweep's fixed
    initial_random_seed -- bit-identical across every run and every connectivity in the sweep (see
    QCBM._log_baseline_step) -- so it is a property of the sweep, not of any one group, and colouring
    it per-group would imply a per-group value that doesn't exist.
    """
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
        ax.plot(agg["x"], agg["line"], label=label, color=colors.get(key))
        ax.fill_between(agg["x"], agg["lower"], agg["upper"], alpha=0.3, lw=0.0, color=colors.get(key))
    # runs predating the baseline-logging change have no step-0 row -> no line rather than a guess
    if baselines:
        ax.axhline(float(np.mean(baselines)), ls=":", lw=0.9, color="0.4", zorder=0,
                   label="MPS baseline")
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")


def plot_mmd_vs_measurements(runs_by_key: dict, metric: str = "train/mmd_train", mode: str = "bootstrap",
                             window: int = 1, n_boot: int = 1000, colors: dict = None,
                             filename: str = "MMD_measurements", plots_dir: str = "plots",
                             save: bool = True, log_axes: bool = False, title: str = None,
                             ylabel: str = None):
    """MMD (or any logged metric) vs cumulative measurements, aggregated across seeds per group.

    Defaults to the bootstrap aggregation (mean +/- across-seed standard error); see
    aggregate_over_measurements for the other modes. The shaded band is the bootstrap std of the mean.

    `log_axes` plots on log-log axes with the iteration-0 value anchoring the curve's left edge (see
    aggregate_over_measurements). `title`/`ylabel` override the defaults (generic metric-name-derived
    ylabel, no title) -- e.g. generate_all_figures passes ylabel="MMD" and a split title for the
    train/mmd_test curve (train/val get the combined side-by-side figure instead, see
    plot_train_val_mmd).
    """
    colors = colors or default_colors(runs_by_key.keys())
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    _draw_mmd_curves(ax, runs_by_key, metric, mode, window, n_boot, colors, log_axes)
    ax.set_xlabel("Measurements")
    ax.set_ylabel(ylabel or metric.replace("_", " ").upper())
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=7, frameon=True)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


def plot_train_val_mmd(runs_by_key: dict, mode: str = "bootstrap", window: int = 1, n_boot: int = 1000,
                       colors: dict = None, filename: str = "train_val_mmd_measurements",
                       plots_dir: str = "plots", save: bool = True, log_axes: bool = True):
    """Train MMD and Val MMD vs. cumulative measurements, as two side-by-side subplots sharing one
    legend. Each panel has its own independent y-axis (not shared -- train and val MMD can differ
    enough in scale that forcing one range would flatten one of the two curves). See
    plot_mmd_vs_measurements for the aggregation/log-axis-anchoring behavior, which is identical here
    per panel (via the shared _draw_mmd_curves helper).
    """
    colors = colors or default_colors(runs_by_key.keys())
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    for ax, metric, split in ((axs[0], "train/mmd_train", "Training"),
                              (axs[1], "train/mmd_val", "Validation")):
        _draw_mmd_curves(ax, runs_by_key, metric, mode, window, n_boot, colors, log_axes)
        ax.set_xlabel("Measurements")
        ax.set_title(split)
        ax.set_ylabel("MMD")
    # merge handles/labels across both panels (in case a group has data in one split but not the
    # other) into a single shared legend rather than repeating an identical one on each panel
    handles_by_label = {}
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles_by_label.setdefault(label, handle)
    # ordered by parameter count -- the MPS baseline has none of the extension's added SU(4) gates,
    # so it always leads regardless of count; everything else follows ascending by how many
    # parameters that extension adds, matching how the benchmark tables are already sorted (see
    # bootstrap_group_metrics). A group whose runs predate parameter logging (None) sorts last rather
    # than crashing the comparison.
    param_count_by_label = {}
    for key, runs in runs_by_key.items():
        label = _EXTENSION_LABELS.get(key, str(key))
        counts = [r.summary.get("train/num_parameters") for r in runs]
        counts = [c for c in counts if c is not None]
        if counts and label not in param_count_by_label:
            param_count_by_label[label] = counts[0]
    def _legend_key(label):
        if label == "MPS baseline":
            return (-1, 0)
        return (0, param_count_by_label.get(label, np.inf))
    ordered_labels = sorted(handles_by_label, key=_legend_key)
    axs[0].legend([handles_by_label[l] for l in ordered_labels], ordered_labels, loc="best",
                 fontsize=7, frameon=True)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


# --------------------------------------------------------------------------------------------------
# benchmark figures (bootstrap across all runs per group)
# --------------------------------------------------------------------------------------------------
# setup/cost facts (see benchmark._run_setup_facts): deterministic per group -- same architecture,
# same measurement/iteration budget for every seed-run -- so they get a plain mean, no bootstrap SE,
# and lead the column order rather than being interleaved among the outcome metrics.
_SETUP_COLUMNS = ("num_parameters", "n_connections", "total_measurements", "iterations_run")


def bootstrap_group_metrics(per_run_df: pd.DataFrame, group_by: str = "circuit.extension",
                            metric_cols: list = None, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Bootstrap held-out benchmark metrics across the runs of each group.

    `per_run_df` is the tidy one-row-per-run table from benchmark.benchmark_all_runs (a `group_by`
    column plus numeric metric columns). For each group and metric, resample the group's runs with
    replacement n_boot times and report the mean and the std of the bootstrap means (the across-seed
    standard error), via utils.bootstrap_mean_std. Non-finite per-run values (e.g. bench_val/* for a
    full_support run) are dropped before resampling. The setup/cost columns (_SETUP_COLUMNS) are
    constant per group, so they get a plain mean and no `_std` companion instead.

    Returns one row per group with, for each metric <m>, a column <m> (bootstrap mean, or plain mean
    for a setup/cost column) and <m>_std (bootstrap SE, omitted for setup/cost columns), plus n_runs.
    Setup/cost columns lead the column order.
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
    """Write the full benchmark table (every group/metric/<metric>_std/n_runs column) as CSV.

    Works for both the bootstrap table (bootstrap_group_metrics) and the point-estimate table
    (benchmark_sweep). No figure is rendered -- a full-precision CSV is for downstream analysis, not
    at-a-glance reading, so a rendered table added no value over it.
    """
    if bench_df is None or bench_df.empty:
        return None
    if save:
        os.makedirs(plots_dir, exist_ok=True)
        bench_df.to_csv(f"{plots_dir}/{filename}.csv", index=False)
    return bench_df


# --------------------------------------------------------------------------------------------------
# rendered benchmark tables (the full-precision CSV above stays the source of truth; these are the
# readable per-family cuts of it)
# --------------------------------------------------------------------------------------------------
# Which direction is "better" per metric, for the bold-best marking. Distances, divergences and
# negative log-likelihoods are MINIMIZED (mmd, kl, tv, nll); scores, fidelities, coverages and rates
# are MAXIMIZED. A column absent from this map is never bolded -- see _TABLE_SPECS for why the
# setup/cost table has no "best". bench_dist/* is scored against the FULL dataset (train+val+test
# merged), not per split -- see benchmark.evaluate.
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

# columns rendered as plain counts (no ± std, no decimals) rather than as metric values
_COUNT_COLUMNS = {"n_runs", "num_parameters", "n_connections", "total_measurements", "iterations_run"}

# (filename, title, show_std, ((column, header label), ...)) for each rendered table.
# The setup/cost table deliberately shows no ± std and marks no "best": its columns are experiment
# DESIGN facts, not outcomes, and they are identical across the seeds of a group (std 0). Bolding a
# "best" there would be actively misleading -- fewest parameters/connections would flag the linear
# circuit as the winner, when the whole point of the comparison is that added connections buy
# expressiveness at the cost of parameters.
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
    ("benchmark_table_distribution", "Distribution distance metrics (full dataset)", True,
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
    """{column index: row index of the best value} for the columns that have a defined direction.

    Direction comes from _LOWER_IS_BETTER, so a minimized metric (mmd/kl/tv) picks the SMALLEST value
    and a maximized one (fidelity/coverage/rate/qbas) the LARGEST. Columns with no defined direction,
    and all-NaN columns, are simply absent from the result (nothing gets bolded).
    """
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
    """Render the benchmark table as several readable per-family figures (see _TABLE_SPECS): setup/
    cost, validity-generalization, distribution distances, and BAS metrics.

    Each metric table shows "mean ± bootstrap SE" and marks the best value per column in bold,
    respecting each metric's own direction (lowest MMD/KL/TV, highest fidelity/coverage/rate/qBAS).
    The full-precision CSV written by save_metric_table remains the complete table; these are cuts of
    it for reading. Tables whose columns are all absent from bench_df are skipped (e.g. the BAS table
    for a JGB sweep). Rows are sorted by parameter count (ascending) so groups line up the same way,
    smallest/cheapest circuit first, across all four tables. Returns {filename: figure}.
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
    # Height the figure to the table rather than the reverse: matplotlib sizes table rows as a
    # FRACTION of the axes (Table._approx_text_height ~ fontsize/72 * 1.2, then scaled), so a figure
    # taller than the rows need leaves the table floating in whitespace. Solving that relation for
    # the figure height makes the table fill the canvas at any row count.
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


# Shared data/encoding context for a JGB sweep's QQ figures. Every run of a sweep trains on the same
# dataset, so the empirical marginals (and the axes they set) are built once from the first run's
# config; a run whose ENCODING differs would be plotted against a reference it never saw, so callers
# compare `encoding` and drop mismatches rather than silently rescaling.
_JGBRef = namedtuple("_JGBRef", "encoding data n_features bits_per_feature quantizer feature_names")


def _jgb_reference(cfg) -> _JGBRef:
    jgb = JGB(cfg.data.N_qubits, cfg.data.N_features, cfg.data.quantizer)
    dl = DataLoader(jgb)
    dl.train_val_test_split(cfg.data.train_split, cfg.data.val_split)
    return _JGBRef((cfg.data.N_qubits, cfg.data.N_features, cfg.data.quantizer),
                   jgb.decimal.values, cfg.data.N_features, jgb.bits_per_feature, dl.quantizer,
                   list(jgb.raw.columns))  # bond tenors, e.g. "5Y"/"10Y"/"20Y", in feature order


def _run_qq_quantiles(run, ref: _JGBRef, n_shots: int, which: str, n_q: int) -> list:
    """One run's per-feature model quantiles, on the shared data-quantile grid (x is the same for
    every run and every reference curve, so only the y-values are returned)."""
    cfg = bm.run_config(run)
    circuit, params = bm.load_checkpoint(run, which=which)
    samples = bm.sample_model(circuit, params, n_shots, seed=cfg.sweep.random_seed)
    feats = bm.reconstruct_features(samples, ref.bits_per_feature, ref.n_features,
                                    quantizer=ref.quantizer)
    return [bm.qq_model_vs_data(*feats[i], ref.data[:, i], n_q)[1] for i in range(ref.n_features)]


def _group_param_count(runs) -> float:
    """Parameter count of a group, for the shared legend ordering (ascending in what the extension
    adds, matching the MMD legends and benchmark tables). A group predating parameter logging sorts
    last instead of crashing the comparison."""
    counts = [r.summary.get("train/num_parameters") for r in runs]
    counts = [c for c in counts if c is not None]
    return counts[0] if counts else np.inf


def _draw_qq_references(ax, ref: _JGBRef, i: int, n_q: int) -> list:
    """The two reference curves for one panel, on the same lattice and the same estimator as the model
    curves (see benchmark's QQ reference section): the floor -- the data itself round-tripped through
    the encoding, so zero model error -- and a Gaussian baseline. Diagonal-to-floor is quantization,
    floor-to-model is model error. The Gaussian is NOT a floor: a good model can beat it. Returns the
    plotted arrays so the caller can include them in the panel's axis range."""
    fdx, fdy = bm.qq_quantized_data_vs_data(ref.quantizer, i, ref.data[:, i], n_q)
    ax.plot(fdx, fdy, "--", lw=1.1, color=OKABE_ITO["black"], zorder=4, label="quantization floor")
    gdx, gdy = bm.qq_quantized_gaussian_vs_data(ref.quantizer, i, ref.data[:, i], n_q)
    ax.plot(gdx, gdy, ":", lw=1.1, color="0.45", zorder=3, label="quantized Gaussian")
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
    ax.set_title(title)
    ax.set_xlabel("data quantile")


def _qq_legend(fig, axs, save: bool, plots_dir: str, filename: str):
    """One shared legend below the panels: with a series per group plus the references it no longer
    fits inside a panel without covering the curves it describes."""
    handles, labels = axs[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                        ncol=min(len(labels), 7), fontsize=7, frameon=False)
    axs[0].set_ylabel("model quantile")
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename, extra_artists=(legend,))
    return legend


def plot_qq_vs_data(sweep_id: str, entity: str, project: str, group_by: str = "circuit.extension",
                    n_shots: int = 10000, n_q: int = 100, plots_dir: str = "plots", save: bool = True,
                    runs_by_key: dict = None, which: str = "final",
                    select_metric: str = "train/mmd_train", filename: str = "JGB_QQ"):
    """One JGB QQ figure comparing EVERY group's model against the data, one panel per feature.

    Every series in a panel is measured against the same reference -- the empirical data quantiles on
    the x-axis -- so the groups are directly comparable within one panel and the y=x diagonal reads as
    "matches the data" for all of them. Two references share the panel, both on the same lattice and
    read off with the same estimator as the models:

      * "quantization floor" -- the data round-tripped through the encoding, i.e. a perfect model
        (benchmark.qq_quantized_data_vs_data). Diagonal-to-floor is what the encoding costs;
        floor-to-model is that model's own error. No model curve can beat it.
      * "quantized Gaussian" -- a Gaussian fitted to the quantized data, placed analytically on the
        same lattice (benchmark.qq_quantized_gaussian_vs_data). A parametric BASELINE, not a floor:
        its offset also contains the Gaussian's misfit of the data, so a good model can and does beat
        it on heavy-tailed features.

    This replaces a per-model figure carrying model-vs-data, model-vs-normal and data-vs-normal
    curves. Both dropped series compared each model to a DIFFERENT reference -- a Gaussian fitted to
    that model's own marginal -- so they measured Gaussian-ness of the model rather than agreement
    with the data, and could not be read across models. Per-model deviation from the data is now
    read off one panel instead of flipping between figures.

    One model per group, sampled with `n_shots`: the run whose FINAL-iteration `select_metric` is
    lowest (benchmark.select_best_run reads the wandb summary, which holds each metric's last-logged
    value unless explicitly overridden), with `which="final"` to match so the checkpoint plotted is
    the one that value was measured on.

    Note `select_metric="train/mmd_train"` is the final iteration's TRAINING MMD, not the "best_mmd_val"
    summary key (the val-selected checkpoint, which for this pipeline lands at initialization for most
    runs -- best_iter == 1 for 63 of 80 on the sweep this was built against -- and so reports a
    near-untrained model). Final-iteration validation is the other leak-free option but barely
    discriminates here (~1.03x between group medians against ~1.13x scatter between seeds of one
    group); training MMD separates by ~10x with tight within-group ranges, which is why it is the
    default. train/mmd_test must never be used: selecting a run by it leaks the split the figure is
    then read against.

    The trade-off: ranking seeds by training fit prefers whichever seed fit the training sample
    hardest, so this picks a REPRESENTATIVE model for judging marginal shape and is not evidence of
    generalization. The bench_dist/* tables, bootstrapped over all seeds, remain the quantitative
    claim. The across-seed spread is NOT drawn here.

    `runs_by_key` optionally supplies the already-fetched {group_key: [runs]} mapping; without it the
    grouping is (re)built from the cached sweep run list. `which` selects "best" or "final" (see
    benchmark.load_checkpoint). Returns the figure, or None if no group had a usable JGB checkpoint.
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
        _finish_qq_panel(ax, span, f"{ref.feature_names[i]} Rate")
    _qq_legend(fig, axs, save, plots_dir, filename)
    return fig


def plot_qq_bootstrap_vs_data(sweep_id: str, entity: str, project: str,
                              group_by: str = "circuit.extension", n_shots: int = 10000,
                              n_q: int = 100, n_boot: int = 1000, plots_dir: str = "plots",
                              save: bool = True, runs_by_key: dict = None, which: str = "final",
                              residual_row: bool = True, filename: str = "JGB_QQ_bootstrap"):
    """The QQ-vs-data figure with EVERY trained model of each group, bootstrapped across seeds.

    Same panels, references and reading as plot_qq_vs_data (one panel per feature, everything measured
    against the data quantiles on the x-axis, quantization floor + Gaussian baseline drawn the same
    way), but each group is a band instead of a line: every seed-run's checkpoint is sampled, and at
    each quantile level the seeds are bootstrapped into a mean +/- across-seed standard error via
    utils.bootstrap_mean_std -- the same aggregation the MMD curves and benchmark tables already use.

    This removes the single-model figure's selection problem entirely: no seed metric to justify, no
    leakage question, and no dependence on a ranking that (for validation) is mostly noise. It also
    answers what one model cannot -- whether a group's departure from the quantization floor is
    resolved above seed-to-seed scatter, i.e. attributable to the extension rather than to the seed.

    `residual_row` adds a second row plotting each band MINUS the quantization floor. On the raw QQ
    axes both the effect and the band are a few percent of the plotted range -- the across-seed SE
    lands at ~0.1%, i.e. thinner than the line drawn over it -- so the top row alone cannot show
    whether bands separate. The residual row rescales the y-axis to the effect itself: zero is the
    floor (a perfect model), vertical separation between bands is extension-attributable difference,
    and band thickness is seed noise, read in the same units.

    Two caveats on reading it. The band is POINTWISE at each quantile level, not a simultaneous
    confidence region for the whole curve. And it covers seed variability only: shot noise is not
    resampled, being <1% of a bin at these n_shots (see the reference-section note in benchmark).

    Costs one checkpoint load + sampling per RUN rather than per group (~n_runs x the single-model
    figure), the same work benchmark_all_runs already does in a full pass; artifacts are cached
    locally, so a repeat pass re-samples but does not re-download. Returns the figure, or None if no
    group had a usable JGB checkpoint.
    """
    grouped = runs_by_key if runs_by_key is not None else fetch_runs(sweep_id, entity, project, group_by)
    colors = default_colors(list(grouped))
    # (group_key, param_count, per-feature (mean, se) across seeds); as above, the shared data axis
    # has to exist before any curve can be placed on it, so all sampling happens first
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
            except Exception as e:   # a run whose artifact/checkpoint is unusable must not sink the group
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
    # the residual row is not equal-aspect (its y is a difference, not a quantile) so it needs less
    # height than the square QQ panels above it
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
        _finish_qq_panel(ax, span, f"{ref.feature_names[i]} Rate")
        if not residual_row:
            continue
        # same series, floor subtracted: zero is a perfect model, so vertical gaps between bands are
        # extension-attributable and band thickness is seed noise -- both in the same units, on a
        # y-axis scaled to the effect rather than to the quantile range
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
        rax.set_xlabel("data quantile")
    if residual_row:
        grid[1][0].set_ylabel("model $-$ floor")
        for ax in axs:      # the shared x-axis is labelled on the residual row instead
            ax.set_xlabel("")
    _qq_legend(fig, axs, save, plots_dir, filename)
    return fig


# --------------------------------------------------------------------------------------------------
# threshold-sweep figures (opt-in: for a sweep where circuit.threshold itself was varied, a
# fundamentally different sweep shape from the standard per-extension pipeline above)
# --------------------------------------------------------------------------------------------------
# y-axis label per benchmark metric, for the one-figure-per-metric threshold sweep. The key order is
# the order the figures are generated in; anything present in bench_df but missing here still gets a
# figure, labelled with its raw column name.
_BENCH_METRIC_LABELS = {
    "selection_metric": "seed-selection metric (see benchmark.select_best_run)",
    "bench_dist/mmd": "MMD (full dataset)",
    "bench_dist/kl": "KL divergence (full dataset)",
    "bench_dist/tv": "total variation (full dataset)",
    "bench_dist/nll": "negative log-likelihood (full dataset)",
    "bench_dist/fidelity": "classical fidelity (full dataset)",
    "bench_val/coverage": "coverage",
    "bench_val/fidelity": "validity fidelity",
    "bench_val/rate": "rate",
    "bench_val/exploration": "exploration",
    "bench_BAS/precision": "precision",
    "bench_BAS/recall": "recall",
    "bench_BAS/qbas": "qBAS",
}

# The two ends of a metric_based threshold sweep coincide with named topologies, so they are drawn
# with their own marker/colour and called out in the legend rather than reading as anonymous points:
# threshold=0 admits no pair (dist < 0 is never true) and so adds NO connections -- the plain linear
# MPS circuit -- while threshold=1 admits every pair and reproduces all-to-all. Verified against
# benchmark.extension_new_connection_count for BAS 4x3: 0 and 55 added connections, exactly matching
# the `none` and `all_to_all` extensions. NB these are the ENDS of a plateau, not isolated points --
# every threshold below the smallest pairwise distance is linear-equivalent, and every threshold above
# the largest is all-to-all-equivalent -- so neighbouring points can be the same circuit.
# {threshold value: (legend label, colour, marker)}
_THRESHOLD_ENDPOINTS = {
    0.0: ("linear @ 0.000", OKABE_ITO["reddish_purple"], "s"),
    1.0: ("all-to-all @ 1.000", OKABE_ITO["bluish_green"], "D"),
}


_THRESHOLD_RULE_COLORS = {"knee": OKABE_ITO["vermillion"], "percolation": OKABE_ITO["orange"]}


def _threshold_reference_values(any_run) -> list:
    """The single (rule, threshold, colour) reference to mark on a threshold-sweep figure: whichever
    rule cfg.circuit.threshold_rule selects for that run, NOT both knee and percolation -- a
    threshold-sweep run picks one rule to auto-select against (see setup._metric_based_connections),
    so only that rule's value is a real reference point for the sweep; the other rule was never used.

    Computed from ONE run's dataset/extension_metric -- constant across a threshold sweep, where only
    circuit.threshold varies between groups -- via the same helpers setup uses, so the marked value is
    the one the run's auto-selecting rule would actually have picked.

    This line is an annotation, not the data: a config whose dataset can't be reconstructed yields an
    empty list (with a warning) so the metric figures are still produced, unmarked, rather than the
    whole sweep failing over a reference value.
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
        return [(rule, value, _THRESHOLD_RULE_COLORS[rule])]
    except Exception as e:
        print(f"[plotting]     no threshold-rule reference line: {e!r}")
        return []


def plot_metrics_vs_threshold(sweep_id: str, entity: str, project: str,
                              group_by: str = "circuit.threshold", n_shots: int = 10000,
                              n_boot: int = 1000, which: str = "final", metrics=None,
                              plots_dir: str = "plots", save: bool = True) -> tuple:
    """One figure per benchmark metric vs. the metric_based threshold, for a sweep where
    `circuit.threshold` (not circuit.extension) was the swept dimension -- e.g. to see how an
    auto-selected knee/percolation threshold compares to a hand-swept range.

    Every metric in the benchmark suite that bench_df carries gets its own figure (mean +/- bootstrap
    SE per threshold, with the config-selected threshold_rule's value marked -- see
    _threshold_reference_values); `metrics` restricts that set. The sweep is fetched, benchmarked and
    bootstrapped ONCE and the reference value computed once, then reused for every figure. Metrics
    that are absent or all-NaN are skipped (e.g. bench_BAS/* on a JGB sweep, bench_val/* on a BAS
    full_support sweep).

    Not part of generate_all_figures -- that pipeline assumes circuit.extension grouping. Saves to
    plots_dir/<sweep_id>-threshold/, mirroring generate_all_figures' <sweep_id>-<dataset> convention.
    Returns (bench_df, {filename: figure}).
    """
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
        # anything benchmarked but not in the label map still gets a figure, under its raw name
        metrics += [m for m in bench_df.columns
                    if m in _LOWER_IS_BETTER and m not in _BENCH_METRIC_LABELS]
    references = _threshold_reference_values(next(iter(grouped.values()))[0])

    figs = {}
    for metric in metrics:
        if metric not in bench_df or not bench_df[metric].notna().any():
            continue
        filename = f"{metric.replace('/', '_')}_vs_threshold"
        figs[filename] = _render_metric_vs_threshold(bench_df, group_by, metric, references,
                                                    plots_dir, filename, save)
    print(f"[plotting]     {len(figs)} metric-vs-threshold figure(s): "
          f"{', '.join(sorted(figs))}")
    return bench_df, figs


def _render_metric_vs_threshold(bench_df, group_by, metric, references, plots_dir, filename, save):
    """Render one metric-vs-threshold figure; see plot_metrics_vs_threshold."""
    sub = bench_df[bench_df[metric].notna()]
    std_col = f"{metric}_std"

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

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    if not main.empty:
        ax.errorbar(main[group_by], main[metric], yerr=_errs(main), fmt="o", ms=4, ls="none",
                   capsize=2, color=OKABE_ITO["blue"], elinewidth=0.8)
    # collected as (threshold value, handle) so the legend below can be ordered by threshold --
    # linear (0.0) first, metric-based (the auto-selected reference value) in the middle, all-to-all
    # (1.0) last -- regardless of the order these artists were drawn in
    legend_entries = []
    for value, (label, color, marker) in _THRESHOLD_ENDPOINTS.items():
        row = ends[np.isclose(ends[group_by], value)]
        if row.empty:
            continue
        handle = ax.errorbar(row[group_by], row[metric], yerr=_errs(row), fmt=marker, ms=5, ls="none",
                            capsize=2, color=color, elinewidth=0.8, label=label, zorder=4)
        legend_entries.append((value, handle))
    for rule, value, color in references:
        # the rule (knee/percolation) is an implementation detail of HOW the threshold was
        # auto-selected; the line marks the metric-based extension's threshold, so it is labelled
        # the same way the topology figures name that extension, not by the internal rule name
        handle = ax.axvline(value, color=color, ls="--", lw=1, label=f"metric-based @ {value:.3f}")
        legend_entries.append((value, handle))
    if legend_entries:
        legend_entries.sort(key=lambda e: e[0])
        ax.legend([h for _, h in legend_entries], [h.get_label() for _, h in legend_entries],
                 fontsize=7, frameon=True)
    ax.set_xlabel("threshold")
    ax.set_ylabel(_BENCH_METRIC_LABELS.get(metric, metric))
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


# --------------------------------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------------------------------
# The per-step MMD targets logged by training (see qcbm.MMD_KEYS): each gets a "MMD" ylabel and a
# title of just the split name, instead of the generic metric-name-derived labeling (see
# plot_mmd_vs_measurements) -- any OTHER metric passed via `metrics` keeps that generic behavior.
# generate_all_figures plots the three single splits by default; the union targets are plotted when
# asked for explicitly (--metrics), and titled from here when they are.
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
    """Generate the training-dependent figure set for a sweep: metric-vs-measurements curves (train
    and val MMD combined into one side-by-side figure, see plot_train_val_mmd) + bootstrap benchmark
    (metric table, and the per-feature QQ figure for JGB). Both the curves and the benchmark aggregate ACROSS ALL
    SEED-RUNS of each group via bootstrap (mean +/- across-seed standard error); `n_boot` sets the
    number of bootstrap resamples. `which` selects which checkpoint of each run is
    benchmarked/sampled and defaults to "final" (last training iteration) rather than "best"
    (validation-selected) -- the val-selected checkpoint is not usable in this pipeline, where
    selection on mmd_val fires at initialization for most runs (best_iter == 1; see plot_qq_vs_data).
    Saves PDFs to plots_dir/<sweep_id>-<dataset>/, so figures from different sweeps/datasets never
    collide.

    <dataset> is read off the sweep's own run configs (detect_dataset), which also decides whether the
    JGB-only QQ figure is generated. `dataset_cfg` is optional and only cross-checked against that:
    passing {"dataset": ...} that disagrees with the runs warns and is ignored.

    `n_shots` is how many shots each checkpoint is sampled with for the benchmark suite and the JGB
    QQ figure. It sets the sampling-noise floor on every reported metric -- the model distribution is
    estimated from n_shots draws, so group differences smaller than that noise are not resolvable --
    and it is the dominant cost of a plotting pass (one simulation per run). Raise it when the
    across-seed error bars are small enough that shot noise dominates them.

    Static dataset/topology/threshold figures (SU(4) gate, preprocessing, threshold curve,
    extension heatmaps, topology networks) don't depend on training and are NOT generated here --
    see src.plot_extension, which is config-driven and only needs to be (re)run when the data or
    extension settings change, not on every sweep. For a sweep where circuit.threshold itself was
    varied, see plot_metrics_vs_threshold instead -- a different sweep shape, not generated here."""
    if science_style:
        use_science_style()
    print(f"[plotting] sweep={sweep_id} group_by={group_by} which={which} n_shots={n_shots:,}")

    # 1) metric-vs-measurements, bootstrapped over all seeds
    #
    # wandb is queried exactly twice per pass: once for the sweep's run list (cached in
    # src.benchmark and reused by the benchmark/QQ steps below) and once per run for the history of
    # ALL requested metrics at once.
    print("[plotting] (1/2) fetching runs from wandb...")
    grouped = fetch_runs(sweep_id, entity, project, group_by)
    n_runs = sum(len(v) for v in grouped.values())
    print(f"[plotting]     found {n_runs} runs across {len(grouped)} group(s): "
          f"{', '.join(str(k) for k in grouped)}")
    # the run list is also what the dataset label comes from, so the output directory can't disagree
    # with the sweep it was built from (the run configs are already cached by the fetch above)
    dataset = detect_dataset(grouped, expected=(dataset_cfg or {}).get("dataset"))
    plots_dir = os.path.join(plots_dir, f"{sweep_id}-{dataset}")
    print(f"[plotting]     dataset={dataset} -> {plots_dir}/")
    print(f"[plotting]     fetching histories ({len(metrics)} metric(s), one request per run)...")
    prefetch_histories(grouped, metrics)
    # train/val MMD are combined into one side-by-side figure (plot_train_val_mmd) rather than two
    # standalone ones, whenever both are present in `metrics` (true for the default tuple; a caller
    # that customizes `metrics` down to just one of the two still gets it as its own standalone plot
    # via the loop below).
    combine_train_val = {"train/mmd_train", "train/mmd_val"}.issubset(metrics)
    for metric in metrics:
        if combine_train_val and metric in ("train/mmd_train", "train/mmd_val"):
            continue
        print(f"[plotting]     plotting {metric} vs. measurements (bootstrap over seeds)...")
        split = _MMD_SPLIT_TITLES.get(metric)
        # metric names may contain "/" (e.g. "bench_dist/mmd"); flatten to "_" so the filename
        # doesn't imply a nested directory that was never created (plots_dir is the only dir made).
        plot_mmd_vs_measurements(grouped, metric=metric, n_boot=n_boot,
                                 log_axes=split is not None,
                                 ylabel="MMD" if split else None,
                                 title=split,
                                 filename=f"{metric.replace('/', '_')}_measurements", plots_dir=plots_dir)
    if combine_train_val:
        print("[plotting]     plotting train/val MMD side by side (bootstrap over seeds)...")
        plot_train_val_mmd(grouped, n_boot=n_boot, plots_dir=plots_dir)
    print("[plotting]     done.")

    # 2) benchmark figures: evaluate every run's checkpoint (per `which`), then bootstrap across seeds
    print("[plotting] (2/2) benchmarking all runs per group (bootstrap over seeds)...")
    # `grouped` is passed through so the benchmark reuses the run list fetched above instead of
    # re-querying the sweep (same for the point-estimate fallback and the QQ figure).
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
        # both QQ figures: one featured model per group, and the across-seed bootstrap band over all
        # of them (the latter carries no seed-selection choice -- see plot_qq_bootstrap_vs_data)
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
        description="Regenerate the training-dependent figures (MMD-vs-measurements, best-model "
                    "benchmark) for a wandb sweep and save them as PDF. For the static dataset/"
                    "topology/threshold figures, use `python -m src.plot_extension` instead.")
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
                             "the JGB QQ figure). Sets the sampling-noise floor on every reported "
                             "metric and is the dominant cost of a pass; raise it once the "
                             "across-seed error bars are smaller than the shot noise.")
    parser.add_argument("--plots-dir", default="plots", help="Output directory for the PDFs/PNGs.")
    parser.add_argument("--no-science-style", action="store_true",
                        help="Skip the scienceplots styling (use matplotlib defaults).")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    group_by = args.group_by or ("circuit.threshold" if args.threshold_sweep else "circuit.extension")

    if args.threshold_sweep:
        if not args.no_science_style:
            use_science_style()  # generate_all_figures does this itself; this branch bypasses it
        bench_df, _figs = plot_metrics_vs_threshold(
            sweep_id=args.sweep_id, entity=args.entity, project=args.project,
            group_by=group_by, n_shots=args.n_shots, n_boot=args.n_boot, which=args.which,
            plots_dir=args.plots_dir,
        )
        print(f"Figures written to {args.plots_dir}/{args.sweep_id}-threshold/")
        if bench_df is not None and not bench_df.empty:
            print(bench_df.to_string(index=False))
        return

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


if __name__ == "__main__":
    main()
