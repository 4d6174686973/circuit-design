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
    generate_all_figures(sweep_id="<sweep>", entity="<you>", project="qcbm-circuit-design",
                         dataset_cfg={"dataset": "BAS"})
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from src.data import JGB, DataLoader
from src.config_schema import from_run_config
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
    "nearest_neighbor": "nearest-neighbor",
    "random": "random",
    "metric_based": "metric-based",
    "chow_liu": "chow-liu",
    "all_to_all": "all-to-all",
}

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

# fixed roles for the recurring extension categories, so their color stays constant across figures
# regardless of which/how-many other keys are present
_ROLE_COLORS = {
    "metric-based": OKABE_ITO["orange"],
    "chow-liu": OKABE_ITO["bluish_green"],
    "nearest-neighbor": OKABE_ITO["sky_blue"],
}
# fixed draw order for any remaining (non-role) categorical keys
_CB_CYCLE = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["reddish_purple"],
            OKABE_ITO["yellow"], OKABE_ITO["orange"], OKABE_ITO["bluish_green"],
            OKABE_ITO["sky_blue"]]

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
    palette for named/extra keys, the shared SEQUENTIAL_CMAP (cividis) for numeric sweeps."""
    keys = list(legend_keys)
    # numeric sweep dimension -> sequential colormap ordered by value
    try:
        numeric = sorted(keys, key=lambda k: float(k))
        shades = plt.get_cmap(SEQUENTIAL_CMAP)(np.linspace(0.15, 0.9, len(numeric)))
        return {k: shades[i] for i, k in enumerate(numeric)}
    except (TypeError, ValueError):
        pass
    colors = {}
    ci = 0
    for k in keys:
        label = _EXTENSION_LABELS.get(k, k)
        if label in _ROLE_COLORS:
            colors[k] = _ROLE_COLORS[label]
        else:
            colors[k] = _CB_CYCLE[ci % len(_CB_CYCLE)]
            ci += 1
    return colors


def _save(fig, plots_dir, filename):
    path = f"{plots_dir}/{filename}"
    os.makedirs(os.path.dirname(path), exist_ok=True)  # filename may itself contain "/"
    fig.savefig(f"{path}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(f"{path}.png", bbox_inches="tight", transparent=False, dpi=300)


# --------------------------------------------------------------------------------------------------
# wandb fetch + aggregation
# --------------------------------------------------------------------------------------------------
def fetch_runs(sweep_id: str, entity: str, project: str, group_by: str = "circuit.extension",
               filters: dict = None) -> dict:
    """Return {legend_key: [runs]} for a real wandb Sweep, grouped by a config dimension.

    group_by and filters keys are dot-separated paths into the run's config, e.g. "circuit.extension".
    filters is an optional {config_key: value} dict to pin non-legend swept params (facet slicing),
    applied client-side since Sweep.runs is a materialized list, not a server-side query.

    Runs whose logged config predates the current schema (e.g. from before a field was renamed or
    regrouped) are skipped with a warning rather than aborting the whole fetch.
    """
    import wandb
    api = wandb.Api()
    entity = entity or api.default_entity  # unresolved None would literally build ".../None/..."
    runs = []
    for r in api.sweep(f"{entity}/{project}/{sweep_id}").runs:
        try:
            runs.append((r, from_run_config(r.config)))
        except Exception as e:
            print(f"[plotting]     skipping run {r.id}: config incompatible with current schema ({e})")
    if filters:
        runs = [(r, cfg) for r, cfg in runs
                if all(OmegaConf.select(cfg, k) == v for k, v in filters.items())]
    grouped = {}
    for r, cfg in runs:
        grouped.setdefault(OmegaConf.select(cfg, group_by), []).append(r)
    return grouped


def fetch_history(run, metric: str = "train/mmd_train") -> pd.DataFrame:
    """Per-run dataframe with train/cumulative_measurements and the requested metric (NaN rows dropped)."""
    keys = ["train/cumulative_measurements", metric]
    df = run.history(keys=keys, pandas=True)
    if df is None or df.empty or metric not in df:
        cfg = from_run_config(run.config)
        n = run.summary.get("iterations_run") or cfg.qcbm.iterations
        P = run.summary.get("train/num_parameters") or run.config.get("num_parameters", 0)
        shots = cfg.qcbm.N_shots
        per = run.summary.get("measurements_per_step") or (2 * P + 1) * shots
        df = pd.DataFrame({"train/cumulative_measurements": np.arange(1, n + 1) * per,
                           metric: [np.nan] * n})
    return df.dropna(subset=[metric]).sort_values("train/cumulative_measurements")


def aggregate_over_measurements(histories: list, metric: str, mode: str = "bootstrap",
                                n_grid: int = 400, window: int = 1, n_boot: int = 1000,
                                boot_seed: int = 0) -> dict:
    """Interpolate each run onto a common measurement grid, then aggregate across seeds.

    Different legend keys have different measurements/iteration, so runs are not index-aligned; we
    interpolate onto a shared x-grid before aggregating. `mode`:
      - "bootstrap" (default): at each grid point, resample the runs with replacement n_boot times;
        `line` is the mean of the bootstrap means and the band is +/- the bootstrap std of the mean
        (the across-seed standard error), via utils.bootstrap_mean_std.
      - "meanstd": plain across-run mean +/- std.
      - "medperc": median with the 10th/90th percentile band.
    """
    curves = [h for h in histories if len(h) > 0]
    if not curves:
        return None
    lo = max(c["train/cumulative_measurements"].min() for c in curves)
    hi = min(c["train/cumulative_measurements"].max() for c in curves)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, n_grid)
    stacked = np.vstack([
        np.interp(grid, c["train/cumulative_measurements"].values, c[metric].values) for c in curves
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


def plot_mmd_vs_measurements(runs_by_key: dict, metric: str = "train/mmd_train", mode: str = "bootstrap",
                             window: int = 1, n_boot: int = 1000, colors: dict = None,
                             filename: str = "MMD_measurements", plots_dir: str = "plots",
                             save: bool = True):
    """MMD (or any logged metric) vs cumulative measurements, aggregated across seeds per group.

    Defaults to the bootstrap aggregation (mean +/- across-seed standard error); see
    aggregate_over_measurements for the other modes. The shaded band is the bootstrap std of the mean.
    """
    colors = colors or default_colors(runs_by_key.keys())
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for key, runs in runs_by_key.items():
        hists = [fetch_history(r, metric) for r in runs]
        agg = aggregate_over_measurements(hists, metric, mode, window=window, n_boot=n_boot)
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
# benchmark figures (bootstrap across all runs per group)
# --------------------------------------------------------------------------------------------------
def bootstrap_group_metrics(per_run_df: pd.DataFrame, group_by: str = "circuit.extension",
                            metric_cols: list = None, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Bootstrap held-out benchmark metrics across the runs of each group.

    `per_run_df` is the tidy one-row-per-run table from benchmark.benchmark_all_runs (a `group_by`
    column plus numeric metric columns). For each group and metric, resample the group's runs with
    replacement n_boot times and report the mean and the std of the bootstrap means (the across-seed
    standard error), via utils.bootstrap_mean_std. Non-finite per-run values (e.g. bench_val/* for a
    full_support run) are dropped before resampling.

    Returns one row per group with, for each metric <m>, a column <m> (bootstrap mean) and <m>_std
    (bootstrap SE), plus n_runs. Column order follows per_run_df.
    """
    if per_run_df is None or per_run_df.empty:
        return None
    if metric_cols is None:
        skip = {group_by, "run_id", "run_name"}
        metric_cols = [c for c in per_run_df.columns
                       if c not in skip and np.issubdtype(per_run_df[c].dropna().dtype, np.number)]
    rows = []
    for key, sub in per_run_df.groupby(group_by, sort=False):
        row = {group_by: key, "n_runs": len(sub)}
        for c in metric_cols:
            vals = sub[c].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                row[c] = row[f"{c}_std"] = np.nan
            else:
                mean, std = bootstrap_mean_std(vals, n_boot=n_boot, seed=seed)
                row[c], row[f"{c}_std"] = float(mean), float(std)
        rows.append(row)
    return pd.DataFrame(rows)


def _cell_text(row, col, precision: int) -> str:
    """Format a benchmark-table cell as "mean ± std" when a companion <col>_std column is present
    (bootstrap table), else just the rounded mean; an em dash for missing/NaN values."""
    mean = row.get(col, np.nan)
    if not np.isfinite(mean):
        return "—"
    std_col = f"{col}_std"
    if std_col in row.index and np.isfinite(row.get(std_col, np.nan)):
        return f"{mean:.{precision}f} ± {row[std_col]:.{precision}f}"
    return f"{mean:.{precision}f}"


def plot_metric_table(bench_df: pd.DataFrame, metric_cols: list = None, group_by: str = "circuit.extension",
                      plots_dir: str = "plots", filename: str = "benchmark_table", save: bool = True,
                      precision: int = 4):
    """Table of benchmark metrics across groups (one row per group).

    Works for both the bootstrap table (bootstrap_group_metrics: cells shown as "mean ± std" using
    the <metric>_std companion columns) and the point-estimate table (benchmark_sweep: bare means).
    Renders as a matplotlib table (PDF/PNG) and writes the full-precision bench_df alongside as CSV,
    since a rendered table is for reading, not downstream analysis.
    """
    if bench_df is None or bench_df.empty:
        return None
    if metric_cols is None:
        metric_cols = [c for c in ["bench_dist/test/mmd", "bench_dist/test/tv", "bench_dist/test/fidelity",
                                   "bench_val/coverage", "bench_val/fidelity", "bench_val/rate",
                                   "bench_BAS/precision", "bench_BAS/recall", "bench_BAS/qbas"]
                       if c in bench_df]
    metric_cols = [c for c in metric_cols if c in bench_df]
    if not metric_cols:
        return None

    labels = [_EXTENSION_LABELS.get(k, str(k)) for k in bench_df[group_by]]
    cell_text = [[_cell_text(row, c, precision) for c in metric_cols]
                 for _, row in bench_df.iterrows()]

    if save:
        os.makedirs(plots_dir, exist_ok=True)
        bench_df.to_csv(f"{plots_dir}/{filename}.csv", index=False)  # full table incl. *_std, n_runs

    n_rows, n_cols = len(cell_text), len(metric_cols)
    fig, ax = plt.subplots(figsize=(1.7 * (n_cols + 1) + 1, 0.4 * (n_rows + 1) + 0.5))
    ax.axis("off")
    table = ax.table(cellText=cell_text, rowLabels=labels, colLabels=metric_cols,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


def plot_generalization_bars(bench_df: pd.DataFrame, group_by: str = "circuit.extension",
                             metrics=("bench_val/coverage", "bench_val/fidelity", "bench_val/rate"),
                             plots_dir: str = "plots", filename: str = "generalization_metrics",
                             colors: dict = None, save: bool = True):
    """Grouped bar chart of the Gili et al. validity-based generalization metrics.

    One bar cluster per metric (coverage, fidelity, rate), one bar per group (extension).
    Computed for both BAS and JGB (see benchmark.generalization_metrics); only draws metrics present
    in bench_df, returning None when none are present or all values are NaN -- i.e. for BAS
    `full_support` sweeps, where the unseen valid space is empty and every metric is undefined
    (these metrics require training on a strict subset of the valid space, i.e.
    bas_split_mode=holdout). For JGB, `fidelity` is trivially 1.0 and `rate` collapses to
    `exploration` -- every bitstring decodes to a valid value, so only `exploration`/`coverage`
    carry signal there.

    Sources: Gili, Mauri & Perdomo-Ortiz, arXiv:2207.13645 (Quantum Sci. Technol. 8, 035021, 2023);
    Gili et al., Phys. Rev. Applied 21, 044032 (2024), arXiv:2201.08770.
    """
    if bench_df is None or bench_df.empty:
        return None
    metrics = [m for m in metrics if m in bench_df]
    if not metrics:
        return None
    sub = bench_df[[group_by, *metrics]].copy()
    sub = sub[sub[metrics].notna().any(axis=1)].reset_index(drop=True)  # drop undefined groups
    if sub.empty:
        return None

    keys = list(sub[group_by])
    labels = [_EXTENSION_LABELS.get(k, str(k)) for k in keys]
    colors = colors or default_colors(keys)
    n_groups, n_metrics = len(keys), len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_groups, 1)

    fig, ax = plt.subplots(1, 1, figsize=(1.3 * n_metrics + 1.5, 3))
    for i, (k, lab) in enumerate(zip(keys, labels)):
        vals = [sub.loc[i, m] for m in metrics]
        # bootstrap SE error bars when the <metric>_std companion columns are present
        errs = [sub.loc[i, f"{m}_std"] if f"{m}_std" in sub.columns else np.nan for m in metrics]
        errs = errs if np.any(np.isfinite(errs)) else None
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=lab, color=colors.get(k),
               yerr=errs, capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in metrics])
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("Generalization (unseen valid space)")
    ax.legend(loc="upper right", fontsize=6)
    plt.tight_layout()
    if save:
        _save(fig, plots_dir, filename)
    return fig


def plot_qq_grid(sweep_id: str, entity: str, project: str, group_by: str = "circuit.extension",
                 n_shots: int = 10000, n_q: int = 100, plots_dir: str = "plots", save: bool = True):
    """Per-group QQ plots (model vs data, model vs normal, data vs normal) for JGB best models."""
    grouped = fetch_runs(sweep_id, entity, project, group_by)
    figs = {}
    for key, runs in grouped.items():
        best = bm.select_best_run(runs)
        if best is None:
            continue
        cfg = from_run_config(best.config)
        if cfg.data.dataset != "JGB":
            continue
        print(f"[plotting]     [{key}] QQ plots from best run {best.id}...")
        circuit, params = bm.load_checkpoint(best)
        samples = bm.sample_model(circuit, params, n_shots, seed=cfg.sweep.random_seed)
        splits, _, _ = bm._test_split_for_config(cfg)
        jgb = JGB(cfg.data.N_qubits, cfg.data.N_features); dl = DataLoader(jgb)
        dl.train_val_test_split(cfg.data.train_split, cfg.data.val_split)
        xmin, xmax = dl.conv_min_max
        bpf = jgb.bits_per_feature
        feats = bm.reconstruct_features(samples, bpf, cfg.data.N_features, xmin, xmax)
        data = jgb.decimal.values
        fig, axs = plt.subplots(1, cfg.data.N_features, figsize=(3 * cfg.data.N_features, 3))
        for i, ax in enumerate(np.atleast_1d(axs)):
            mv, mp = feats[i]
            dx, my = bm.qq_model_vs_data(mv, mp, data[:, i], n_q)
            nx_, ny = bm.qq_model_vs_normal(mv, mp, n_q)
            ndx, ndy = bm.qq_data_vs_normal(data[:, i], n_q)
            ax.plot(dx, my, ".", ms=3, label="model vs data", color=OKABE_ITO["blue"])
            ax.plot(nx_, ny, ".", ms=3, label="model vs normal", color=OKABE_ITO["vermillion"])
            ax.plot(ndx, ndy, ".", ms=3, label="data vs normal", color=OKABE_ITO["bluish_green"])
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
                         group_by: str = "circuit.extension", metrics=("train/mmd_train", "bench_dist/test/mmd"),
                         plots_dir: str = "plots", science_style: bool = True, n_boot: int = 1000):
    """Generate the training-dependent figure set for a sweep: metric-vs-measurements curves +
    bootstrap benchmark (metric table, and QQ grids for JGB). Both the curves and the benchmark
    aggregate ACROSS ALL SEED-RUNS of each group via bootstrap (mean +/- across-seed standard
    error); `n_boot` sets the number of bootstrap resamples. Saves PDFs to
    plots_dir/<sweep_id>-<dataset>/, so figures from different sweeps/datasets never collide.

    Static dataset/topology/threshold figures (SU(4) gate, preprocessing, threshold curve,
    extension heatmaps, topology networks) don't depend on training and are NOT generated here --
    see src.plot_extension, which is config-driven and only needs to be (re)run when the data or
    extension settings change, not on every sweep."""
    if science_style:
        use_science_style()
    dataset = dataset_cfg.get("dataset", "BAS")
    plots_dir = os.path.join(plots_dir, f"{sweep_id}-{dataset}")

    print(f"[plotting] sweep={sweep_id} dataset={dataset} group_by={group_by} -> {plots_dir}/")

    # 1) metric-vs-measurements, bootstrapped over all seeds
    print("[plotting] (1/2) fetching runs from wandb...")
    grouped = fetch_runs(sweep_id, entity, project, group_by)
    n_runs = sum(len(v) for v in grouped.values())
    print(f"[plotting]     found {n_runs} runs across {len(grouped)} group(s): "
          f"{', '.join(str(k) for k in grouped)}")
    for metric in metrics:
        print(f"[plotting]     plotting {metric} vs. measurements (bootstrap over seeds)...")
        # metric names may contain "/" (e.g. "bench_dist/test/mmd"); flatten to "_" so the filename
        # doesn't imply a nested directory that was never created (plots_dir is the only dir made).
        plot_mmd_vs_measurements(grouped, metric=metric, n_boot=n_boot,
                                 filename=f"{metric.replace('/', '_')}_measurements", plots_dir=plots_dir)
    print("[plotting]     done.")

    # 2) benchmark figures: evaluate every run's best checkpoint, then bootstrap across seeds
    print("[plotting] (2/2) benchmarking all runs per group (bootstrap over seeds)...")
    per_run = bm.benchmark_all_runs(sweep_id, entity, project, group_by)
    bench_df = bootstrap_group_metrics(per_run, group_by=group_by, n_boot=n_boot)
    if bench_df is None or bench_df.empty:
        # e.g. no run had a usable checkpoint -> fall back to the best-per-group point estimate
        print("[plotting]     no per-run metrics; falling back to best-per-group point estimate.")
        bench_df = bm.benchmark_sweep(sweep_id, entity, project, group_by)
    plot_metric_table(bench_df, group_by=group_by, plots_dir=plots_dir)
    # Gili et al. generalization bars: computed for both datasets (no-op if bench_val/* columns are
    # absent or all-NaN, e.g. a BAS full_support sweep -- see plot_generalization_bars docstring)
    plot_generalization_bars(bench_df, group_by=group_by, plots_dir=plots_dir)
    if dataset == "JGB":
        plot_qq_grid(sweep_id, entity, project, group_by, plots_dir=plots_dir)
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
    parser.add_argument("--dataset", choices=["BAS", "JGB"], default="BAS")
    parser.add_argument("--group-by", default="circuit.extension",
                        help="Dot-separated config key to use as the plot legend/grouping dimension "
                             "(default: circuit.extension; can be any swept key).")
    parser.add_argument("--metrics", nargs="+", default=["train/mmd_train", "bench_dist/test/mmd"],
                        help="Logged metrics to plot vs. cumulative measurements.")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap resamples for the across-seed mean/SE (curves + benchmark).")
    parser.add_argument("--plots-dir", default="plots", help="Output directory for the PDFs/PNGs.")
    parser.add_argument("--no-science-style", action="store_true",
                        help="Skip the scienceplots styling (use matplotlib defaults).")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    dataset_cfg = {"dataset": args.dataset}

    bench_df = generate_all_figures(
        sweep_id=args.sweep_id,
        entity=args.entity,
        project=args.project,
        dataset_cfg=dataset_cfg,
        group_by=args.group_by,
        metrics=tuple(args.metrics),
        plots_dir=args.plots_dir,
        science_style=not args.no_science_style,
        n_boot=args.n_boot,
    )
    print(f"Figures written to {args.plots_dir}/{args.sweep_id}-{args.dataset}/")
    if bench_df is not None and not bench_df.empty:
        print(bench_df.to_string(index=False))


if __name__ == "__main__":
    main()
