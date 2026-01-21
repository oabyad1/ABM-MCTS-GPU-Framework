"""
This script post processes the results from the rollout depth experiment

  • Filters to MCTS results only
  • Computes composite = w_area*norm_area + w_bldg*norm_bldg
  • Groups by rollout_depth_adjustment
  • Outputs:
      - plots/mcts_mean_composite_by_depth.png
      - plots/mcts_composite_vs_depth_scatter.png
      - plots/mcts_composite_vs_depth_scatter_colored.png
      - plots/mcts_mean_runtime_by_depth.png (if 'seconds' present)
      - plots/mcts_runtime_vs_depth_scatter.png (if 'seconds' present)
      - plots/mcts_runtime_vs_depth_scatter_colored.png (if 'seconds' present)
      - plots/mcts_by_depth_stats.csv


"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("summary_depth_camp_high_final.csv")
# CSV_PATH = Path("summary_depth_camp_med_final.csv")
# CSV_PATH = Path("summary_depth_marshall_high_final.csv")
# CSV_PATH = Path("summary_depth_marshall_med_final.csv")

OUTDIR = Path("plots_final_depth_camp_high")
COMPOSITE_WEIGHTS = (1 / 3, 2 / 3)  # (weight_area, weight_bldg), must sum to 1.0

DEPTH_SHIFT = 3

# Exclusions (exclude entire runs, regardless of seed)
EXCLUDE_RUNS: set[int] = set()      # e.g., {15, 16, 23}
# Optional: exclude specific (run,seed) pairs
EXCLUDE_RUN_UIDS: set[str] = set()  # e.g., {"15|42", "16|777"}

REQUIRED_COLS = {
    "run", "seed", "strategy", "area_score", "buildings_destroyed",
    "baseline_area", "baseline_buildings", "rollout_depth_adjustment"
}

FIGSIZE = (10, 6)
FIG_DPI = 200
SUBPLOT_ADJUST = dict(left=0.10, right=0.98, bottom=0.12, top=0.86)
COMPOSITE_YLIM = (0.0, 1.05)


plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})


def sem_or_std(x_like) -> float:
    x = pd.Series(x_like).dropna()
    if len(x) <= 1:
        return float(np.nan if len(x) == 0 else x.std(ddof=0))
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def load_and_clean(csv_path: Path,
                   exclude_runs: set[int] | None = None,
                   exclude_run_uids: set[str] | None = None) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    for c in ["area_score", "buildings_destroyed", "baseline_area", "baseline_buildings", "rollout_depth_adjustment"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "seconds" in df.columns:
        df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")

    df["run_uid"] = df["run"].astype(str) + "|" + df["seed"].astype(str)

    if exclude_runs:
        df = df[~df["run"].isin(exclude_runs)].copy()
    if exclude_run_uids:
        df = df[~df["run_uid"].isin(exclude_run_uids)].copy()

    def run_is_bad(g: pd.DataFrame) -> bool:
        has_nan_scores = g["area_score"].isna().any() or g["buildings_destroyed"].isna().any()
        has_nan_baselines = g["baseline_area"].isna().any() or g["baseline_buildings"].isna().any()
        has_zero_baselines = (g["baseline_area"] == 0).any() or (g["baseline_buildings"] == 0).any()
        return bool(has_nan_scores or has_nan_baselines or has_zero_baselines)

    bad_runs = df.groupby("run_uid").apply(run_is_bad)
    bad_uids = set(bad_runs[bad_runs].index)
    if bad_uids:
        df = df[~df["run_uid"].isin(bad_uids)].copy()

    df["strategy_lc"] = df["strategy"].astype(str).str.lower()
    uids_with_mcts = set(df.loc[df["strategy_lc"] == "mcts", "run_uid"].unique())
    df = df[df["run_uid"].isin(uids_with_mcts)].copy()

    if df.empty:
        raise RuntimeError("No usable rows after cleaning; check the CSV contents.")

    df["norm_area"] = df["area_score"] / df["baseline_area"]
    df["norm_bldg"] = df["buildings_destroyed"] / df["baseline_buildings"]

    w_area, w_bldg = COMPOSITE_WEIGHTS
    df["composite"] = w_area * df["norm_area"] + w_bldg * df["norm_bldg"]

    return df


def make_fig_ax():
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI)
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    return fig, ax


def finalize_and_save(fig, outpath: Path):
    fig.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)


def _enclose_bold_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color("black")


def _legend_top(ax, handles=None, ncol=5):
    if handles is None:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
                  ncol=ncol, frameon=False, handletextpad=0.4, columnspacing=1.0)
    else:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.15),
                  ncol=ncol, frameon=False, handletextpad=0.4, columnspacing=1.0)


def short_seed_label(seed: int) -> str:
    s = str(int(seed))
    return f"Seed …{s[-4:]}" if len(s) > 4 else f"Seed {s}"


def bar_with_error(values, errors, labels, outpath: Path, ylabel: str,
                   show_baseline: bool = True, xlabel: str = "Rollout Depth",
                   ylim=None):
    fig, ax = make_fig_ax()

    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=6)
    if show_baseline:
        ax.axhline(1.0, linestyle="--", linewidth=1, label="Baseline (=1.0)")

    ax.set_xticks(x, labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(-0.5, len(labels) - 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if show_baseline:
        _legend_top(ax, ncol=3)

    finalize_and_save(fig, outpath)


def scatter_points_basic(x_vals, y_vals, x_labels, outpath: Path, ylabel: str,
                         show_baseline: bool = True, xlabel: str = "Rollout Depth",
                         marker_size: float = 52, ylim=None):
    fig, ax = make_fig_ax()

    ax.scatter(x_vals, y_vals, alpha=0.78, s=marker_size, edgecolors="none")
    if show_baseline:
        ax.axhline(1.0, linestyle="--", linewidth=1, label="Baseline (=1.0)")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if show_baseline:
        _legend_top(ax, ncol=3)

    finalize_and_save(fig, outpath)


def scatter_points_colored_by_seed(
    x_vals, y_vals, seeds, x_labels, outpath: Path, ylabel: str,
    show_baseline: bool = True, xlabel: str = "Rollout Depth",
    annotate_counts: bool = False, max_legend_seeds: int | None = None,
    marker_size: float = 58, ylim=None
):
    fig, ax = make_fig_ax()

    rng = np.random.default_rng(42)
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    seeds = np.asarray(seeds)

    jitter = rng.normal(0.0, 0.03, size=len(x_vals))
    xj = x_vals + jitter

    unique_seeds = np.array(sorted(pd.unique(seeds)))
    cmap = plt.get_cmap("tab20")
    color_map = {s: cmap(i % 20) for i, s in enumerate(unique_seeds)}
    colors = [color_map[s] for s in seeds]

    ax.scatter(xj, y_vals, c=colors, alpha=0.85, s=marker_size, edgecolors="none", linewidths=0)
    if show_baseline:
        ax.axhline(1.0, linestyle="--", linewidth=1, label="Baseline (=1.0)")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)

    handles = []
    if show_baseline:
        handles.append(plt.Line2D([0], [0], color="black", linestyle="--", label="Baseline (=1.0)"))

    seed_list_for_legend = unique_seeds
    if max_legend_seeds is not None and len(seed_list_for_legend) > max_legend_seeds:
        seed_list_for_legend = seed_list_for_legend[:max_legend_seeds]

    for s in seed_list_for_legend:
        handles.append(plt.Line2D([0], [0], marker="o", color=color_map[s],
                                  linestyle="None", label=short_seed_label(s)))

    if len(seed_list_for_legend) < len(unique_seeds):
        handles.append(plt.Line2D([0], [0], linestyle="None", label="…more"))

    if handles:
        _legend_top(ax, handles=handles, ncol=min(6, len(handles)))

    if annotate_counts and len(x_vals):
        import collections
        depth_counts = collections.Counter(x_vals.astype(int))
        ymin, ymax = ax.get_ylim()
        y_text = ymax - 0.03 * (ymax - ymin)
        for x_idx, n in depth_counts.items():
            ax.text(x_idx, y_text, f"n={n}", ha="center", va="top", fontsize=9)

    finalize_and_save(fig, outpath)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(
        CSV_PATH,
        exclude_runs=EXCLUDE_RUNS,
        exclude_run_uids=EXCLUDE_RUN_UIDS
    )

    mcts = df[df["strategy_lc"] == "mcts"].copy()
    if mcts.empty:
        raise RuntimeError("No MCTS rows found after cleaning.")

    mcts = mcts.dropna(subset=["rollout_depth_adjustment"]).copy()

    mcts["rollout_depth_display"] = mcts["rollout_depth_adjustment"] + DEPTH_SHIFT

    agg_dict = dict(
        n_samples=("composite", "count"),
        n_run_uids=("run_uid", "nunique"),
        mean_norm_area=("norm_area", "mean"),
        se_norm_area=("norm_area", sem_or_std),
        mean_norm_bldg=("norm_bldg", "mean"),
        se_norm_bldg=("norm_bldg", sem_or_std),
        mean_composite=("composite", "mean"),
        se_composite=("composite", sem_or_std),
    )
    if "seconds" in mcts.columns:
        agg_dict.update(
            mean_seconds=("seconds", "mean"),
            se_seconds=("seconds", sem_or_std),
        )

    by_depth = (
        mcts.groupby("rollout_depth_display")
            .agg(**agg_dict)
            .sort_index()
    )

    by_depth_path = OUTDIR / "mcts_by_depth_stats.csv"
    by_depth.to_csv(by_depth_path, index=True)

    depths = by_depth.index.tolist()
    labels = [str(d) for d in depths]

    per_seed_depth = (
        mcts.groupby(["seed", "rollout_depth_display"], as_index=False)
            .agg(
                composite=("composite", "mean"),
                **({"seconds": ("seconds", "mean")} if "seconds" in mcts.columns else {})
            )
            .sort_values(["rollout_depth_display", "seed"])
    )

    depth_to_x = {d: i for i, d in enumerate(depths)}
    xs = per_seed_depth["rollout_depth_display"].map(depth_to_x).values

    composite_ylim = COMPOSITE_YLIM

    runtime_ylim = None
    if "seconds" in mcts.columns and not mcts["seconds"].dropna().empty:
        runtime_mins_all = (per_seed_depth["seconds"].values / 60.0)
        rmin = float(np.nanmin(runtime_mins_all))
        rmax = float(np.nanmax(runtime_mins_all))
        pad = 0.05 * (rmax - rmin if rmax > rmin else 1.0)
        runtime_ylim = (max(0.0, rmin - pad), rmax + pad)

    bar_with_error(
        by_depth["mean_composite"].values,
        by_depth["se_composite"].values,
        labels,
        outpath=OUTDIR / "mcts_mean_composite_by_depth.png",
        ylabel="Composite Reward",
        show_baseline=True,
        ylim=composite_ylim
    )

    scatter_points_basic(
        xs,
        per_seed_depth["composite"].values,
        x_labels=labels,
        outpath=OUTDIR / "mcts_composite_vs_depth_scatter.png",
        ylabel="Composite Reward",
        show_baseline=True,
        marker_size=52,
        ylim=composite_ylim
    )

    scatter_points_colored_by_seed(
        xs,
        per_seed_depth["composite"].values,
        per_seed_depth["seed"].values,
        x_labels=labels,
        outpath=OUTDIR / "mcts_composite_vs_depth_scatter_colored.png",
        ylabel="Composite Reward",
        show_baseline=True,
        marker_size=58,
        ylim=composite_ylim,
        annotate_counts=False,
        max_legend_seeds=None
    )

    if "seconds" in mcts.columns and not mcts["seconds"].dropna().empty:
        mins_mean = by_depth["mean_seconds"].values / 60.0
        mins_se = by_depth["se_seconds"].values / 60.0
        per_seed_mins = per_seed_depth["seconds"].values / 60.0

        bar_with_error(
            mins_mean,
            mins_se,
            labels,
            outpath=OUTDIR / "mcts_mean_runtime_by_depth.png",
            ylabel="Runtime (minutes)",
            show_baseline=False,
            ylim=runtime_ylim
        )

        scatter_points_basic(
            xs,
            per_seed_mins,
            x_labels=labels,
            outpath=OUTDIR / "mcts_runtime_vs_depth_scatter.png",
            ylabel="Runtime (minutes)",
            show_baseline=False,
            marker_size=52,
            ylim=runtime_ylim
        )

        scatter_points_colored_by_seed(
            xs,
            per_seed_mins,
            per_seed_depth["seed"].values,
            x_labels=labels,
            outpath=OUTDIR / "mcts_runtime_vs_depth_scatter_colored.png",
            ylabel="Runtime (minutes)",
            show_baseline=False,
            marker_size=58,
            ylim=runtime_ylim,
            annotate_counts=False,
            max_legend_seeds=None
        )
    else:
        print("\nNote: 'seconds' column not found or empty; skipping runtime plots.")

    with pd.option_context("display.max_columns", 20, "display.width", 140):
        print("\n=== MCTS by depth (lower is better; sorted by depth) ===")
        print(by_depth.to_string())

    print(f"\nWrote outputs to: {OUTDIR.resolve()}")
    print(f"- {by_depth_path.name}")
    print("- mcts_mean_composite_by_depth.png")
    print("- mcts_composite_vs_depth_scatter.png")
    print("- mcts_composite_vs_depth_scatter_colored.png")
    if "seconds" in mcts.columns and not mcts["seconds"].dropna().empty:
        print("- mcts_mean_runtime_by_depth.png")
        print("- mcts_runtime_vs_depth_scatter.png")
        print("- mcts_runtime_vs_depth_scatter_colored.png")


if __name__ == "__main__":
    main()
