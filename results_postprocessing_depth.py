
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


# CSV_PATH = Path("summary_depth_camp_high_final.csv")
# CSV_PATH = Path("summary_depth_camp_med_final.csv")
CSV_PATH = Path("summary_depth_marshall_high_final.csv")
# CSV_PATH = Path("summary_depth_marshall_med_final.csv")

# CSV_PATH = Path("summary_depth_med.csv")
# CSV_PATH = Path("summary_depth_high.csv")

OUTDIR   = Path("plots_final_depth_marshall_high")
COMPOSITE_WEIGHTS = (1/3, 2/3)  # (weight_area, weight_bldg), must sum to 1.0

# Exclusions (exclude entire runs, regardless of seed)
EXCLUDE_RUNS: set[int] = set()      # e.g., {15, 16, 23}
# Optional: exclude specific (run,seed) pairs
EXCLUDE_RUN_UIDS: set[str] = set()  # e.g., {"15|42", "16|777"}

REQUIRED_COLS = {
    "run", "seed", "strategy", "area_score", "buildings_destroyed",
    "baseline_area", "baseline_buildings", "rollout_depth_adjustment"
}


plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 200,
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

    # Ensure numeric, coerce "N/A" etc. to NaN
    for c in ["area_score", "buildings_destroyed", "baseline_area", "baseline_buildings", "rollout_depth_adjustment"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "seconds" in df.columns:
        df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")

    # Robust run id = (run, seed)
    df["run_uid"] = df["run"].astype(str) + "|" + df["seed"].astype(str)

    # Apply exclusions early
    if exclude_runs:
        df = df[~df["run"].isin(exclude_runs)].copy()
    if exclude_run_uids:
        df = df[~df["run_uid"].isin(exclude_run_uids)].copy()

    # Drop an entire run if it has any NaN scores/baselines or zero baselines
    def run_is_bad(g: pd.DataFrame) -> bool:
        has_nan_scores = g["area_score"].isna().any() or g["buildings_destroyed"].isna().any()
        has_nan_baselines = g["baseline_area"].isna().any() or g["baseline_buildings"].isna().any()
        has_zero_baselines = (g["baseline_area"] == 0).any() or (g["baseline_buildings"] == 0).any()
        return bool(has_nan_scores or has_nan_baselines or has_zero_baselines)

    bad_runs = df.groupby("run_uid").apply(run_is_bad)
    bad_uids = set(bad_runs[bad_runs].index)
    if bad_uids:
        df = df[~df["run_uid"].isin(bad_uids)].copy()

    # Require MCTS present in a run for head-to-head parity
    df["strategy_lc"] = df["strategy"].astype(str).str.lower()
    uids_with_mcts = set(df.loc[df["strategy_lc"] == "mcts", "run_uid"].unique())
    df = df[df["run_uid"].isin(uids_with_mcts)].copy()

    if df.empty:
        raise RuntimeError("No usable rows after cleaning; check the CSV contents.")

    # Normalized metrics (lower is better; 1.0 == baseline)
    df["norm_area"] = df["area_score"] / df["baseline_area"]
    df["norm_bldg"] = df["buildings_destroyed"] / df["baseline_buildings"]

    w_area, w_bldg = COMPOSITE_WEIGHTS
    df["composite"] = w_area * df["norm_area"] + w_bldg * df["norm_bldg"]

    return df


def _enclose_bold_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color("black")

def _legend_top(ax, handles=None, ncol=4):
    if handles is None:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=ncol, frameon=False)
    else:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                  ncol=ncol, frameon=False)

def bar_with_error(values, errors, labels, outpath: Path, ylabel: str,
                   show_baseline: bool = True, xlabel: str = "Rollout Depth"):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()


    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=6)
    if show_baseline:
        ax.axhline(1.0, linestyle="--", linewidth=1, label="Baseline (=1.0)")

    ax.set_xticks(x, labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Only add legend if there's something to show
    if show_baseline:
        _legend_top(ax, ncol=3)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def scatter_points_basic(x_vals, y_vals, x_labels, outpath: Path, ylabel: str,
                         show_baseline: bool = True, xlabel: str = "Rollout Depth",
                         marker_size: float = 52):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()


    ax.scatter(x_vals, y_vals, alpha=0.78, s=marker_size, edgecolors="none")
    if show_baseline:
        ax.axhline(1.0, linestyle="--", linewidth=1, label="Baseline (=1.0)")

    ax.set_xticks(sorted(set(x_vals)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_baseline:
        _legend_top(ax, ncol=3)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def scatter_points_colored_by_seed(
    x_vals, y_vals, seeds, x_labels, outpath: Path, ylabel: str,
    show_baseline: bool = True, xlabel: str = "Rollout Depth",
    annotate_counts: bool = True, max_legend_seeds: int | None = None,
    marker_size: float = 58
):
    """
    Colors points by random seed and places the legend where the title would be.
    If there are many unique seeds, cap legend entries with max_legend_seeds.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()


    rng = np.random.default_rng(42)
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    seeds = np.asarray(seeds)

    jitter = rng.normal(0.0, 0.03, size=len(x_vals))
    xj = x_vals + jitter

    # Color map per unique seed
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

    # Build legend entries (seeds; include baseline only if shown)
    handles = []
    if show_baseline:
        handles.append(plt.Line2D([0], [0], color="black", linestyle="--", label="Baseline (=1.0)"))

    seed_list_for_legend = unique_seeds
    if max_legend_seeds is not None and len(seed_list_for_legend) > max_legend_seeds:
        seed_list_for_legend = seed_list_for_legend[:max_legend_seeds]

    for s in seed_list_for_legend:
        handles.append(plt.Line2D([0], [0], marker="o", color=color_map[s],
                                  linestyle="None", label=f"Seed {s}"))

    if len(seed_list_for_legend) < len(unique_seeds):
        handles.append(plt.Line2D([0], [0], linestyle="None", label="…more seeds"))

    if handles:
        _legend_top(ax, handles=handles, ncol=min(6, len(handles)))

    # Optional: annotate counts above each depth
    if annotate_counts and len(x_vals):
        import collections
        depth_counts = collections.Counter(x_vals)
        ymin, ymax = np.nanmin(y_vals), np.nanmax(y_vals)
        ypad = 0.03 * (ymax - ymin if ymax > ymin else 1.0)
        for x_idx, n in depth_counts.items():
            ax.text(x_idx, ymax + ypad, f"n={n}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(
        CSV_PATH,
        exclude_runs=EXCLUDE_RUNS,
        exclude_run_uids=EXCLUDE_RUN_UIDS
    )

    # Focus on MCTS only
    mcts = df[df["strategy_lc"] == "mcts"].copy()
    if mcts.empty:
        raise RuntimeError("No MCTS rows found after cleaning.")

    # Clean depth values (drop NaNs)
    mcts = mcts.dropna(subset=["rollout_depth_adjustment"]).copy()

    # Summary by depth
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
        mcts.groupby("rollout_depth_adjustment")
            .agg(**agg_dict)
            .sort_index()
    )

    # Save stats table
    by_depth_path = OUTDIR / "mcts_by_depth_stats.csv"
    by_depth.to_csv(by_depth_path, index=True)

    # --- Bar plot: mean composite by depth (±SE); lower is better ---
    depths = by_depth.index.tolist()
    labels = [str(d) for d in depths]
    bar_with_error(
        by_depth["mean_composite"].values,
        by_depth["se_composite"].values,
        labels,
        outpath=OUTDIR / "mcts_mean_composite_by_depth.png",
        ylabel="Composite Reward",
        show_baseline=True,
    )

    # --- Per-seed means by depth (for seed legend) ---
    per_seed_depth = (
        mcts.groupby(["seed", "rollout_depth_adjustment"], as_index=False)
            .agg(
                composite=("composite", "mean"),
                **({"seconds": ("seconds", "mean")} if "seconds" in mcts.columns else {})
            )
            .sort_values(["rollout_depth_adjustment", "seed"])
    )

    # Map depth to ordinal x position
    depth_to_x = {d: i for i, d in enumerate(depths)}
    xs = per_seed_depth["rollout_depth_adjustment"].map(depth_to_x).values

    # Scatter: per-seed composite vs depth (plain)
    scatter_points_basic(
        xs,
        per_seed_depth["composite"].values,
        x_labels=labels,
        outpath=OUTDIR / "mcts_composite_vs_depth_scatter.png",
        ylabel="Composite Reward",
        show_baseline=True,
        marker_size=52,   # larger circles
    )

    # Scatter: colored by seed with legend at the top
    scatter_points_colored_by_seed(
        xs,
        per_seed_depth["composite"].values,
        per_seed_depth["seed"].values,
        x_labels=labels,
        outpath=OUTDIR / "mcts_composite_vs_depth_scatter_colored.png",
        ylabel="Composite Reward",
        show_baseline=True,
        marker_size=58,   # larger circles
    )

    # -------- Runtime plots (seconds -> minutes) --------
    if "seconds" in mcts.columns and not mcts["seconds"].dropna().empty:
        mins_mean = by_depth["mean_seconds"].values / 60.0
        mins_se   = by_depth["se_seconds"].values / 60.0
        per_seed_mins = per_seed_depth["seconds"].values / 60.0

        # Bar: mean minutes by depth (NO baseline line)
        bar_with_error(
            mins_mean,
            mins_se,
            labels,
            outpath=OUTDIR / "mcts_mean_runtime_by_depth.png",
            ylabel="Runtime (minutes)",
            show_baseline=False,
        )

        scatter_points_basic(
            xs,
            per_seed_mins,
            x_labels=labels,
            outpath=OUTDIR / "mcts_runtime_vs_depth_scatter.png",
            ylabel="Runtime (minutes)",
            show_baseline=False,
            marker_size=52,  # larger circles
        )

        scatter_points_colored_by_seed(
            xs,
            per_seed_mins,
            per_seed_depth["seed"].values,
            x_labels=labels,
            outpath=OUTDIR / "mcts_runtime_vs_depth_scatter_colored.png",
            ylabel="Runtime (minutes)",
            show_baseline=False,
            marker_size=58,  # larger circles
        )
    else:
        print("\nNote: 'seconds' column not found or empty; skipping runtime plots.")

    # Console quick look
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
