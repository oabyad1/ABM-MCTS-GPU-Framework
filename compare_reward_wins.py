#!/usr/bin/env python3
"""
compare_reward_wins.py

Compares how many times each strategy achieves the best (lowest composite score)
across two datasets (e.g., camp_med vs camp_high).

Outputs:
• plots_final_comparison/composite_win_counts.png
• plots_final_comparison/composite_win_counts.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Input CSVs


CSV_FILES = {
    "Esperanza (Low Uncertainty)": Path("summary_final_esper_med.csv"),
    "Esperanza (High Uncertainty)": Path("summary_final_esper_high.csv"),
}
#
# CSV_FILES = {
#     "Camp (Low Uncertainty)": Path("summary_final_camp_med.csv"),
#     "Camp (High Uncertainty)": Path("summary_final_camp_high.csv"),
# }

# CSV_FILES = {
#     "Marshall (Low Uncertainty)": Path("summary_final_marshall_med.csv"),
#     "Marshall (High Uncertainty)": Path("summary_final_marshall_high.csv"),
# }


# SCENARIO_COLORS = {
#     "Camp (Low Uncertainty)": "#1F4E79",
#     "Camp (High Uncertainty)": "#D95F02",  }


SCENARIO_COLORS = {
    "Esperanza (Low Uncertainty)": "#1F4E79",
    "Esperanza (High Uncertainty)": "#D95F02",  }


colors = ["#1F4E79", "#D95F02"]

# SCENARIO_COLORS = {
#     "Marshall (Low Uncertainty)": "#1F4E79",  # greenish
#     "Marshall (High Uncertainty)": "#D95F02",    # orange
# }
# ---------------------------------------------------------------------


colors = ["#1F4E79", "#D95F02"]

# Exclusions (exclude entire runs, regardless of seed)
# EXCLUDE_RUNS: set[int] = {9}   # esper high
# EXCLUDE_RUNS: set[int] = {43,49,91,     92,93,94,95,96,97}   # marshall med only up to 91 bd
# EXCLUDE_RUNS: set[int] = {0,105,110,12,36,53,94,9,46, 91,92,93,95}   # marshall high  only up to 46 bd
# Optional: exclude specific (run,seed) pairs if you ever need to

# Exclusions for specific datasets (optional)
# EXCLUDE_RUNS = {
#     "Camp (Low Uncertainty)": {},                       # marshall med
#     "Camp (High Uncertainty)": {},  # marshall high
# }
# EXCLUDE_RUNS = {
#     "Marshall (Low Uncertainty)": {43,49,91,     92,93,94,95,96,97},                       # marshall med
#     "Marshall (High Uncertainty)": {0,105,110,12,36,53,94,9,46, 91,92,93,95},  # marshall high
# }

EXCLUDE_RUNS = {
    "Esperanza (Low Uncertainty)": {},                       # marshall med
    "Esperanza (High Uncertainty)": {9},  # marshall high
}


OUTDIR = Path("plots_final_esper_wins")
OUTDIR.mkdir(parents=True, exist_ok=True)



# ---------------------------------------------------------------------
# Pretty names for strategies
STRATEGY_NAME_MAP = {
    # MCTS
    "mcts": "MCTS",

    # Baseline
    "random": "Random Allocation",

    # Rate-of-Spread–Driven
    "ics_ros_weighted": "ROS–Truth",
    "ics_ros_weighted_mean": "ROS–Mean",

    # Structure–Driven
    "ics_buildings": "Structures–Exposure",
    "ics_mean_burned_buildings": "Structures–Outcome (Mean)",
    "ics_truth_burned_buildings": "Structures–Outcome (Truth)",

    # Wind–Driven
    "ics_dynamic": "Wind–Truth",
    "ics_dynamic_mean": "Wind–Mean",
    "ics_dynamic_mean_lookahead": "Wind–Mean Lookahead",
    "ics_dynamic_truth_lookahead": "Wind–Truth Lookahead",
}
# Colors for scenario bars


# ---------------------------------------------------------------------
# def load_data(csv_path: Path, exclude_runs: set[int] | None = None) -> pd.DataFrame:
#     if not csv_path.exists():
#         raise FileNotFoundError(csv_path)
#     df = pd.read_csv(csv_path)
#
#     for c in ["area_score", "buildings_destroyed", "baseline_area", "baseline_buildings"]:
#         df[c] = pd.to_numeric(df[c], errors="coerce")
#
#     df["run_uid"] = df["run"].astype(str) + "|" + df["seed"].astype(str)
#     df["norm_area"] = df["area_score"] / df["baseline_area"]
#     df["norm_bldg"] = df["buildings_destroyed"] / df["baseline_buildings"]
#     df["composite"] = (1/3) * df["norm_area"] + (2/3) * df["norm_bldg"]
#
#     # --- Apply exclusions if provided ---
#     if exclude_runs:
#         before = len(df)
#         df = df[~df["run"].isin(exclude_runs)].copy()
#         after = len(df)
#         print(f"Excluded {before - after} rows for runs {exclude_runs} in {csv_path.name}")
#
#     return df
def load_data(csv_path: Path, exclude_runs: set[int] | None = None) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    for c in ["area_score", "buildings_destroyed", "baseline_area", "baseline_buildings"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["run_uid"] = df["run"].astype(str) + "|" + df["seed"].astype(str)
    df["norm_area"] = df["area_score"] / df["baseline_area"]
    df["norm_bldg"] = df["buildings_destroyed"] / df["baseline_buildings"]
    df["composite"] = (1/3) * df["norm_area"] + (2/3) * df["norm_bldg"]

    # --- Apply exclusions if provided ---
    if exclude_runs:
        before = len(df)
        df = df[~df["run"].isin(exclude_runs)].copy()
        after = len(df)
        print(f"Excluded {before - after} rows for runs {exclude_runs} in {csv_path.name}")

    # --- Keep only (run,seed) pairs where MCTS has a valid composite ---
    mcts_ok = df[
        (df["strategy"] == "mcts") &
        np.isfinite(df["composite"])
    ][["run_uid"]].drop_duplicates()

    before_runs = df["run_uid"].nunique()
    df = df[df["run_uid"].isin(set(mcts_ok["run_uid"]))].copy()
    after_runs = df["run_uid"].nunique()
    if before_runs != after_runs:
        print(f"[{csv_path.name}] Kept {after_runs}/{before_runs} run_uids with valid MCTS; "
              f"dropped {before_runs - after_runs} without MCTS.")

    return df


def compute_composite_wins(df: pd.DataFrame, label: str) -> pd.DataFrame:
    pivot = df.pivot_table(index="run_uid", columns="strategy", values="composite", aggfunc="mean")
    best_mask = pivot.eq(pivot.min(axis=1), axis=0)
    counts = best_mask.sum().rename("wins").reset_index()
    counts["dataset"] = label
    return counts


def plot_composite_wins(df_combined: pd.DataFrame, outpath: Path):
    agg = (
        df_combined.groupby(["strategy", "dataset"])["wins"]
        .sum()
        .unstack(fill_value=0)
    )
    agg["Total"] = agg.sum(axis=1)
    agg = agg.sort_values("Total", ascending=False)
    strategies = agg.index.tolist()
    datasets = list(agg.columns[:-1])  # exclude Total

    x = np.arange(len(strategies))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Plot bars for each scenario ---
    offsets = np.linspace(-width/2, width/2, len(datasets))
    for i, ds in enumerate(datasets):
        color = SCENARIO_COLORS.get(ds, "#888888")
        ax.bar(
            x + offsets[i],
            agg[ds].values,
            width / len(datasets) * 2,
            label=ds,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )

    # --- Label cleanup ---
    display_labels = [STRATEGY_NAME_MAP.get(s, s) for s in strategies]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=25, ha="right")

    # --- Formatting ---
    ax.set_ylabel("Number of Wins (Lowest Composite Score)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.margins(y=0.1)

    # --- Legend ABOVE chart ---
    ax.legend(
        title="Scenario",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(datasets),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    all_frames = []

    for label, path in CSV_FILES.items():
        exclude = EXCLUDE_RUNS.get(label, set())
        df = load_data(path, exclude_runs=exclude)
        wins = compute_composite_wins(df, label)
        all_frames.append(wins)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUTDIR / "composite_win_counts.csv", index=False)

    plot_composite_wins(combined, OUTDIR / "composite_win_counts.png")

    print("\n=== Composite Win Counts ===")
    pivot = (
        combined.groupby(["dataset", "strategy"])["wins"]
        .sum()
        .unstack(fill_value=0)
    )
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False)
    print(pivot)


if __name__ == "__main__":
    main()
