#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main purpose of this script is to visualize the uncertainty

Runs multiple schedule/TIF cases, simulating many "truth" fires per case
to quantify forecast-induced uncertainty.

For each case, the script:
  • samples N wind schedules (using per-segment Normal noise, clipped for speed)
  • runs SurrogateFireModelROS with the sampled schedule
  • records metrics per run: area_cells, buildings, perimeter_m, compactness
  • computes distributional summaries (mean/std/CV, quantiles) for area & buildings
  • computes pairwise Jaccard similarities between burned masks (shape variability)
  • saves: per-case metrics CSV, summary CSV, and plots:
        - violin+box for area & buildings
        - ECDFs for area & buildings
        - area vs buildings scatter (with Pearson r)
        - histogram of off-diagonal Jaccard values
        - optional gallery (set INCLUDE_GALLERY=True) of burned extents with wind roses
  • builds cross-case comparison plots (violin for area/buildings, and mean±95% CI)

"""

import math, pathlib, datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────
# CONFIG — edit these
# ───────────────────────────────────────────────

CASES = [
    dict(name="Esperanza Medium Uncertainty",
         SCHEDULE_CSV="wind_schedule_esper_med.csv",
         TIF_PATH="cali_test_big_enhanced.tif",
         SIM_MIN=3000),
    dict(name="Esperanza High Uncertainty",
         SCHEDULE_CSV="wind_schedule_esper_high.csv",
         TIF_PATH="cali_test_big_enhanced.tif",
         SIM_MIN=3000),

    # Examples you can enable:
    # dict(name="Camp Medium Uncertainty",
    #      SCHEDULE_CSV="wind_schedule_camp_med.csv",
    #      TIF_PATH="camp_fire_three_enhanced.tif",
    #      SIM_MIN=3000),
    #
    #
    # dict(name="Camp High Uncertainty",
    #      SCHEDULE_CSV="wind_schedule_camp_high.csv",
    #      TIF_PATH="camp_fire_three_enhanced.tif",
    #      SIM_MIN=3000),

    # dict(name="Marshall Medium Uncertainty",
    #      SCHEDULE_CSV="wind_schedule_marshall_med.csv",
    #      TIF_PATH="marshall_enhanced.tif",
    #      SIM_MIN=1200),
    #
    # dict(name="Marshall High Uncertainty",
    #      SCHEDULE_CSV="wind_schedule_marshall_high.csv",
    #      TIF_PATH="marshall_enhanced.tif",
    #      SIM_MIN=1200),
]

RUNS_PER_CASE = 1000     # number of stochastic truth fires per case
BASE_SEED     = None   # set an int for reproducibility, or None for time-based
MAX_ITER      = 250
TOL           = 1e-3
OUT_ROOT      = "final_out_uncertainty_esper"
INCLUDE_GALLERY = False   # galleries for 40 runs can be large; toggle as needed

# ───────────────────────────────────────────────
# Import fire model
# ───────────────────────────────────────────────
from FIRE_MODEL_CUDA import SurrogateFireModelROS  # , compute_ros_field

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────
def sample_truth_schedule(csv_path, rng):
    """Sample one truth schedule by perturbing each row’s mean with Normal noise."""
    df = pd.read_csv(csv_path)
    sched = []
    for r in df.itertuples():
        spd = float(max(rng.normal(r.speed_mean, r.speed_std), 0.0))
        deg = float(rng.normal(r.dir_mean, r.dir_std) % 360.0)
        sched.append((int(r.start_min), int(r.end_min), spd, deg))
    return sched

def raster_extent(tif_path: str):
    import rasterio
    with rasterio.open(tif_path) as src:
        return (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top), src.transform

def contour_perimeter_length(mask, transform):
    from skimage import measure
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return 0.0
    a = transform[0]; e = transform[4]
    dx, dy = abs(a), abs(e)
    length_m = 0.0
    for c in contours:
        dr = np.diff(c[:, 0]); dc = np.diff(c[:, 1])
        length_m += float(np.sum(np.sqrt((dr*dy)**2 + (dc*dx)**2)))
    return length_m

def compactness(area_cells, perim_m, transform):
    a, e = transform[0], transform[4]
    cell_area = abs(a*e)
    A = area_cells * cell_area
    if perim_m <= 0 or A <= 0:
        return float("nan")
    return float(4*math.pi*A/(perim_m**2))

def jaccard(maskA, maskB):
    inter = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()
    return float(inter/union) if union > 0 else float("nan")

def upper_triangle_values(M):
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    return M[iu]

def draw_wind_rose(ax, schedule):
    """
    Polar wind rose weighted by (duration * speed).
    Orientation: 0° = West, clockwise → 90° S, 180° E, 270° N.
    """
    dirs = np.array([seg[3] for seg in schedule], dtype=float) % 360.0
    dirs_rad = np.deg2rad(dirs)
    dur = np.array([seg[1]-seg[0] for seg in schedule], dtype=float)
    spd = np.array([seg[2] for seg in schedule], dtype=float)
    w   = np.clip(dur, 0, None) * np.clip(spd, 0, None)

    nbin = 16
    th_edges = np.linspace(0, 2*np.pi, nbin+1)
    hist, _ = np.histogram(dirs_rad, bins=th_edges, weights=w)
    th_centers = (th_edges[:-1]+th_edges[1:])/2

    ax.bar(th_centers, hist, width=(2*np.pi/nbin))
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(+1)
    ax.set_yticklabels([]); ax.set_xticklabels([])

def compute_summary_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, cv=np.nan, mn=np.nan,
                    p05=np.nan, p50=np.nan, p95=np.nan, mx=np.nan)
    n  = x.size
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    cv = float(sd/mu) if mu != 0 else np.nan
    return dict(
        n=n, mean=mu, std=sd, cv=cv,
        mn=float(np.min(x)),
        p05=float(np.percentile(x, 5)),
        p50=float(np.percentile(x, 50)),
        p95=float(np.percentile(x, 95)),
        mx=float(np.max(x))
    )

# ───────────────────────────────────────────────
# Plotting helpers
# ───────────────────────────────────────────────
def plot_case_distributions(df_case, case_name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Violin+box for area and buildings
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    vals = [df_case["area_cells"].values, df_case["buildings"].values]
    parts = ax.violinplot(vals, showmeans=True, showextrema=True, showmedians=True)
    ax.set_xticks([1,2], labels=["Area (cells)", "Buildings"])
    ax.set_title(f"{case_name}: Distribution of outcomes (n={len(df_case)})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{case_name}_violin_area_buildings.png", dpi=220)
    plt.close(fig)

    # ECDFs
    def _ecdf(x):
        x = np.sort(np.asarray(x, dtype=float))
        y = np.linspace(0, 1, x.size, endpoint=True)
        return x, y

    fig2, ax2 = plt.subplots(figsize=(6.6, 4.6))
    xA, yA = _ecdf(df_case["area_cells"].values)
    xB, yB = _ecdf(df_case["buildings"].values)
    ax2.plot(xA, yA, label="Area (cells)")
    ax2.plot(xB, yB, label="Buildings")
    ax2.set_xlabel("Value"); ax2.set_ylabel("ECDF")
    ax2.set_title(f"{case_name}: ECDFs")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{case_name}_ecdf_area_buildings.png", dpi=220)
    plt.close(fig2)

    # Scatter: area vs buildings
    fig3, ax3 = plt.subplots(figsize=(6.0, 5.2))
    ax3.scatter(df_case["area_cells"], df_case["buildings"], s=16, alpha=0.7)
    ax3.set_xlabel("Area (cells)")
    ax3.set_ylabel("Buildings")
    r = np.corrcoef(df_case["area_cells"], df_case["buildings"])[0,1] if len(df_case) > 1 else np.nan
    ax3.set_title(f"{case_name}: Area vs Buildings (r = {r:.2f})")
    fig3.tight_layout()
    fig3.savefig(out_dir / f"{case_name}_scatter_area_vs_buildings.png", dpi=220)
    plt.close(fig3)

def plot_case_jaccard_hist(J, case_name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    vals = upper_triangle_values(J)
    vals = vals[np.isfinite(vals)]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.hist(vals, bins=20, density=True)
    ax.set_xlabel("Jaccard index")
    ax.set_ylabel("Density")
    ax.set_title(f"{case_name}: Pairwise Jaccard (shape variability)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{case_name}_jaccard_hist.png", dpi=220)
    plt.close(fig)

def plot_case_gallery(burned_masks, schedules, extent, df_case, case_name, out_dir):
    n = len(burned_masks)
    ncols = 8; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 2.8*nrows))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols); ax = axes[r, c]
        if i >= n:
            ax.axis("off"); continue
        ax.imshow(burned_masks[i], cmap="hot", extent=extent, origin="upper", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Run {i:02d}  A={int(df_case.loc[i,'area_cells'])}, B={int(df_case.loc[i,'buildings'])}", fontsize=8)
        inset = ax.inset_axes([0.02, 0.02, 0.24, 0.24], projection='polar')
        draw_wind_rose(inset, schedules[i][1])
    fig.suptitle(f"{case_name}: burned extents gallery (n={n}) — 0°=W, 90°=S", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.2, w_pad=0.7)
    fig.savefig(out_dir / f"{case_name}_gallery.png", dpi=200)
    plt.close(fig)

def plot_compare_cases_violin(all_metrics, metric, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [g[metric].values for _, g in all_metrics.groupby("case")]
    labels = [k for k, _ in all_metrics.groupby("case")]
    fig, ax = plt.subplots(figsize=(max(6.6, 1.6*len(groups)), 4.6))
    ax.violinplot(groups, showmeans=True, showmedians=True, showextrema=True)
    ax.set_xticks(range(1, len(labels)+1), labels=labels, rotation=20)
    ax.set_title(f"Across cases: {metric}")
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_cases_violin_{metric}.png", dpi=220)
    plt.close(fig)

def plot_compare_cases_means(all_metrics, metric, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, g in all_metrics.groupby("case"):
        s = compute_summary_stats(g[metric].values)
        ci95 = 1.96 * (s["std"]/np.sqrt(max(1, s["n"])))
        rows.append((case, s["mean"], ci95))
    rows.sort(key=lambda x: x[0])
    cases = [r[0] for r in rows]
    means = [r[1] for r in rows]
    ci95s = [r[2] for r in rows]
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(max(6.6, 1.6*len(cases)), 4.4))
    ax.errorbar(x, means, yerr=ci95s, fmt='o', capsize=4)
    ax.set_xticks(x, cases, rotation=20)
    ax.set_title(f"Across cases: mean {metric} ± 95% CI")
    fig.tight_layout()
    fig.savefig(out_dir / f"compare_cases_means_{metric}.png", dpi=220)
    plt.close(fig)

# def _plot_hist_with_normal(ax, data, label, bins="auto"):
#     """Plot a density histogram and overlay a Normal(mu, sigma) PDF fit."""
#     x = np.asarray(data, dtype=float)
#     x = x[np.isfinite(x)]
#     if x.size == 0:
#         ax.text(0.5, 0.5, f"No data for {label}", ha="center", va="center", transform=ax.transAxes)
#         return
#     mu = float(np.mean(x))
#     sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
#
#     # Histogram as density
#     ax.hist(x, bins=bins, density=True, alpha=0.35, label=f"{label} Histogram")
#
#     # Overlay Normal if sd>0
#     if sd > 0:
#         # Generate support (truncate below zero)
#         xs = np.linspace(max(0, mu - 4 * sd), mu + 4 * sd, 400)
#
#         # Standard Normal PDF
#         pdf = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
#
#         # Explicitly zero out anything below 0 (safety in case of rounding)
#         pdf = np.where(xs < 0, 0.0, pdf)
#
#         ax.plot(xs, pdf, linewidth=2, label=f"{label} ~ N({mu:.1f}, {sd:.1f}²)")

def _plot_hist_with_normal(ax, data, label, bins="auto", color=None):
    """Plot a density histogram with matching Normal(mu, sigma) PDF overlay."""
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        ax.text(0.5, 0.5, f"No data for {label}", ha="center", va="center", transform=ax.transAxes)
        return
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0

    # Assign default color if not passed
    if color is None:
        color = "#4F81BD"  # default blue

    # Histogram
    ax.hist(x, bins=bins, density=True, alpha=0.4, color=color, label=f"{label} Histogram")

    # Overlay Normal PDF
    if sd > 0:
        xs = np.linspace(max(0, mu - 4 * sd), mu + 4 * sd, 400)
        pdf = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
        pdf = np.where(xs < 0, 0.0, pdf)
        ax.plot(xs, pdf, color=color, linewidth=2.0, label=f"{label} ~ N({mu:.1f}, {sd:.1f}²)")

# ───────────────────────────────────────────────
# Two-case comparison dashboard
# ───────────────────────────────────────────────
def _ecdf_vals(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x.sort()
    if x.size == 0:
        return np.array([]), np.array([])
    y = np.linspace(0, 1, x.size, endpoint=True)
    return x, y

def _exceedance(x, thresholds):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full(len(thresholds), np.nan)
    return np.array([(x >= t).mean() for t in thresholds], dtype=float)

def compare_two_cases_dashboard(caseA, caseB, df_all, out_dir, thresholds_area=None, thresholds_bldg=None):
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gA = df_all[df_all["case"] == caseA]
    gB = df_all[df_all["case"] == caseB]

    # 1) 2×2 grid: ECDF (area), violin (area), ECDF (buildings), violin (buildings)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # ECDF: Area
    xA, yA = _ecdf_vals(gA["area_cells"].values)
    xB, yB = _ecdf_vals(gB["area_cells"].values)
    axs[0,0].plot(xA, yA, label=caseA)
    axs[0,0].plot(xB, yB, label=caseB)
    axs[0,0].set_xlabel("Area (cells)"); axs[0,0].set_ylabel("ECDF")
    axs[0,0].set_title("Area: ECDF (lower is rarer, right is larger)")
    axs[0,0].legend()

    # Violin: Area
    axs[0,1].violinplot([gA["area_cells"].values, gB["area_cells"].values],
                        showmeans=True, showmedians=True, showextrema=True)
    axs[0,1].set_xticks([1,2], labels=[caseA, caseB])
    axs[0,1].set_title("Area: distribution (violin)")

    # ECDF: Buildings
    xA_b, yA_b = _ecdf_vals(gA["buildings"].values)
    xB_b, yB_b = _ecdf_vals(gB["buildings"].values)
    axs[1,0].plot(xA_b, yA_b, label=caseA)
    axs[1,0].plot(xB_b, yB_b, label=caseB)
    axs[1,0].set_xlabel("Buildings"); axs[1,0].set_ylabel("ECDF")
    axs[1,0].set_title("Buildings: ECDF")
    axs[1,0].legend()

    # Violin: Buildings
    axs[1,1].violinplot([gA["buildings"].values, gB["buildings"].values],
                        showmeans=True, showmedians=True, showextrema=True)
    axs[1,1].set_xticks([1,2], labels=[caseA, caseB])
    axs[1,1].set_title("Buildings: distribution (violin)")

    fig.suptitle(f"Variability comparison — {caseA} vs {caseB}", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out_dir / f"compare_{caseA}_vs_{caseB}_dashboard.png", dpi=220)
    plt.close(fig)

    # 2) Probability of exceedance curves (1−ECDF) at chosen thresholds
    if thresholds_area is None:
        pooled_area = np.concatenate([gA["area_cells"].values, gB["area_cells"].values])
        pooled_area = pooled_area[np.isfinite(pooled_area)]
        if pooled_area.size:
            thresholds_area = np.quantile(pooled_area, [0.5, 0.7, 0.8, 0.9, 0.95]).astype(float).tolist()
        else:
            thresholds_area = []
    if thresholds_bldg is None:
        pooled_b = np.concatenate([gA["buildings"].values, gB["buildings"].values])
        pooled_b = pooled_b[np.isfinite(pooled_b)]
        if pooled_b.size:

            qs = np.quantile(pooled_b, [0.5, 0.7, 0.8, 0.9, 0.95])
            thresholds_bldg = sorted({int(round(v)) for v in qs})
        else:
            thresholds_bldg = []

    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 4.2))

    # Area exceedance
    if thresholds_area:
        pA = _exceedance(gA["area_cells"].values, thresholds_area)
        pB = _exceedance(gB["area_cells"].values, thresholds_area)
        axs2[0].plot(thresholds_area, pA, marker='o', label=caseA)
        axs2[0].plot(thresholds_area, pB, marker='o', label=caseB)
    axs2[0].set_xlabel("Area threshold (cells)")
    axs2[0].set_ylabel("P(Area ≥ threshold)")
    axs2[0].set_title("Area: probability of exceedance")
    axs2[0].legend()

    # Buildings exceedance
    if thresholds_bldg:
        pA_b = _exceedance(gA["buildings"].values, thresholds_bldg)
        pB_b = _exceedance(gB["buildings"].values, thresholds_bldg)
        axs2[1].plot(thresholds_bldg, pA_b, marker='o', label=caseA)
        axs2[1].plot(thresholds_bldg, pB_b, marker='o', label=caseB)
    axs2[1].set_xlabel("Buildings threshold (count)")
    axs2[1].set_ylabel("P(Buildings ≥ threshold)")
    axs2[1].set_title("Buildings: probability of exceedance")
    axs2[1].legend()

    fig2.tight_layout()
    fig2.savefig(out_dir / f"compare_{caseA}_vs_{caseB}_exceedance.png", dpi=220)
    plt.close(fig2)

    colors = ["#1F4E79", "#D95F02"]


    fig_pdf_b, ax_pdf_b = plt.subplots(figsize=(10.0, 6.0))

    _plot_hist_with_normal(ax_pdf_b, gA["buildings"].values, caseA, bins="auto", color=colors[0])
    _plot_hist_with_normal(ax_pdf_b, gB["buildings"].values, caseB, bins="auto", color=colors[1])

    ax_pdf_b.set_xlabel("Structures Destroyed", fontsize=11)
    ax_pdf_b.set_ylabel("Probability Density", fontsize=11)
    # ax_pdf_b.set_title(f"Structures Distribution — {caseA} vs {caseB}", fontsize=12)
    ax_pdf_b.tick_params(axis="both", labelsize=10)

    # Add faint grid markers
    ax_pdf_b.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.4)
    ax_pdf_b.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)
    ax_pdf_b.minorticks_on()

    # Legend inside upper right corner, compact
    ax_pdf_b.legend(
        loc="upper right",
        fontsize=9.5,
        frameon=True,
        framealpha=0.8,
        borderpad=0.8
    )

    fig_pdf_b.tight_layout()
    fig_pdf_b.savefig(out_dir / f"compare_{caseA}_vs_{caseB}_pdf_buildings.png", dpi=220, bbox_inches="tight")
    plt.close(fig_pdf_b)

    # === NEW: Variance view — Area (cells) (PDF overlays) ===
    fig_pdf_a, ax_pdf_a = plt.subplots(figsize=(10.0, 6.0))
    # fig_pdf_a, ax_pdf_a = plt.subplots(figsize=(9.5, 6.0))
    _plot_hist_with_normal(ax_pdf_a, gA["area_cells"].values, caseA, bins="auto", color=colors[0])
    _plot_hist_with_normal(ax_pdf_a, gB["area_cells"].values, caseB, bins="auto", color=colors[1])

    ax_pdf_a.set_xlabel("Area Burned (# of cells)", fontsize=11)
    ax_pdf_a.set_ylabel("Probability Density", fontsize=11)
    # ax_pdf_a.set_title(f"Area Distribution — {caseA} vs {caseB}", fontsize=12)
    ax_pdf_a.tick_params(axis="both", labelsize=10)

    # Add faint grid markers
    ax_pdf_a.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.4)
    ax_pdf_a.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)
    ax_pdf_a.minorticks_on()

    ax_pdf_a.legend(
        loc="upper right",
        fontsize=9.5,
        frameon=True,
        framealpha=0.8,
        borderpad=0.8
    )

    fig_pdf_a.tight_layout()
    fig_pdf_a.savefig(out_dir / f"compare_{caseA}_vs_{caseB}_pdf_area.png", dpi=220, bbox_inches="tight")
    plt.close(fig_pdf_a)


def try_overlay_jaccard_hist(caseA, caseB, out_root):

    out_root = pathlib.Path(out_root)
    fA = out_root / caseA / "jaccard_values.csv"
    fB = out_root / caseB / "jaccard_values.csv"
    if not (fA.exists() and fB.exists()):
        return  # silently skip if not present
    jA = pd.read_csv(fA)["j"].values
    jB = pd.read_csv(fB)["j"].values
    jA = jA[np.isfinite(jA)]; jB = jB[np.isfinite(jB)]
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.hist(jA, bins=bins, alpha=0.6, density=True, label=caseA)
    ax.hist(jB, bins=bins, alpha=0.6, density=True, label=caseB)
    ax.set_xlabel("Jaccard index"); ax.set_ylabel("Density")
    ax.set_title(f"Shape variability: Jaccard overlay — {caseA} vs {caseB}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / f"compare_{caseA}_vs_{caseB}_jaccard_overlay.png", dpi=220)
    plt.close(fig)


# ───────────────────────────────────────────────
# Core per-case runner
# ───────────────────────────────────────────────
def run_case(case, rng_master):
    name = case["name"]
    schedule_csv = case["SCHEDULE_CSV"]
    tif_path     = case["TIF_PATH"]
    sim_min      = int(case["SIM_MIN"])

    case_out = pathlib.Path(OUT_ROOT) / name
    case_out.mkdir(parents=True, exist_ok=True)

    (xmin, xmax, ymin, ymax), transform = raster_extent(tif_path)
    extent = (xmin, xmax, ymin, ymax)

    # Sample schedules
    schedules = []
    for k in range(RUNS_PER_CASE):
        seed_k = int(rng_master.integers(0, 2**32-1, dtype=np.uint32))
        rng_k  = np.random.default_rng(seed_k)
        schedules.append((seed_k, sample_truth_schedule(schedule_csv, rng_k)))

    burned_masks, rows = [], []

    print(f"\n===== CASE: {name} — {RUNS_PER_CASE} runs =====")
    for idx, (seed, sched) in enumerate(schedules):
        print(f"\n── Run {idx:02d} (seed={seed})")
        for (t0, t1, spd, deg) in sched:
            print(f"  {t0:4d}–{t1:4d} min : {spd:5.2f} mph @ {deg:6.2f}°")

        mdl = SurrogateFireModelROS(
            tif_path=tif_path,
            sim_time=sim_min,
            wind_speed=sched[-1][2],
            wind_direction_deg=sched[-1][3],
            max_iter=MAX_ITER, tol=TOL,
            wind_schedule=sched
        )

        T_np   = mdl.arrival_time_grid
        burned = (T_np <= sim_min)
        burned_masks.append(burned)

        area_cells = int(burned.sum())
        try:
            buildings = int(mdl.calculate_building_score(sim_min))
        except Exception:
            buildings = 0
        perim = contour_perimeter_length(burned, transform)
        comp  = compactness(area_cells, perim, transform)

        rows.append(dict(case=name, run=idx, seed=seed, area_cells=area_cells,
                         buildings=buildings, perimeter_m=perim, compactness=comp))

    df_case = pd.DataFrame(rows)
    df_case.to_csv(case_out / f"{name}_metrics.csv", index=False)

    # Stats + Jaccard
    area_stats = compute_summary_stats(df_case["area_cells"].values)
    bldg_stats = compute_summary_stats(df_case["buildings"].values)

    # Jaccard matrix + stats
    n = len(burned_masks)
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = jaccard(burned_masks[i], burned_masks[j])
            J[i, j] = J[j, i] = val
    jvals = upper_triangle_values(J)
    jvals = jvals[np.isfinite(jvals)]
    j_stats = compute_summary_stats(jvals)

    # Save per-case summary
    summary = pd.DataFrame([
        dict(case=name, metric="area_cells", **area_stats),
        dict(case=name, metric="buildings", **bldg_stats),
        dict(case=name, metric="jaccard", **j_stats),
    ])
    summary.to_csv(case_out / f"{name}_summary.csv", index=False)

    # Plots
    plot_case_distributions(df_case, name, case_out)
    plot_case_jaccard_hist(J, name, case_out)
    if INCLUDE_GALLERY:
        plot_case_gallery(burned_masks, schedules, extent, df_case, name, case_out)

    return df_case, summary

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
def main():
    out_root = pathlib.Path(OUT_ROOT); out_root.mkdir(parents=True, exist_ok=True)
    base_seed = BASE_SEED or (int(dt.datetime.now().timestamp()*1e9) & 0xFFFFFFFF)
    rng_master = np.random.default_rng(base_seed)

    all_metrics = []
    all_summaries = []

    for case in CASES:
        df_case, s_case = run_case(case, rng_master)
        all_metrics.append(df_case)
        all_summaries.append(s_case)

    # Save aggregated results
    df_all = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    sm_all = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    df_all.to_csv(pathlib.Path(OUT_ROOT) / "all_cases_metrics.csv", index=False)
    sm_all.to_csv(pathlib.Path(OUT_ROOT) / "all_cases_summary.csv", index=False)

    # Cross-case comparison plots
    if not df_all.empty:
        plot_compare_cases_violin(df_all, "area_cells", pathlib.Path(OUT_ROOT))
        plot_compare_cases_violin(df_all, "buildings", pathlib.Path(OUT_ROOT))
        plot_compare_cases_means(df_all, "area_cells", pathlib.Path(OUT_ROOT))
        plot_compare_cases_means(df_all, "buildings", pathlib.Path(OUT_ROOT))

        # === Auto-discover active case names from CASES and compare all pairs ===
        active_cases = [c["name"] for c in CASES]
        if len(active_cases) >= 2:
            for i in range(len(active_cases)):
                for j in range(i + 1, len(active_cases)):
                    ca, cb = active_cases[i], active_cases[j]
                    compare_two_cases_dashboard(ca, cb, df_all, pathlib.Path(OUT_ROOT))
                    try_overlay_jaccard_hist(ca, cb, pathlib.Path(OUT_ROOT))
        else:
            print("Not enough active cases in CASES to build comparison dashboards.")

    print("\nDone. Results in:", OUT_ROOT)

if __name__ == "__main__":
    main()
