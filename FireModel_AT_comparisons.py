
"""

---------------------------------------------------------------------------
compare_arrival_times_dynamic.py   (FARSITE variable-wind runs, 2025-06-25)
---------------------------------------------------------------------------
• Reads each ComparisonRunsDynamic/FARSITE_RUN_* run
• Loads wind_schedule.json and stitches the GPU surrogate arrival map
• Generates:
    - side-by-side arrival maps
    - confusion map *and* 2×2 matrix
    - full time-series plots (Jaccard, Sørensen, Accuracy + area burned)
    - aggregated mean ± SD curves across all runs
    - CSV of snapshot metrics
---------------------------------------------------------------------------

"""
from pathlib import Path
import re, json, math
import numpy as np
import cupy as cp
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from collections import defaultdict
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x         # type: ignore

# from surrogate_fire_model_roth_w import SurrogateFireModelROS   # GPU model
# from surrogate_fire_model_CK2_multi_phase import SurrogateFireModelROS_CK2Multi
from SFM_CK2_adjusted import SurrogateFireModelROS_CK2Multi

from FIRE_MODEL_CUDA import SurrogateFireModelROS
#
# ---------------------------------------------------------------------------

# ----------------------------- USER SETTINGS -------------------------------
TIF_PATH       = Path("camp_fire_three_enhanced.tif").resolve()
# ROOT           = Path("ComparisonRunsDynamicShortened").resolve()
ROOT           = Path("ComparisonRunsDynamicCamp").resolve()

SIM_TIME_MIN   = 2000          # must match the batch script
INTERVAL_MIN   = 20           # step for time-series metrics
MAX_ITER       = 60
TOL            = 1e-4
CMAP           = "inferno"
CELL_ACRE      = (30 * 30) / 4046.8564224   # 30-m cell → acres
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------
def flammap_to_surrogate(wdir_flammap: int | float) -> int:
    """FlamMap: 0°=south (CW)  →  surrogate: 0°=west (CCW)."""
    return int((90 - wdir_flammap) % 360)


def binary_burn_mask(arrival: np.ndarray, threshold: float) -> np.ndarray:
    return arrival <= threshold


def compute_metrics(ref_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    tp = np.logical_and(ref_mask,  pred_mask).sum(dtype=np.int64)
    fp = np.logical_and(~ref_mask, pred_mask).sum(dtype=np.int64)
    fn = np.logical_and(ref_mask,  ~pred_mask).sum(dtype=np.int64)
    tn = np.logical_and(~ref_mask, ~pred_mask).sum(dtype=np.int64)
    total = tp + fp + fn + tn
    jaccard  = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    accuracy = (tp + tn) / total   if total           else np.nan
    dice     = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    ref_1d, pred_1d = ref_mask.ravel().astype(np.uint8), pred_mask.ravel().astype(np.uint8)
    kappa = cohen_kappa_score(ref_1d, pred_1d)
    mcc   = matthews_corrcoef(ref_1d, pred_1d)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "jaccard": jaccard, "accuracy": accuracy,
            "dice": dice, "kappa": kappa, "mcc": mcc}


def timeseries_metrics(arrival_ref: np.ndarray,
                       arrival_pred: np.ndarray,
                       sim_time: int,
                       step: int = 20) -> dict[str, np.ndarray]:

    thresholds = np.arange(step, sim_time + step, step, dtype=np.float32)
    jaccard, sorensen, accuracy = [], [], []
    area_ref, area_pred = [], []

    for t in thresholds:
        ref_mask  = arrival_ref  <= t
        pred_mask = arrival_pred <= t
        m = compute_metrics(ref_mask, pred_mask)
        jaccard.append(m["jaccard"])
        sorensen.append(m["dice"])
        accuracy.append(m["accuracy"])
        area_ref.append(ref_mask.sum(dtype=np.int64)  * CELL_ACRE)
        area_pred.append(pred_mask.sum(dtype=np.int64) * CELL_ACRE)

    return {"t": thresholds,
            "jaccard": np.asarray(jaccard, dtype=np.float32),
            "sorensen": np.asarray(sorensen, dtype=np.float32),
            "accuracy": np.asarray(accuracy, dtype=np.float32),
            "area_ref": np.asarray(area_ref, dtype=np.float32),
            "area_pred": np.asarray(area_pred, dtype=np.float32)}


_T_CACHE: dict[tuple[int, int], np.ndarray] = {}

# ---------------------------------------------------------------------------
# run the gpu firemodel with its built-in multi-phase solver
# ---------------------------------------------------------------------------
def surrogate_arrival_with_schedule(meta: dict, schedule_json: list[dict]) -> np.ndarray:
    """
    Run SurrogateFireModelROS once with the wind schedule supplied by FARSITE.

    Parameters
    ----------
    meta : dict
        Parsed contents of wind_schedule.json (we need meta["block_minutes"]).
    schedule_json : list[dict]
        Each item like {"start_min": 0, "wind_speed": 10, "wind_dir": 270}

    Returns
    -------
    np.ndarray
        Arrival-time raster (minutes) as a NumPy array.
    """
    # 1) Build the (t_start, t_end, wspd, wdir_surrogate) tuples -------------
    schedule_json = sorted(schedule_json, key=lambda x: x["start_min"])
    block_len     = meta["block_minutes"]
    wind_sched    = []

    for i, blk in enumerate(schedule_json):
        t_start = blk["start_min"]
        # End either at the next block’s start or at SIM_TIME_MIN
        if i + 1 < len(schedule_json):
            t_end = schedule_json[i + 1]["start_min"]
        else:
            t_end = SIM_TIME_MIN
        wspd      = blk["wind_speed"]
        wdir_surr = flammap_to_surrogate(blk["wind_dir"])
        wind_sched.append((t_start, t_end, wspd, wdir_surr))

    # 2) Run the GPU surrogate once, using the built-in multi-phase solver ---
    surrogate = SurrogateFireModelROS(
        tif_path=TIF_PATH,
        sim_time=SIM_TIME_MIN,
        wind_speed=wind_sched[0][2],            # ignored because wind_schedule provided
        wind_direction_deg=wind_sched[0][3],    # ignored because wind_schedule provided
        max_iter=MAX_ITER,
        tol=TOL,
        wind_schedule=wind_sched
    )

    # surrogate = SurrogateFireModelROS_CK2Multi(
    #     tif_path=TIF_PATH,
    #     sim_time=SIM_TIME_MIN,
    #     wind_speed=wind_sched[0][2],  # ignored because wind_schedule provided
    #     wind_direction_deg=wind_sched[0][3],  # ignored because wind_schedule provided
    #     max_iter=MAX_ITER,
    #     tol=TOL,
    #     wind_schedule=wind_sched
    # )
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(surrogate.T)



# ---------------------------------------------------------------------------
# ORIGINAL arrival-pair plot
# ---------------------------------------------------------------------------
def plot_arrival_pair(arr_ref: np.ndarray, arr_pred: np.ndarray,
                      mask: np.ndarray,                     # ← kept for call-site parity
                      wspd: int, wdir_f: int, wdir_gpu: int,
                      ref_model: str) -> None:
    """Side-by-side arrival-time maps with clear titles & annotation."""

    finite_vals = np.hstack([arr_ref[np.isfinite(arr_ref)],
                             arr_pred[np.isfinite(arr_pred)]])
    if finite_vals.size == 0:
        print("[skip plot] All cells NaN/Inf – no arrival map to show")
        return

    vmin, vmax = finite_vals.min(), finite_vals.max()
    if vmin == vmax:
        vmax = vmin + 1.0

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    im0 = axs[0].imshow(arr_ref,  cmap=CMAP, vmin=vmin, vmax=vmax, origin="upper")
    im1 = axs[1].imshow(arr_pred, cmap=CMAP, vmin=vmin, vmax=vmax, origin="upper")

    axs[0].set_title(f"Reference: {ref_model.upper()}")
    axs[1].set_title("GPU-enhanced Fire Model")
    for ax in axs:
        ax.axis("off")

    fig.subplots_adjust(right=0.88, bottom=0.25, top=0.92, wspace=0.02)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.67])
    fig.colorbar(im0, cax=cbar_ax, label="minutes")

    txt = (f"Wind speed: {wspd} mph   |   "
           f"{ref_model} dir: {wdir_f}° (0 = south, CW)   |   "
           f"GPU-model dir: {wdir_gpu}° (0 = west, CCW)")
    fig.text(0.5, 0.02, txt, ha="center", va="bottom", fontsize=9)
    plt.show()


def plot_confusion_map(ref_mask: np.ndarray, pred_mask: np.ndarray) -> None:
    cats = np.zeros_like(ref_mask, dtype=np.uint8)           # TN
    cats[np.logical_and(~ref_mask, pred_mask)] = 1           # FP
    cats[np.logical_and(ref_mask,  ~pred_mask)] = 2          # FN
    cats[np.logical_and(ref_mask,  pred_mask)]  = 3          # TP

    # map --------------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(cats, interpolation="none")
    plt.title("0=TN  1=FP  2=FN  3=TP")
    plt.axis("off");  plt.colorbar();  plt.tight_layout();  plt.show()

    # 2×2 matrix -------------------------------------------------------------
    tp, fp, fn, tn = (cats == 3).sum(), (cats == 1).sum(), (cats == 2).sum(), (cats == 0).sum()
    cm = np.array([[tp, fp],
                   [fn, tn]])
    plt.figure(figsize=(4, 4))
    vmax = cm.max() if cm.max() else 1
    im = plt.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    plt.title("Confusion matrix")
    plt.xticks([0, 1], ["Pred burn", "Pred unburn"])
    plt.yticks([0, 1], ["Actual burn", "Actual unburn"])
    thresh = vmax / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     fontsize=10, color=color)
    plt.tight_layout();  plt.show()


def plot_timeseries(ts: dict[str, np.ndarray], title_stub: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(ts["t"], ts["jaccard"],  marker="o", label="Jaccard")
    plt.plot(ts["t"], ts["sorensen"], marker="o", label="Sørensen–Dice")
    plt.plot(ts["t"], ts["accuracy"], marker="o", label="Accuracy")
    plt.xlabel("Elapsed time (min)");  plt.ylabel("Score");  plt.ylim(0, 1.1)
    plt.title(f"Spatial overlap over time – {title_stub}")
    plt.legend();  plt.grid(True);  plt.tight_layout();  plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(ts["t"], ts["area_ref"],  marker="o", label="Reference")
    plt.plot(ts["t"], ts["area_pred"], marker="o", label="Predicted")
    plt.xlabel("Elapsed time (min)");  plt.ylabel("Area burned (acres)")
    plt.title(f"Area burned over time – {title_stub}")
    plt.legend();  plt.grid(True);  plt.tight_layout();  plt.show()


def plot_aggregated_series(agg: defaultdict, thresholds: np.ndarray) -> None:
    for key, label in [("jaccard", "Jaccard"),
                       ("sorensen", "Sørensen–Dice"),
                       ("accuracy", "Accuracy")]:
        if not agg[key]: continue
        data = np.stack(agg[key])
        mean, std = data.mean(0), data.std(0, ddof=0)
        plt.figure(figsize=(6, 4))
        plt.errorbar(thresholds, mean, yerr=std, marker="o", capsize=3)
        plt.xlabel("Elapsed time (min)");  plt.ylabel(label);  plt.ylim(0, 1.1)
        plt.title(f"{label} over time – all runs (mean ± 1 SD)")
        plt.grid(True);  plt.tight_layout();  plt.show()


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
FOLDER_RE = re.compile(r"^FARSITE_RUN_(\d+)_([^/\\]+)$")
agg_ts = defaultdict(list)           # aggregated time-series curves
results = []
run_folders = sorted([p for p in ROOT.iterdir() if p.is_dir()])

for run_folder in tqdm(run_folders, desc="Comparing runs"):
    m = FOLDER_RE.match(run_folder.name)
    if not m:
        print(f"[skip] {run_folder.name}");  continue

    # ---- read wind schedule ------------------------------------------------
    sched_path = run_folder / "wind_schedule.json"
    if not sched_path.exists():
        print(f"[warn] {run_folder.name}: missing wind_schedule.json");  continue
    with sched_path.open() as f:
        meta = json.load(f)
    schedule = meta["wind_schedule"]
    first_blk = schedule[0]
    wspd0, wdir0_f = first_blk["wind_speed"], first_blk["wind_dir"]
    wdir0_s = flammap_to_surrogate(wdir0_f)

    # ---- load FARSITE raster ----------------------------------------------
    tif_candidates = list((run_folder / "Outputs").glob("*_Arrival*Time.tif"))
    if not tif_candidates:
        print(f"[warn] {run_folder.name}: no arrival raster");  continue
    arrival_path = tif_candidates[0]
    with rasterio.open(arrival_path) as src:
        arrival_ref_full = src.read(1).astype(np.float32)

    # clip negative & absurdly large values (same rule as before)
    arrival_ref_full = np.where(arrival_ref_full < 0,               np.inf, arrival_ref_full)
    arrival_ref_full = np.where(arrival_ref_full >= SIM_TIME_MIN * 20, np.inf, arrival_ref_full)

    # ---- surrogate variable-wind prediction -------------------------------
    arrival_pred_full = surrogate_arrival_with_schedule(meta, schedule)

    ##############################################################################################################
    ##############################################################################################################
    # ---------------------------------------------------------------------------
    # keep a copy and save it as a GeoTIFF in the same format as FARSITE’s
    # ---------------------------------------------------------------------------

    # gpu_arrival_full = arrival_pred_full.astype(np.float32)  # just a clearer name
    #
    # gpu_tif_path = (run_folder / "Outputs" / "GPU_Arrival_Time.tif")
    # with rasterio.open(arrival_path) as src_ref:
    #     profile = src_ref.profile.copy()  # crs, transform, width, height, etc.
    #     # Ensure the dtype matches our array; keep the same nodata as FARSITE
    #     profile.update(dtype=rasterio.float32)
    #
    #     # Replace NaN/+Inf with the reference raster’s nodata value, if needed
    #     nodata_val = profile.get("nodata", None)
    #     if nodata_val is not None:
    #         gpu_band = np.where(np.isfinite(gpu_arrival_full),
    #                             gpu_arrival_full,
    #                             nodata_val).astype(np.float32)
    #     else:
    #         gpu_band = gpu_arrival_full
    #
    #     with rasterio.open(gpu_tif_path, "w", **profile) as dst:
    #         dst.write(gpu_band, 1)
    #
    # print(f"✔  GPU arrival time written to {gpu_tif_path}")

    ##############################################################################################################
    ##############################################################################################################
    # ---- create burn masks -------------------------------------------------
    ref_mask  = binary_burn_mask(arrival_ref_full,  SIM_TIME_MIN)
    pred_mask = binary_burn_mask(arrival_pred_full, SIM_TIME_MIN)

    arrival_ref  = np.where(ref_mask,  arrival_ref_full,  np.nan)
    arrival_pred = np.where(pred_mask, arrival_pred_full, np.nan)
    mask_any     = ref_mask | pred_mask

    # ---- plots -------------------------------------------------------------
    plot_arrival_pair(arrival_ref, arrival_pred, mask_any,
                      wspd0, wdir0_f, wdir0_s, "FARSITE")
    plot_confusion_map(ref_mask, pred_mask)

    ts = timeseries_metrics(arrival_ref_full, arrival_pred_full,
                            SIM_TIME_MIN, INTERVAL_MIN)
    plot_timeseries(ts, run_folder.name)
    for key in ("jaccard", "sorensen", "accuracy"):
        agg_ts[key].append(ts[key])

    metrics = compute_metrics(ref_mask, pred_mask)
    results.append({"folder": run_folder.name,
                    "wind_speed_0": wspd0,
                    "wind_dir_0_f": wdir0_f,
                    **metrics})

# ---- aggregated curves ----------------------------------------------------
thresholds = np.arange(INTERVAL_MIN, SIM_TIME_MIN+INTERVAL_MIN, INTERVAL_MIN, dtype=np.float32)
plot_aggregated_series(agg_ts, thresholds)

# ---- CSV output -----------------------------------------------------------
df = pd.DataFrame(results).sort_values(["wind_speed_0", "wind_dir_0_f"])
csv_path = ROOT / "comparison_metrics.csv"
df.to_csv(csv_path, index=False)

print(f"\nPer-run snapshot metrics written to {csv_path.resolve()}")
for metric in ["accuracy", "jaccard", "dice"]:
    print(f"\n=== {metric.upper()} summary ===")
    print(df[metric].describe()[["mean", "min", "max", "std"]].round(4))
print("\nDone.")
