"""
This script was used to vary the slope factor to minimize discrepancies between fires on landscapes with identical fuel models that are
run in FARSITE and the GPU model

USE slope_preprocessing.py to generate the landscapes with the constant fuel model across the landscape
These landscapes are necessary for this calibration

"""


import cupy as cp
import numpy as np
import rasterio
import json
import re
from pathlib import Path
import time
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from rothermal_2d_slope_optimization import compute_landscape_ros




def get_barrier_mask(tif_path, radius=2):
    """
    Returns a GPU boolean array (True where the mid‑cell can never burn).
    We mark fuel models 90‑99 as permanent barriers.
    """
    with rasterio.open(tif_path) as src:
        fuel_np = src.read(4).astype(np.int32)
    UNBURNABLE = np.arange(90, 100, dtype=np.int32)
    mask_np = np.isin(fuel_np, UNBURNABLE)
    return cp.asarray(mask_np, dtype=cp.bool_)


def shift_no_wrap(array, di, dj, fill_value=cp.nan):
    """
    Shift a CuPy array without wrapping.
    Out-of-bound positions are filled with fill_value.
    Returns the shifted array and a boolean mask of valid (in-bound) positions.
    """
    rows, cols = array.shape
    shifted = cp.full((rows, cols), fill_value, dtype=array.dtype)
    if di < 0:
        src_row = slice(0, di)
        dest_row = slice(-di, rows)
    elif di > 0:
        src_row = slice(di, rows)
        dest_row = slice(0, rows - di)
    else:
        src_row = slice(0, rows)
        dest_row = slice(0, rows)
    if dj < 0:
        src_col = slice(0, dj)
        dest_col = slice(-dj, cols)
    elif dj > 0:
        src_col = slice(dj, cols)
        dest_col = slice(0, cols - dj)
    else:
        src_col = slice(0, cols)
        dest_col = slice(0, cols)
    shifted[dest_row, dest_col] = array[src_row, src_col]
    valid = ~cp.isnan(shifted)
    return shifted, valid


def compute_delay_maps_from_ros_opt(tiff_path, rows, cols, slope_adjustment, eps=1e-6):
    """
    Compute delay maps using directional ROS computed with the given slope_adjustment.
    Each cell is assumed to be 30 m x 30 m (converted to feet).
    """
    ros_array = compute_landscape_ros(tiff_path, slope_adjustment=slope_adjustment)
    cell_size_ft = 30 * 3.28084  # 30 m in feet
    radius = 2
    neighbors = [(di, dj) for di in range(-radius, radius + 1)
                 for dj in range(-radius, radius + 1) if not (di == 0 and dj == 0)]
    multipliers = [cp.sqrt(di ** 2 + dj ** 2) for di, dj in neighbors]
    delay_maps = []
    for k, (di, dj) in enumerate(neighbors):
        ros = ros_array[..., k]
        distance = cell_size_ft * multipliers[k]
        delay = distance / (ros + eps)
        delay_maps.append((di, dj, delay))
    return delay_maps


def run_deterministic_arrival_with_ros_opt(tiff_path, grid_size, slope_adjustment, barrier_cp, max_iter=500, tol=1e-4):
    """
    Compute the fire arrival time map T (as a CuPy array) using an iterative update
    based on delay maps computed from the ROS (which in turn depends on slope_adjustment).
    """
    rows, cols = grid_size
    T = cp.full(grid_size, cp.inf, dtype=cp.float32)
    center = (rows // 2, cols // 2)
    T[center] = 0.0

    delay_maps = compute_delay_maps_from_ros_opt(tiff_path, rows, cols, slope_adjustment)
    for it in range(max_iter):
        T_old = T.copy()
        candidates = []

        #old
        # for di, dj, delay in delay_maps:
        #     T_neighbor, valid_mask = shift_no_wrap(T, di, dj)
        #     T_neighbor = cp.where(valid_mask, T_neighbor, cp.inf)
        #     candidate = T_neighbor + delay
        #     candidates.append(candidate)

        # new
        for di, dj, delay in delay_maps:
            T_neighbor, valid_mask = shift_no_wrap(T, di, dj)
            T_neighbor = cp.where(valid_mask, T_neighbor, cp.inf)
            candidate = T_neighbor + delay

            # # veto radius‑2 hops over a barrier cell
            # if abs(di) > 1 or abs(dj) > 1:
            #     mid_i, mid_j = di // 2, dj // 2
            #     mid_block, _ = shift_no_wrap(barrier_cp, mid_i, mid_j)
            #     candidate = cp.where(mid_block, cp.inf, candidate)

            # 3️⃣ veto any diagonal that “cuts the corner”
            if di and dj:  # diagonal step
                side1, _ = shift_no_wrap(barrier_cp, 0, dj, fill_value=True)
                side2, _ = shift_no_wrap(barrier_cp, di, 0, fill_value=True)
                corner_block = side1 | side2
                candidate = cp.where(corner_block, cp.inf, candidate)

            # 4️⃣ veto hops ≥2 whose mid‑cell or flanks are blocked
            if abs(di) > 1 or abs(dj) > 1:
                mid_i, mid_j = di // 2, dj // 2
                mid_block, _ = shift_no_wrap(barrier_cp, mid_i, mid_j, fill_value=True)
                flank1, _ = shift_no_wrap(barrier_cp, mid_i, 0, fill_value=True)
                flank2, _ = shift_no_wrap(barrier_cp, 0, mid_j, fill_value=True)
                block_long = mid_block | flank1 | flank2
                candidate = cp.where(block_long, cp.inf, candidate)
            candidates.append(candidate)

        candidates_stack = cp.stack(candidates, axis=0)
        T_new = cp.minimum(T, cp.min(candidates_stack, axis=0))
        T = T_new
        if cp.max(cp.abs(T - T_old)) < tol:
            print(f"Converged after {it + 1} iterations.")
            break
    else:
        print("Reached maximum iterations without full convergence.")
    return T


def load_truth_arrival(tiff_truth_path):
    """
    Load the truth arrival time TIFF (assumed to be in band 1) and return as a NumPy array.
    Any cell with a value of -9999 is set to NaN.
    """
    with rasterio.open(tiff_truth_path) as src:
        truth = src.read(1).astype(np.float32)
    truth[truth == -9999] = np.nan
    return truth


# -------------------------------
# New functions for optimization loop and plotting
# -------------------------------

def objective_no_plot(slope_adjustment, tiff_test, tiff_truth, time_threshold=1000):
    """
    Objective function to be minimized (without plotting on each iteration).
    Computes the raw mismatch error between the predicted and truth arrival time maps.
    """
    with rasterio.open(tiff_test) as src:
        rows, cols = src.height, src.width
    grid_size = (rows, cols)
    barrier_cp = get_barrier_mask(tiff_test)  # ← add

    T_pred = run_deterministic_arrival_with_ros_opt(tiff_test, grid_size, slope_adjustment, barrier_cp)
    T_pred_np = cp.asnumpy(T_pred)
    pred_mask = (T_pred_np <= time_threshold).astype(np.float32)
    truth = load_truth_arrival(tiff_truth)
    truth_mask = (truth <= time_threshold).astype(np.float32)
    error = np.nansum(np.abs(pred_mask - truth_mask))
    print(f"Slope adjustment: {slope_adjustment:.4f}, Raw mismatch error: {error}")
    return error


def plot_final_arrival_comparison(tiff_test, tiff_truth, slope_adjustment, time_threshold=1000):
    """
    Plot the predicted and truth arrival time maps side-by-side for the final (optimal)
    slope_adjustment.
    """
    with rasterio.open(tiff_test) as src:
        rows, cols = src.height, src.width
    grid_size = (rows, cols)
    barrier_cp = get_barrier_mask(tiff_test)

    T_pred = run_deterministic_arrival_with_ros_opt(tiff_test, grid_size, slope_adjustment, barrier_cp)
    T_pred_np = cp.asnumpy(T_pred)
    pred_mask = (T_pred_np <= time_threshold).astype(np.float32)
    truth = load_truth_arrival(tiff_truth)
    truth_mask = (truth <= time_threshold).astype(np.float32)

    plt.figure("Final Arrival Times", figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(pred_mask, cmap='viridis')
    plt.title("Predicted Arrival Times")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(truth_mask, cmap='viridis')
    plt.title("Truth Arrival Times")
    plt.colorbar()
    plt.suptitle(f"Final Slope Adjustment = {slope_adjustment:.4f}")
    plt.show()


# -------------------------------
# Main optimization loop
# -------------------------------

def main():
    time_threshold = 1000
    bounds = (0, 100)

    # Define directories
    landscape_dir = Path("Optimization Landscapes Camp Fire")
    truth_root = Path("OptimizationSlopeRunsCampFire")
    # landscape_dir = Path("Optimization Landscapes Cedar")
    # truth_root = Path("OptimizationSlopeRunsCedar")

    # Dictionary to store the optimal slope_adjustment keyed by fuel model number.
    optimal_slopes = {}

    # Get a sorted list of all landscape TIFF files that match the pattern.
    landscape_files = sorted(landscape_dir.glob("testslope_modified*.tif"))
    if not landscape_files:
        print("No landscape files found in 'Optimization Landscapes'.")
        return

    for landscape_file in landscape_files:
        # Extract the fuel model number (e.g., "101") from the filename.
        match = re.search(r"testslope_modified(\d+)\.tif", landscape_file.name)
        if not match:
            print(f"Filename {landscape_file.name} does not match expected pattern. Skipping.")
            continue
        fuel_model = match.group(1)
        # Construct the corresponding truth file path.
        truth_file = truth_root / fuel_model / "Outputs" / "_Arrival Time.tif"
        # truth_file = truth_root / fuel_model / "Outputs" / "_MTT_ArrivalTime.tif"
        if not truth_file.exists():
            print(f"Truth file {truth_file} does not exist for {landscape_file.name}. Skipping.")
            continue

        print(f"\nOptimizing fuel model: {fuel_model} ({landscape_file.name})")
        print(f"Using truth file: {truth_file}")
        start_time = time.time()

        result = minimize_scalar(
            objective_no_plot,
            args=(str(landscape_file), str(truth_file), time_threshold),
            bounds=bounds,
            method='bounded',
            options={'xatol': 1e-4}
        )
        elapsed = time.time() - start_time

        optimal_slope = result.x
        print(f"Optimal slope adjustment for fuel model {fuel_model}: {optimal_slope:.4f}")
        print(f"Optimization time: {elapsed:.2f} seconds")

        plot_final_arrival_comparison(str(landscape_file), str(truth_file), optimal_slope, time_threshold)

        optimal_slopes[fuel_model] = optimal_slope

    # Write the slope adjustments to a JSON file.
    slopes_file = "optimal_slope_adjustments_barrier_cali_scenario.json"
    with open(slopes_file, "w") as f:
        json.dump(optimal_slopes, f, indent=4)
    print(f"\nSlope adjustment results saved to {slopes_file}")

    #load the experimental spread rates and merge with the slope adjustments.
    try:
        with open("experimental_spreadrates.json", "r") as f:
            spreadrates = json.load(f)
    except FileNotFoundError:
        print("experimental_spreadrates.json not found.")
        return

    combined_results = {}
    for fuel_model, slope in optimal_slope.items() if False else optimal_slopes.items():
        # Look up the spread rate; it may be None if not found
        spread_rate = spreadrates.get(fuel_model)
        combined_results[fuel_model] = {
            "slope_adjustment": slope,
            "spread_rate": spread_rate
        }

    # Write the combined results to a new JSON file.
    combined_file = "combined_results_camp_fire.json"
    with open(combined_file, "w") as f:
        json.dump(combined_results, f, indent=4)
    print(f"Combined results saved to {combined_file}")


if __name__ == "__main__":
    main()
