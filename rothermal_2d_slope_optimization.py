#!/usr/bin/env python3
import cupy as cp
import pandas as pd
import rasterio
import json
import numpy as np
from pathlib import Path
import math


def read_fuel_csv():
    """
    Reads a fuel model CSV file (named 'custom_fuel_model.csv') from the current directory.
    Expected CSV format (without header):
      ID, Label, 1hr_load, 10hr_load, 100hr_load, Live_herb_load, Live_woody_load,
      Static_Dynamic, 1hr_SAV, Live_Herb_SAV, Live_Woody_SAV, Fuel_Bed_Depth,
      Moisture_of_Extinction, Dead_Heat_Content, Live_Heat_Content
    Returns a dictionary mapping fuel IDs to a vector of fuel parameters.
    """
    fuels_path = Path("custom_fuel_model.csv").resolve()
    fuels = pd.read_csv(
        fuels_path,
        header=None,
        names=["ID", "Label", "1hr_load", "10hr_load", "100hr_load", "Live_herb_load", "Live_woody_load",
               "Static_Dynamic", "1hr_SAV", "Live_Herb_SAV", "Live_Woody_SAV", "Fuel_Bed_Depth",
               "Moisture_of_Extinction", "Dead_Heat_Content", "Live_Heat_Content"],
    )
    lookup_dict = fuels.set_index("ID").T.to_dict("list")

    def cast_to_float(values):
        float_values = []
        if len(values) > 6:
            values = values[:6] + values[7:]
        if len(values) > 0:
            values = values[1:]
        for value in values:
            try:
                float_values.append(float(value))
            except ValueError:
                float_values.append(cp.nan)
        return float_values

    filtered_lookup_dict = {k: cast_to_float(v) for k, v in lookup_dict.items()}
    filtered_lookup_dict = {k: v for k, v in filtered_lookup_dict.items() if v}
    return filtered_lookup_dict



def compute_landscape_ros(tiff_path, slope_adjustment=0.0):
    """
    Compute the rate of spread (ROS) for every cell in a TIFF landscape in
    all neighbour directions inside the chosen radius.
    The number of directions is (2*radius + 1)^2 – 1.
    """
    # ── read slope, aspect, fuel model ───────────────────────────────
    with rasterio.open(tiff_path) as src:
        slope_np       = src.read(2).astype(np.float64)
        aspect_np      = src.read(3).astype(np.float64)
        fuel_model_np  = src.read(4).astype(np.int32)

    slope        = cp.asarray(slope_np)
    aspect       = cp.asarray(aspect_np)
    rows, cols   = slope.shape

    # ── fuel-model → base-ROS lookup (CPU, then move to GPU) ─────────
    with open("experimental_spreadrates.json") as f:
        ros_lookup = json.load(f)

    base_ros_np  = np.vectorize(lambda fid: ros_lookup.get(str(fid), np.nan))(fuel_model_np)
    base_ros     = cp.asarray(base_ros_np)

    # ── build neighbourhood for any radius ──────────────────────────
    radius = 2                       # radius chosen at 2 for whole project
    directions_list = [
        (di, dj)
        for di in range(-radius, radius + 1)
        for dj in range(-radius, radius + 1)
        if not (di == 0 and dj == 0)
    ]
    directions = cp.array(directions_list, dtype=cp.float64)      # (N, 2)
    N          = directions.shape[0]                              # number of dirs
    phi            = cp.arctan2(-directions[:, 0], directions[:, 1])   # (N,)
    phi_expanded   = phi.reshape(1, 1, N)
    aspect_rad     = cp.deg2rad(aspect)
    effective_slope = slope[:, :, None] * cp.cos(aspect_rad[:, :, None] - phi_expanded)
    base_ros_expanded = cp.broadcast_to(base_ros[:, :, None], (rows, cols, N))
    final_ros = base_ros_expanded * (1 + slope_adjustment * (cp.tan(effective_slope * 0.01) ** 2))

    return final_ros


if __name__ == "__main__":
    # Specify the TIFF file name
    tiff_file = "landscape_fuel_104.tif"
    # Set an arbitrary slope adjustment factor (for example, 0.01) for testing
    slope_adjustment = 0.01
    # Compute the landscape ROS.
    ros_landscape = compute_landscape_ros(tiff_file, slope_adjustment=slope_adjustment)
    # Bring result back to CPU for display
    ros_landscape_cpu = cp.asnumpy(ros_landscape)
    print("Computed directional ROS for each cell (array shape: {}):".format(ros_landscape_cpu.shape))

    # For demonstration, print the 8 directional ROS for a sample cell.
    sample_i, sample_j = 30, 60
    if sample_i < ros_landscape_cpu.shape[0] and sample_j < ros_landscape_cpu.shape[1]:
        print("Cell ({}, {}) directional ROS:".format(sample_i, sample_j))
        print(ros_landscape_cpu[sample_i, sample_j, :])
    else:
        print("Sample cell ({}, {}) is out of bounds.".format(sample_i, sample_j))

    with rasterio.open(tiff_file) as src:
        fuel_model_np = src.read(4).astype(int)  # fuel model IDs
    fuel_model_id = fuel_model_np[sample_i, sample_j]
    print("Fuel model ID for cell ({}, {}): {}".format(sample_i, sample_j, fuel_model_id))

    fuel_lookup = read_fuel_csv()
    if fuel_model_id in fuel_lookup:
        print("Fuel model parameters for cell ({}, {}):".format(sample_i, sample_j))
        print(fuel_lookup[fuel_model_id])
    else:
        print("Fuel model ID {} not found in the lookup.".format(fuel_model_id))
