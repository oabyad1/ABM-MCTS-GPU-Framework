#!/usr/bin/env python3
"""
rothermal_2d_wind_CK.py

A "mega-kernel" version of the Rothermel calculation + multi-direction logic,
all in one CuPy RawKernel.
"""
import cupy as cp
import rasterio
import numpy as np
import json
import math
import pandas as pd
from pathlib import Path
import functools


from typing import NamedTuple          # ← NEW import (top of file)

_DIRECTION_LISTS: dict[int, list[tuple[int, int]]] = {}   # radius -> [(di,dj),..]

class _LandscapeCache(NamedTuple):
    slope_cp:  cp.ndarray;  aspect_cp:  cp.ndarray
    base_ros_cp: cp.ndarray;  slope_adj_cp: cp.ndarray
    l1_cp: cp.ndarray;   l10_cp: cp.ndarray;   l100_cp: cp.ndarray
    lh_cp: cp.ndarray;   lw_cp: cp.ndarray
    sav1_cp: cp.ndarray; savh_cp: cp.ndarray; savw_cp: cp.ndarray
    depth_cp: cp.ndarray; dmext_cp: cp.ndarray
    directions_phi_cp: cp.ndarray
    rows: int; cols: int; n_dirs: int; barrier_cp: cp.ndarray
###############################################################################
# 1) A raw CUDA kernel that computes final ROS(i,j,d) in a single pass
###############################################################################
# We'll define a single kernel "ros_kernel_3d".  For each thread index tid, we:
#   - convert tid -> (i, j, d) to locate which cell and which direction
#   - load data from the various input arrays (slope, aspect, base_ros, etc.)
#   - apply the Rothermel steps for that one pixel
#   - incorporate slope factor (based on aspect vs. direction)
#   - incorporate wind factor (wind direction vs. direction)
#   - store result in ros_out(i,j,d)
#
# The code below in C++ style is the direct translation of your previous
# "compute_landscape_ros" sub-steps, but compressed into one pass.  We also
# apply the "cured herb" logic for the loads, etc.
#
# NOTE: For brevity, "nansum" and partial sums are replaced by straightforward
# loops in the kernel code. That is typically how you'd handle sums over the 5
# fuel classes. This keeps everything purely on GPU in one pass.
#
# We'll supply the kernel with "launch_compute_ros_kernel_3d(...)" below.

ros_kernel_3d_code = r'''
extern "C" __global__
void ros_kernel_3d(
    // Dimensions
    const int rows,
    const int cols,
    const int n_dirs,

    // Fuel and terrain arrays (2D shape: rows*cols) flattened:
    const float* __restrict__ slope,         // slope[i,j]
    const float* __restrict__ aspect,        // aspect[i,j]
    const float* __restrict__ base_ros,      // base ROS from JSON
    const float* __restrict__ slope_adj,     // slope adjustment from JSON

    // Fuel param arrays (each is rows*cols):
    const float* __restrict__ load_1hr,
    const float* __restrict__ load_10hr,
    const float* __restrict__ load_100hr,
    const float* __restrict__ load_herb,
    const float* __restrict__ load_woody,
    const float* __restrict__ SAV_1hr,
    const float* __restrict__ SAV_herb,
    const float* __restrict__ SAV_woody,
    const float* __restrict__ fuel_bed_depth,
    const float* __restrict__ dead_moisture_of_ext, // M_x_dead from CSV

    // Overall wind inputs:
    const float wind_speed,       // e.g. 5 mph
    const float wind_dir_rad,     // e.g. deg2rad(wind_direction_deg)
    const float wind_adjustment,  // e.g. 1.05 or 1/15, etc.
    const float dir_exp,
    // Directions array (n_dirs), store the "phi" angle in radians for each direction
    //   phi[d] = angle of the outward vector from cell center
    //   We'll do: phi = arctan2(-di, dj). Then we adjust sign as needed
    const float* __restrict__ directions_phi,

    // Output: shape (rows*cols*n_dirs), flattened
    float* __restrict__ ros_out
)
{
    // Global thread index
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_elems = rows * cols * n_dirs;
    if (tid >= total_elems) return;

    // Convert tid -> (i, j, d)
    int d = tid % n_dirs;
    int rem = tid / n_dirs;
    int j = rem % cols;
    int i = rem / cols;

    // Flattened index for per-cell data
    int idx = i * cols + j;

    // Gather per-cell inputs
    float cell_slope    = slope[idx];
    float cell_aspect   = aspect[idx];
    float cell_base_ros = base_ros[idx];
    float cell_slopeadj = slope_adj[idx];
    float l_1hr   = load_1hr[idx];
    float l_10hr  = load_10hr[idx];
    float l_100hr = load_100hr[idx];
    float l_herb  = load_herb[idx];
    float l_woody = load_woody[idx];
    float sav_1hr  = SAV_1hr[idx];
    float sav_herb = SAV_herb[idx];
    float sav_woody= SAV_woody[idx];
    float depth    = fuel_bed_depth[idx];
    float Mx_dead  = dead_moisture_of_ext[idx];

    // Convert slope, aspect to radians for the direction offset
    float aspect_rad = 3.1415926535f * cell_aspect / 180.0f;

    // "phi" for this direction
    float phi_dir = directions_phi[d];

    ///////////////////////////////////////////////////////////////////////////
    // 1) Effective slope factor in direction d
    //    slope is e.g. slope[i,j] in degrees or percent?
    //    If slope array is in degrees, you might do "tan(slope_rad)". If it's
    //    in percent, you might do slope * 0.01.  We'll assume your slope is
    //    in percent and do "cos(aspect - phi) * slope(...)"
    ///////////////////////////////////////////////////////////////////////////

    // Convert slope% to a fraction
    // e.g. if slope=40 means 40% => 0.4
    float slope_frac = cell_slope * 0.01f;
    // Now "effective_slope" = slope_frac * cos(aspect - phi_dir)
    float angle_diff = aspect_rad - phi_dir;
    float slope_factor = cosf(angle_diff);
    // We clamp cos(...) to [-1..1], but it should be in that range anyway.
    float effective_slope = slope_frac * slope_factor;

    ///////////////////////////////////////////////////////////////////////////
    // 2) Wind factor in direction d
    //    We do a simple "cos(wind_dir - phi_dir)" => clamp to 0 => multiply by
    //    wind_speed => then pass that into your "wind_factor" formula.
    ///////////////////////////////////////////////////////////////////////////
    float wind_angle_diff = wind_dir_rad - phi_dir;
    float wind_component  = cosf(wind_angle_diff);
    if (wind_component < 0.f) wind_component = 0.f;
    wind_component = powf(wind_component, dir_exp);   // elliptical head–flank
    float midflame_ws = wind_speed * wind_component;

    ///////////////////////////////////////////////////////////////////////////
    // 3) Fuel load adjustments (apply_cured_herb)
    //    fraction_cured = 0.33, etc. We'll do that inline
    ///////////////////////////////////////////////////////////////////////////
    float fraction_cured = 0.33f;
    // We'll assume 1hr dead moisture = 0.03, herb=0.90, woody=1.20 from your example
    float moisture_1hr_dead = 0.03f;
    float moisture_live_herb = 0.90f;
    float moisture_live_woody= 1.20f;

    // Start with original loads
    float L_1hr = l_1hr;
    float L_herb = l_herb;
    // Move fraction_cured from live herb to dead 1hr
    float herb_cured = L_herb * fraction_cured;
    L_1hr  += herb_cured;
    L_herb -= herb_cured;

    // So final loads after "cured" step
    // (We'll rename them below for convenience)
    float final_1hr   = L_1hr;
    float final_10hr  = l_10hr;
    float final_100hr = l_100hr;
    float final_herb  = L_herb;
    float final_woody = l_woody;

    // final moisture array
    //   dead: 1hr=0.03, 10hr= ???, 100hr=???
    //   live: herb=0.90, woody=1.20
    // For simplicity, let's fix them here:
    float m1 = moisture_1hr_dead;
    float m2 = 0.04f;   // guess
    float m3 = 0.07f;   // guess
    float m4 = moisture_live_herb;
    float m5 = moisture_live_woody;

    ///////////////////////////////////////////////////////////////////////////
    // 4) Now replicate your Rothermel steps. We'll define 5 classes:
    //    Class0=1hr, Class1=10hr, Class2=100hr, Class3=herb, Class4=woody
    ///////////////////////////////////////////////////////////////////////////
    // A) compute surface area-to-volume (SAV) for each class
    float SAV_10hr = 109.0f;
    float SAV_100hr= 30.0f;
    float sav_vals[5];
    sav_vals[0] = sav_1hr;       // e.g. from CSV
    sav_vals[1] = SAV_10hr;
    sav_vals[2] = SAV_100hr;
    sav_vals[3] = sav_herb;
    sav_vals[4] = sav_woody;

    // B) loads:
    float load_vals[5];
    load_vals[0] = final_1hr;
    load_vals[1] = final_10hr;
    load_vals[2] = final_100hr;
    load_vals[3] = final_herb;
    load_vals[4] = final_woody;

    // C) moistures:
    float moist_vals[5];
    moist_vals[0] = m1;
    moist_vals[1] = m2;
    moist_vals[2] = m3;
    moist_vals[3] = m4;
    moist_vals[4] = m5;

    // Particle densities, heat content, etc. We'll assume constants:
    float density_vals[5];
    float heat_vals[5];
    float total_mineral_vals[5];
    float effective_mineral_vals[5];
    for (int k=0; k<5; k++){
        density_vals[k] = 32.0f;
        heat_vals[k]    = 8000.0f;
        total_mineral_vals[k]     = 0.0555f;
        effective_mineral_vals[k] = 0.010f;
    }

    // We'll replicate the main Rothermel formula pieces. Because it's 5 classes,
    // we can do partial sums in small loops.

    // Some inline lamdbas might help, but let's just do direct summations:

    // Step 1: Aij = (SAV * load_vals) / density
    float Aij[5];
    for(int p=0; p<5; p++){
        float SA = sav_vals[p];
        float LD = load_vals[p];
        float DN = density_vals[p];
        Aij[p] = (SA * LD) / (DN + 1e-9f);
    }

    // sum up "dead" = Aij[0..2], sum up "live"=Aij[3..4]
    float Aij_dead = 0.f;
    for(int p=0; p<3; p++){
        Aij_dead += Aij[p];
    }
    float Aij_live = 0.f;
    for(int p=3; p<5; p++){
        Aij_live += Aij[p];
    }
    float Ai[2];
    Ai[0] = Aij_dead;
    Ai[1] = Aij_live;
    // total
    float AT = Aij_dead + Aij_live;

    // weighting fij = Aij_dead / Ai_dead, etc. We'll just store them in place:
    float fij[5];
    for(int p=0; p<5; p++){
        float denominator = (p<3) ? (Ai[0]+1e-9f) : (Ai[1]+1e-9f);
        fij[p] = Aij[p] / denominator;
    }

    // fi = Ai / AT
    float fi_dead = (Ai[0]) / (AT + 1e-9f);
    float fi_live = (Ai[1]) / (AT + 1e-9f);

    // Next: Dead/Live partition for "W = dead_live_fuel_ratio"
    // Just replicate your formula: sum(dead_load * exp(-138/SAV_dead)) / sum(live_load*exp(-500/SAV_live))
    // We'll do it in two loops
    float numerator_d = 0.f;
    float denominator_d = 0.f;
    float numerator_l = 0.f;
    float denominator_l = 0.f;
    for (int p=0; p<3; p++){
        numerator_d += load_vals[p] * expf(-138.f / (sav_vals[p]+1e-9f));
        denominator_d += load_vals[p] * expf(-138.f / (sav_vals[p]+1e-9f)); // same
    }
    for (int p=3; p<5; p++){
        numerator_l += load_vals[p] * expf(-500.f / (sav_vals[p]+1e-9f));
        denominator_l += load_vals[p] * expf(-500.f / (sav_vals[p]+1e-9f));
    }
    float W=0.f;
    if (denominator_l > 0.001f){
        W = (numerator_d) / (denominator_l);
    }

    // "fine_dead_fuel_moisture" => weighted by exponent of -138 / SAV
    float num_mfdead=0.f, den_mfdead=0.f;
    for(int p=0; p<3; p++){
        float x = expf(-138.f / (sav_vals[p]+1e-9f));
        num_mfdead += moist_vals[p] * load_vals[p] * x;
        den_mfdead  += load_vals[p] * x;
    }
    float Mf_dead = 0.f;
    if (den_mfdead>1e-9f){
        Mf_dead = num_mfdead / den_mfdead;
    }

    // Mx_live from your formula: Mx2 = max(2.9*W*(1 - Mf_dead/Mx_dead)-0.226, Mx_dead)
    // But be sure to handle division if Mx_dead>0
    float Mx_live= Mx_dead; // default
    if (Mx_dead>1e-9f){
        float ratio_m = Mf_dead / Mx_dead;
        float candidate = 2.9f*W*(1.f - ratio_m)-0.226f;
        if (candidate < Mx_dead){
            Mx_live = Mx_dead; // per your ">= dead_fuel_moisture_of_ext"
        } else {
            Mx_live = candidate;
        }
    }

    // Next: combine "sigma" => characteristic SAV
    // sigma_dead = sum( f_i_dead * SAV_dead ), etc. We'll do simpler approach:
    float sigma_dead=0.f;
    float sigma_live=0.f;
    // for p=0..2 => dead
    for(int p=0; p<3; p++){
        sigma_dead += fij[p] * sav_vals[p];
    }
    // for p=3..4 => live
    for(int p=3; p<5; p++){
        sigma_live += fij[p] * sav_vals[p];
    }
    float sigma = sigma_dead * fi_dead + sigma_live * fi_live;

    // Mineral damping:
    // first we sum up e.g. f_ij * effective_mineral
    float S_dead=0.f; 
    float S_live=0.f;
    for(int p=0; p<3; p++){
        S_dead += fij[p] * effective_mineral_vals[p];
    }
    for(int p=3; p<5; p++){
        S_live += fij[p] * effective_mineral_vals[p];
    }
    float eta_s_dead = 0.174f*powf(S_dead, -0.19f);
    if (eta_s_dead>1.f) eta_s_dead=1.f;
    float eta_s_live = 0.174f*powf(S_live, -0.19f);
    if (eta_s_live>1.f) eta_s_live=1.f;

    // moisture_damping
    // We do Mx_dead for dead, Mx_live for live:
    float MfD=0.f, MfL=0.f;
    // sum dead moisture: sum( f_ij_dead * moist_vals[p] ), ...
    for(int p=0; p<3; p++){
        MfD += fij[p]*moist_vals[p];
    }
    float r_m_dead = MfD / (Mx_dead+1e-9f);
    if (r_m_dead>1.f) r_m_dead=1.f;
    float eta_m_dead = 1.f - 2.59f*r_m_dead + 5.11f*(r_m_dead*r_m_dead) - 3.52f*(r_m_dead*r_m_dead*r_m_dead);
    if (eta_m_dead<0.f) eta_m_dead=0.f;

    for(int p=3; p<5; p++){
        MfL += fij[p]*moist_vals[p];
    }
    float r_m_live = MfL / (Mx_live+1e-9f);
    if (r_m_live>1.f) r_m_live=1.f;
    float eta_m_live = 1.f - 2.59f*r_m_live + 5.11f*(r_m_live*r_m_live) - 3.52f*(r_m_live*r_m_live*r_m_live);
    if (eta_m_live<0.f) eta_m_live=0.f;

    // net fuel load w_n = load*(1-mineral)*g_ij. We'll do approximate approach:
    // For each of the 5 classes, w_n = load*(1 - total_mineral) * ??? We skip "g_fuel_ratio" detail
    // We'll just sum up dead, sum up live:
    float w_dead=0.f, w_live=0.f;
    for(int p=0; p<3; p++){
        float wn = load_vals[p]*(1.f - total_mineral_vals[p]); 
        w_dead += wn;
    }
    for(int p=3; p<5; p++){
        float wn = load_vals[p]*(1.f - total_mineral_vals[p]);
        w_live += wn;
    }

    // Heat content
    float h_dead=0.f, h_live=0.f;
    for(int p=0; p<3; p++){
        h_dead += fij[p]*heat_vals[p];
    }
    for(int p=3; p<5; p++){
        h_live += fij[p]*heat_vals[p];
    }

    // Bulk density
    // sum of loads / depth => approximate
    float total_load = final_1hr + final_10hr + final_100hr + final_herb + final_woody;
    float pb = (depth>1e-9f) ? (total_load/depth) : 0.f;

    // packing ratio
    // sum(load/density)/depth
    float sum_load_density=0.f;
    for(int p=0; p<5; p++){
        sum_load_density += (load_vals[p]/(density_vals[p]+1e-9f));
    }
    float beta = (depth>1e-9f) ? sum_load_density/depth : 0.f;

    // Beta optimum
    //    = 3.348 * sigma^(-0.8189)
    float beta_op = 3.348f * powf(sigma, -0.8189f);

    // relative packing ratio
    float rel_pack = (beta_op>1e-9f)? beta/beta_op : 0.f;

    // gamma_max
    //  = sigma^(1.5) / (495 + 0.0594*sigma^(1.5))
    float sigma_15 = powf(sigma, 1.5f);
    float gamma_max = sigma_15 / (495.f + 0.0594f*sigma_15 + 1e-9f);

    // optimum reaction velocity
    //   A = 133*sigma^(-0.7913)
    //   gamma = gamma_max*(rel_pack^A)*exp(A*(1-rel_pack))
    float A_val = 133.f*powf(sigma, -0.7913f);
    float relA = powf(rel_pack, A_val);
    float gamma = gamma_max * relA * expf(A_val*(1.f - rel_pack));

    // reaction intensity
    // Ir = gamma*( w_dead*h_dead*eta_m_dead*eta_s_dead + w_live*h_live*eta_m_live*eta_s_live )
    float Ir = gamma*(
        (w_dead*h_dead*eta_m_dead*eta_s_dead) +
        (w_live*h_live*eta_m_live*eta_s_live)
    );

    ///////////////////////////////////////////////////////////////////////////
    // 5) final ROS = base_ros(i,j) * [1 + slope_adj(i,j)* (tan(effective_slope))^2 + wind_factor(...) ]
    //    But let's define a "wind_factor" function inline: 
    //    from your code: factor = C*(mw^B)*(rel_pack^(-E)), etc.
    ///////////////////////////////////////////////////////////////////////////
    // We'll do a simplified approach or a direct approach from your example:

    // We'll replicate your "wind_factor" snippet:
    //   midflame_ws in ft/min => mph * 88 => we'll do that here:
    float mph_to_ft_min = 88.0f;
    float mw = midflame_ws * mph_to_ft_min;
    // from your code:
    //   C=7.47*exp(-0.133*sigma^(0.55*wind_adjustment))
    //   B=0.02526*sigma^0.54
    //   E=0.715*exp(-3.59e-4*sigma)
    float exponent_val = powf(sigma, (0.55f*wind_adjustment));
    float C = 7.47f*expf(-0.133f*exponent_val);
    float B = 0.02526f*powf(sigma, 0.54f);
    float E = 0.715f*expf(-3.59e-4f*sigma);
    float wind_fact = C* powf(mw, B)* powf(rel_pack, -E);

    // slope term => slope_adj*(tan(effective_slope))^2
    // if effective_slope is slope fraction, we do "tan(effective_slope??)???"
    // Possibly you want: slope_frac = slope[i,j]*0.01 => "tan(slope_rad)?"
    // We'll do a small approach:
    //   float slope_term = slope_adj*(tanf(effective_slope))^2, but be sure effective_slope is small
    float slope_term = cell_slopeadj * powf(tanf(effective_slope), 2.0f);

    float final_ros = cell_base_ros * (1.f + slope_term + wind_fact);

    // Done.  We store final into ros_out[tid]
    ros_out[tid] = final_ros;
}
'''


def launch_compute_ros_kernel_3d(
    slope_cp, aspect_cp, base_ros_cp, slope_adj_cp,
    load_1hr_cp, load_10hr_cp, load_100hr_cp, load_herb_cp, load_woody_cp,
    SAV_1hr_cp, SAV_herb_cp, SAV_woody_cp,
    bed_depth_cp, dead_mext_cp,
    wind_speed, wind_dir_rad, wind_adjustment,
    dir_exp,
    directions_phi_cp
):
    """
    Launch the raw kernel for shape (rows, cols, n_dirs).
    We'll build the 'ros_out' array of shape (rows, cols, n_dirs) and return it.
    """
    rows, cols = slope_cp.shape
    n_dirs = directions_phi_cp.shape[0]

    ros_out = cp.zeros((rows, cols, n_dirs), dtype=cp.float32)

    # Flatten arrays for kernel
    slope_flat     = slope_cp.ravel()
    aspect_flat    = aspect_cp.ravel()
    base_ros_flat  = base_ros_cp.ravel()
    slope_adj_flat = slope_adj_cp.ravel()
    l1_flat   = load_1hr_cp.ravel()
    l10_flat  = load_10hr_cp.ravel()
    l100_flat = load_100hr_cp.ravel()
    lh_flat   = load_herb_cp.ravel()
    lw_flat   = load_woody_cp.ravel()
    sav1_flat = SAV_1hr_cp.ravel()
    savh_flat = SAV_herb_cp.ravel()
    savw_flat = SAV_woody_cp.ravel()
    depth_flat= bed_depth_cp.ravel()
    dmext_flat= dead_mext_cp.ravel()
    directions_phi_flat = directions_phi_cp

    ros_out_flat = ros_out.ravel()

    # Compile the raw kernel if not already done
    # We do it once globally:
    raw_ker = cp.RawKernel(
        ros_kernel_3d_code,
        "ros_kernel_3d"
    )

    total_elems = rows * cols * n_dirs
    block_size = 256
    grid = ((total_elems + block_size - 1) // block_size,)  # tuple
    block = (block_size,)  # tuple

    # Launch
    raw_ker(
        grid,
        block,
        (
            rows,
            cols,
            n_dirs,
            slope_flat,
            aspect_flat,
            base_ros_flat,
            slope_adj_flat,
            l1_flat, l10_flat, l100_flat, lh_flat, lw_flat,
            sav1_flat, savh_flat, savw_flat,
            depth_flat,
            dmext_flat,
            np.float32(wind_speed),
            np.float32(wind_dir_rad),
            np.float32(wind_adjustment),
            # float(wind_speed),
            # float(wind_dir_rad),
            # float(wind_adjustment),
            np.float32(dir_exp),
            directions_phi_flat,
            ros_out_flat
        )
    )

    return ros_out


###############################################################################
# 2) Helper: build (base_ros, slope_adj) from JSON in one pass
###############################################################################
def build_combined_json_lookup(json_path):
    """
    Reads your 'combined_results.json' and builds a CPU NumPy array
    that maps 'fuel_id' -> (base_ros, slope_adj).
    We'll assume you have an upper bound on fuel_id (like 256 or 999).
    """
    with open(json_path, "r") as f:
        combined_lookup = json.load(f)

    max_fuel_id = 0
    for k_str in combined_lookup.keys():
        val = int(k_str)
        if val>max_fuel_id:
            max_fuel_id = val

    # We'll store results in shape (max_fuel_id+1, 2): col0=baseROS, col1=slopeAdj
    table = np.full((max_fuel_id+1, 2), np.nan, dtype=np.float32)

    for k_str, val_dict in combined_lookup.items():
        k_int = int(k_str)
        srate = float(val_dict.get("spread_rate", np.nan))
        sadd  = float(val_dict.get("slope_adjustment", np.nan))
        if 0<=k_int<=max_fuel_id:
            table[k_int, 0] = srate
            table[k_int, 1] = sadd

    return table


###############################################################################
# 3) Helper: read your "custom_fuel_model.csv" into a (maxID+1, paramCount) table
###############################################################################
def read_fuel_csv_and_build_table(csv_path):
    """
    We'll create a single NumPy table mapping 'fuel_id' -> columns:
      [1hr_load, 10hr_load, 100hr_load, live_herb_load, live_woody_load,
       1hr_SAV, live_herb_SAV, live_woody_SAV,
       bed_depth, dead_moisture_ext]
    """
    import pandas as pd
    import numpy as np

    # Read CSV to a DataFrame. Adjust 'header' and 'names' if your CSV has/has not a header row
    df = pd.read_csv(
        csv_path,
        header=None,
        names=[
            "ID", "Label", "1hr_load", "10hr_load", "100hr_load",
            "Live_herb_load", "Live_woody_load","Static_Dynamic",
            "1hr_SAV", "Live_Herb_SAV", "Live_Woody_SAV",
            "Fuel_Bed_Depth","Moisture_of_Extinction",
            "Dead_Heat_Content","Live_Heat_Content"
        ],
    )

    # Make sure numeric columns are floats
    numeric_cols = [
        "1hr_load", "10hr_load", "100hr_load",
        "Live_herb_load", "Live_woody_load",
        "1hr_SAV", "Live_Herb_SAV", "Live_Woody_SAV",
        "Fuel_Bed_Depth", "Moisture_of_Extinction",
        "Dead_Heat_Content", "Live_Heat_Content"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Maximum fuel-model ID
    max_id = int(df["ID"].max())

    # We store 10 columns in a table
    table = np.full((max_id+1, 10), np.nan, dtype=np.float32)

    tpa_to_lb_ft2 = 0.0459  # factor to convert from tons/acre to lb/ft^2

    # Fill table
    for _, row in df.iterrows():
        f_id = int(row["ID"])
        if f_id<0 or f_id>max_id:
            continue

        # Convert from "tons/acre" to "lb/ft^2"
        # Make sure row[...] is indeed a float now
        onehr   = float(row["1hr_load"])       * tpa_to_lb_ft2 if not np.isnan(row["1hr_load"])       else 0.0
        tenhr   = float(row["10hr_load"])      * tpa_to_lb_ft2 if not np.isnan(row["10hr_load"])      else 0.0
        hunhr   = float(row["100hr_load"])     * tpa_to_lb_ft2 if not np.isnan(row["100hr_load"])     else 0.0
        herb    = float(row["Live_herb_load"]) * tpa_to_lb_ft2 if not np.isnan(row["Live_herb_load"]) else 0.0
        woody   = float(row["Live_woody_load"])* tpa_to_lb_ft2 if not np.isnan(row["Live_woody_load"])else 0.0
        sav1    = float(row["1hr_SAV"])        if not np.isnan(row["1hr_SAV"])        else 0.0
        savh    = float(row["Live_Herb_SAV"])  if not np.isnan(row["Live_Herb_SAV"])  else 0.0
        savw    = float(row["Live_Woody_SAV"]) if not np.isnan(row["Live_Woody_SAV"]) else 0.0
        depth   = float(row["Fuel_Bed_Depth"]) if not np.isnan(row["Fuel_Bed_Depth"]) else 0.0
        mx_dead = float(row["Moisture_of_Extinction"])/100.0 if not np.isnan(row["Moisture_of_Extinction"]) else 0.0

        table[f_id, 0] = onehr
        table[f_id, 1] = tenhr
        table[f_id, 2] = hunhr
        table[f_id, 3] = herb
        table[f_id, 4] = woody
        table[f_id, 5] = sav1
        table[f_id, 6] = savh
        table[f_id, 7] = savw
        table[f_id, 8] = depth
        table[f_id, 9] = mx_dead

    return table


@functools.lru_cache(maxsize=None)            # ← works now that functools is imported
def _load_tiff_bands(tif_path: str):
    """
    Read the three bands we need from disk **once** and return NumPy arrays.
    The result is cached by path, so repeat calls with the same path are free.
    """
    with rasterio.open(tif_path) as src:
        return (
            src.read(2).astype(np.float32),   # slope
            src.read(3).astype(np.float32),   # aspect
            src.read(4).astype(np.int32),     # fuel-model
        )


@functools.lru_cache(maxsize=None)
def _combined_table(json_path: str) -> np.ndarray:
    """
    Return shape (max_id+1, 2) – base_ros and slope_adj per fuel-model ID.
    Cached after the first read.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    max_id = max(map(int, data.keys()))
    tbl = np.full((max_id + 1, 2), np.nan, dtype=np.float32)
    for k, v in data.items():
        k = int(k)
        tbl[k, 0] = float(v["spread_rate"])
        tbl[k, 1] = float(v["slope_adjustment"])
    return tbl

@functools.lru_cache(maxsize=None)
def _fuel_table(csv_path: str) -> np.ndarray:
    """
    Return a NumPy lookup table with shape (max_id+1, 10):

        [1-h, 10-h, 100-h, live_herb, live_woody,
         SAV_1h, SAV_herb, SAV_woody,
         bed_depth, dead_moist_ext]

    All loads are converted from tons / acre to lb / ft² exactly like before.
    The result is cached by *file path* – subsequent calls are free.
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        names=[
            "ID", "Label",
            "1hr_load", "10hr_load", "100hr_load",
            "Live_herb_load", "Live_woody_load", "Static_Dynamic",
            "1hr_SAV", "Live_Herb_SAV", "Live_Woody_SAV",
            "Fuel_Bed_Depth", "Moisture_of_Extinction",
            "Dead_Heat_Content", "Live_Heat_Content",
        ],
    )

    # numeric → float32
    num_cols = [
        "1hr_load", "10hr_load", "100hr_load",
        "Live_herb_load", "Live_woody_load",
        "1hr_SAV", "Live_Herb_SAV", "Live_Woody_SAV",
        "Fuel_Bed_Depth", "Moisture_of_Extinction",
    ]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").astype("float32")

    max_id = int(df["ID"].max())
    tbl = np.full((max_id + 1, 10), np.nan, dtype=np.float32)

    tpa_to_lb_ft2 = 0.0459  # tons/acre ➜ lb/ft²

    for _, row in df.iterrows():
        i = int(row["ID"])
        tbl[i] = np.array(
            [
                row["1hr_load"]       * tpa_to_lb_ft2,
                row["10hr_load"]      * tpa_to_lb_ft2,
                row["100hr_load"]     * tpa_to_lb_ft2,
                row["Live_herb_load"] * tpa_to_lb_ft2,
                row["Live_woody_load"]* tpa_to_lb_ft2,
                row["1hr_SAV"],
                row["Live_Herb_SAV"],
                row["Live_Woody_SAV"],
                row["Fuel_Bed_Depth"],
                row["Moisture_of_Extinction"] / 100.0,  # convert % ➜ fraction
            ],
            dtype=np.float32,
        )

    return tbl

@functools.lru_cache(maxsize=4)        # keep a few landscapes alive on GPU
def _get_landscape_cache(tif_path: str, radius: int) -> _LandscapeCache:
    """Read TIFF + look-ups, build CuPy tensors once, return a GPU cache."""
    # --- disk I/O (cached helpers) ----------------------------------------
    slope_np, aspect_np, fuel_np = _load_tiff_bands(str(tif_path))
    # combo_tbl = _combined_table("combined_results.json")
    # combo_tbl = _combined_table("combined_results_hc_radius2.json")
    # combo_tbl = _combined_table("combined_results_cedar.json")
    # combo_tbl = _combined_table("combined_results_hc_fixed_2.json")
    # combo_tbl = _combined_table("combined_results_hc3.json")

    # combo_tbl = _combined_table("combined_results_cali.json")
    # combo_tbl = _combined_table("combined_results_georgia.json")
    # combo_tbl = _combined_table("combined_results_test_experiment.json")


    combo_tbl = _combined_table("combined_results_esperenza.json")
    # combo_tbl = _combined_table("combined_results_marshall.json")

    # combo_tbl = _combined_table("combined_results_camp_fire.json")

    fuel_tbl  = _fuel_table("custom_fuel_model.csv")

    # --- per-pixel look-ups -----------------------------------------------
    max_id = combo_tbl.shape[0] - 1
    f_clamp = np.clip(fuel_np, 0, max_id)

    # ------------- NEW: boolean mask of permanent barriers ----------------
    UNBURNABLE = np.arange(90, 100, dtype=np.int32)  # 90‑99 stay forever
    barrier_np = np.isin(fuel_np, UNBURNABLE)  # True where rock / water
    barrier_cp = cp.asarray(barrier_np, dtype=cp.bool_)

    base_ros_np   = combo_tbl[f_clamp, 0]
    slope_adj_np  = combo_tbl[f_clamp, 1]
    fm_params     = fuel_tbl[f_clamp]          # shape (..., 10)

    # split params
    l1, l10, l100, lh, lw   = fm_params[..., :5].transpose(2,0,1)   # 5 tensors
    sav1, savh, savw        = fm_params[..., 5:8].transpose(2,0,1)
    depth_np, dmext_np      = fm_params[..., 8], fm_params[..., 9]

    # --- upload to GPU (one-time) -----------------------------------------
    slope_cp     = cp.asarray(slope_np,   dtype=cp.float32)
    aspect_cp    = cp.asarray(aspect_np,  dtype=cp.float32)
    base_ros_cp  = cp.asarray(base_ros_np,   dtype=cp.float32)
    slope_adj_cp = cp.asarray(slope_adj_np,  dtype=cp.float32)
    l1_cp, l10_cp, l100_cp = map(cp.asarray, (l1,  l10,  l100))
    lh_cp, lw_cp           = map(cp.asarray, (lh,  lw))
    sav1_cp, savh_cp, savw_cp = map(cp.asarray, (sav1, savh, savw))
    depth_cp   = cp.asarray(depth_np,  dtype=cp.float32)
    dmext_cp   = cp.asarray(dmext_np,  dtype=cp.float32)

    # --- neighbour directions (cached per radius) -------------------------
    if radius not in _DIRECTION_LISTS:
        _DIRECTION_LISTS[radius] = [
            (di, dj)
            for di in range(-radius, radius + 1)
            for dj in range(-radius, radius + 1)
            if (di, dj) != (0, 0)
        ]
    phis = np.array([math.atan2(-di, dj) for di, dj in _DIRECTION_LISTS[radius]],
                    dtype=np.float32)
    directions_phi_cp = cp.asarray(phis)
    n_dirs = directions_phi_cp.shape[0]

    rows, cols = slope_np.shape
    return _LandscapeCache(
        slope_cp, aspect_cp, base_ros_cp, slope_adj_cp,
        l1_cp, l10_cp, l100_cp, lh_cp, lw_cp,
        sav1_cp, savh_cp, savw_cp, depth_cp, dmext_cp,
        directions_phi_cp, rows, cols, n_dirs, barrier_cp
    )

###############################################################################
# 4) The main function: compute_landscape_ros
###############################################################################
# def compute_landscape_ros(tiff_path, wind_speed=5, wind_direction_deg=0,
#                           wind_adjustment=1.05, radius=2):
#     """
#     A single-pass GPU approach to compute rate of spread for each cell (i,j)
#     and each direction d in a radius. We return an array of shape (rows, cols, n_dirs)
#     plus the directions_list so you can interpret the output.
#
#     Steps:
#       1) Read from tiff: slope band=2, aspect band=3, fuel_model band=4
#       2) Build GPU arrays for slope, aspect
#       3) Use "combined_results.json" to get base_ros, slope_adj for each fuel model
#       4) Use "custom_fuel_model.csv" to get [1hr_load,10hr_load,etc.] for each fuel
#       5) Launch the rothermel mega-kernel
#       6) Return ros_out, directions_list
#     """
#     # # 1) read the relevant TIFF bands
#     # with rasterio.open(tiff_path) as src:
#     #     slope_np = src.read(2).astype(np.float32)   # slope
#     #     aspect_np= src.read(3).astype(np.float32)   # aspect
#     #     fuel_model_np = src.read(4).astype(np.int32)
#
#     slope_np, aspect_np, fuel_model_np = _load_tiff_bands(str(tiff_path))
#
#     rows, cols = slope_np.shape
#
#     # 2) Build the combined_results JSON lookup => map fuel_model -> (baseROS, slopeAdj)
#     #    We'll do it once and store in a CPU table. Then we index it with "fuel_model_np".
#     # combined_table = build_combined_json_lookup("combined_results.json")
#     combined_table = _combined_table("combined_results.json")  # ← cached
#     max_fuel_id = combined_table.shape[0]-1
#
#     # clamp fuel IDs
#     fuel_model_clamped = np.clip(fuel_model_np, 0, max_fuel_id)
#     # shape (rows, cols, 2)
#     base_and_slope = combined_table[fuel_model_clamped]
#     base_ros_np   = base_and_slope[..., 0]
#     slope_adj_np  = base_and_slope[..., 1]
#
#
#     # 3) Build the fuel CSV table => map fuel_model -> the 10 columns we stored
#     # fuel_csv_table = read_fuel_csv_and_build_table("custom_fuel_model.csv")
#     fuel_csv_table = _fuel_table("custom_fuel_model.csv")
#     # shape (rows, cols, 10)
#     fm_params = fuel_csv_table[fuel_model_clamped]
#
#     # Extract each param:
#     load_1hr_np   = fm_params[..., 0]
#     load_10hr_np  = fm_params[..., 1]
#     load_100hr_np = fm_params[..., 2]
#     load_herb_np  = fm_params[..., 3]
#     load_woody_np = fm_params[..., 4]
#     SAV_1hr_np    = fm_params[..., 5]
#     SAV_herb_np   = fm_params[..., 6]
#     SAV_woody_np  = fm_params[..., 7]
#     bed_depth_np  = fm_params[..., 8]
#     dead_mext_np  = fm_params[..., 9]
#
#     # 4) move everything to GPU
#     slope_cp     = cp.asarray(slope_np, dtype=cp.float32)
#     aspect_cp    = cp.asarray(aspect_np, dtype=cp.float32)
#     base_ros_cp  = cp.asarray(base_ros_np, dtype=cp.float32)
#     slope_adj_cp = cp.asarray(slope_adj_np, dtype=cp.float32)
#
#     l1_cp   = cp.asarray(load_1hr_np, dtype=cp.float32)
#     l10_cp  = cp.asarray(load_10hr_np, dtype=cp.float32)
#     l100_cp = cp.asarray(load_100hr_np, dtype=cp.float32)
#     lh_cp   = cp.asarray(load_herb_np, dtype=cp.float32)
#     lw_cp   = cp.asarray(load_woody_np, dtype=cp.float32)
#     sav1_cp = cp.asarray(SAV_1hr_np, dtype=cp.float32)
#     savh_cp = cp.asarray(SAV_herb_np, dtype=cp.float32)
#     savw_cp = cp.asarray(SAV_woody_np, dtype=cp.float32)
#     depth_cp= cp.asarray(bed_depth_np, dtype=cp.float32)
#     dmext_cp= cp.asarray(dead_mext_np, dtype=cp.float32)
#
#     # 5) Prepare neighbor directions
#     #    For radius=2, we get all (di,dj) with -2..2, skipping (0,0).
#     directions_list = [
#         (di, dj)
#         for di in range(-radius, radius + 1)
#         for dj in range(-radius, radius + 1)
#         if not (di == 0 and dj == 0)
#     ]
#     # We'll define phi[d] = arctan2(-di, dj) consistent with your earlier approach
#     phis = []
#     for (di, dj) in directions_list:
#         # your code: phi = arctan2(-di, dj)
#         phi_val = math.atan2(-di, dj)
#         phis.append(phi_val)
#     phis_np = np.array(phis, dtype=np.float32)
#
#     directions_phi_cp = cp.asarray(phis_np, dtype=cp.float32)
#     n_dirs = directions_phi_cp.shape[0]
#
#     # 6) Convert wind_direction_deg to radians
#     wind_dir_rad = math.radians(wind_direction_deg)
#
#     # 7) Launch the big raw kernel
#     ros_cp = launch_compute_ros_kernel_3d(
#         slope_cp, aspect_cp,
#         base_ros_cp, slope_adj_cp,
#         l1_cp, l10_cp, l100_cp, lh_cp, lw_cp,
#         sav1_cp, savh_cp, savw_cp,
#         depth_cp, dmext_cp,
#         wind_speed, wind_dir_rad, wind_adjustment,
#         directions_phi_cp
#     )
#
#     # 8) Return to user as a NumPy array if needed. Or keep as CuPy array.
#     # We'll match your original function signature => return (ros_array, directions_list).
#     # ros_array => shape (rows, cols, n_dirs)
#     return ros_cp, directions_list
# ---------------------------------------------------------------------------
#  NEW ultra-thin compute_landscape_ros
# ---------------------------------------------------------------------------
def compute_landscape_ros(tiff_path,
                          wind_speed: float = 5,
                          wind_direction_deg: float = 0,
                          wind_adjustment: float = 1.05,
                          radius: int = 2,
                          dir_exp: float = 2.0):
    """
    Launch the mega-kernel for a given landscape + wind scenario.

    All data that depend *only* on (tiff_path, radius) are pulled from the
    GPU-resident _LandscapeCache – so the hot loop is now just:
        * simple cache lookup
        * one kernel launch
    """
    # one cheap cache hit – everything we need is already on the GPU
    cache = _get_landscape_cache(str(tiff_path), radius)

    ros_cp = launch_compute_ros_kernel_3d(
        cache.slope_cp,  cache.aspect_cp,
        cache.base_ros_cp, cache.slope_adj_cp,
        cache.l1_cp,  cache.l10_cp,  cache.l100_cp,
        cache.lh_cp,  cache.lw_cp,
        cache.sav1_cp, cache.savh_cp, cache.savw_cp,
        cache.depth_cp, cache.dmext_cp,
        wind_speed,
        math.radians(wind_direction_deg),
        wind_adjustment,
        dir_exp,
        cache.directions_phi_cp
    )

    # return the CuPy array and the pre-built direction list
    return ros_cp, _DIRECTION_LISTS[radius]