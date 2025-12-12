#!/usr/bin/env python3
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import json
import time
from pathlib import Path
from skimage.draw import polygon
import heapq
from skimage import measure                     # perimeter extraction
import matplotlib.cm as cm                      # colormap for ROS plots
from matplotlib.lines import Line2D             # custom legend handles

# Import your new ROS function that returns (ros_array, directions_list)
# from rothermal_2d_wind_CK import compute_landscape_ros
from rothermal_2d_wind_CK2_p import compute_landscape_ros
# from rothermal_2d_wind_CK2_memory import compute_landscape_ros
# from rothermal_2d_wind_CK2_DEBUG import compute_landscape_ros
from rothermal_2d_wind_CK2_p import _get_landscape_cache      # already in env


def get_barrier_mask(tif_path: str, radius: int = 2) -> cp.ndarray:
    """Boolean CuPy array (True where fuel 90‑99).  Cached on first call."""
    return _get_landscape_cache(str(tif_path), radius).barrier_cp

def _shift(arr: cp.ndarray, di: int, dj: int, *, fill=True) -> cp.ndarray:
    """
    Like numpy.roll but fills the out‑of‑bounds region with *fill*
    instead of wrapping.  Works on GPU CuPy arrays.
    """
    if di == 0 and dj == 0:
        return arr
    rows, cols = arr.shape
    out = cp.empty_like(arr)
    # roll then overwrite the “wrapped” strip
    out[:] = cp.roll(arr, (di, dj), axis=(0, 1))
    if di > 0:   out[:di, :]  = fill
    if di < 0:   out[di:, :]  = fill
    if dj > 0:   out[:, :dj]  = fill
    if dj < 0:   out[:, dj:]  = fill
    return out
def apply_barrier_rules(
        delays_cp: cp.ndarray,
        barrier_cp: cp.ndarray,
        directions: list[tuple[int, int]],
        *,
        inf: float = 1e30
) -> cp.ndarray:
    """
    In‑place: for every direction k, set delays_cp[k, …] = ∞ wherever the
    path would violate the corner‑cut / long‑hop rules.
    """
    INF = cp.float32(inf)
    for k, (di, dj) in enumerate(directions):
        # start with “no extra block”
        blocked = cp.zeros_like(barrier_cp, dtype=cp.bool_)

        # ① diagonal corner‑cut
        if di and dj:
            side1 = _shift(barrier_cp, 0,  dj,  fill=True)
            side2 = _shift(barrier_cp, di, 0,  fill=True)
            blocked |= side1 | side2

        # ② long hops ≥ 2 cells
        if abs(di) > 1 or abs(dj) > 1:
            mid_i, mid_j = di // 2, dj // 2
            mid   = _shift(barrier_cp, mid_i, mid_j, fill=True)
            flank1= _shift(barrier_cp, mid_i, 0,      fill=True)
            flank2= _shift(barrier_cp, 0,      mid_j, fill=True)
            blocked |= mid | flank1 | flank2

        # write ∞ where blocked
        delays_cp[k] = cp.where(blocked, INF, delays_cp[k])

    return delays_cp

#this blocked mask is beter for the marshal fire and the other one is better for camp (esperenza unknown yet)
# --------------------------------------------- helpers
def _blocked_mask(barrier_cp: cp.ndarray,
                  dirs: list[tuple[int,int]]) -> cp.ndarray:
    """
    Return a Boolean tensor M  (n_dirs, H, W)  where
      M[k,i,j] == True   → the step (i,j) → (i+di,j+dj) is disallowed.
    """
    n_dirs, H, W = len(dirs), *barrier_cp.shape
    M = cp.zeros((n_dirs, H, W), dtype=cp.bool_)

    for k, (di, dj) in enumerate(dirs):
        # corner‑cut
        if di and dj:
            M[k] |= _shift(barrier_cp, 0,  dj) | _shift(barrier_cp, di, 0)
        # ≥ 2‑cell hops
        if abs(di) > 1 or abs(dj) > 1:
            mid_i, mid_j = di // 2, dj // 2
            M[k] |= (_shift(barrier_cp, mid_i, mid_j) |
                     _shift(barrier_cp, mid_i, 0)      |
                     _shift(barrier_cp, 0,      mid_j))

            # M[k] |= (_shift(barrier_cp, mid_i, mid_j))
    return M


# def _blocked_mask(barrier_cp: cp.ndarray,
#                   dirs: list[tuple[int,int]]) -> cp.ndarray:
#     """
#     Return a Boolean tensor M  (n_dirs, H, W)  where
#       M[k,i,j] == True → the step (i,j) → (i+di,j+dj) is disallowed.
#
#     Rules:
#       • Near diagonal (±1,±1): classic corner-cut → (0,dj) OR (di,0).
#       • Far diagonal (|di|==|dj|>1): block if ANY of
#           { origin flanks (sign(di),0) or (0,sign(dj)),
#             destination flanks (di-sign(di),dj) or (di,dj-sign(dj)),
#             midpoint (di//2,dj//2) } is a barrier.
#       • Knight (2,1) / (1,2): long-hop checks → midpoint and flanks.
#       • Orthogonal long hops (±2,0)/(0,±2): midpoint/flanks.
#     """
#     n_dirs, H, W = len(dirs), *barrier_cp.shape
#     M = cp.zeros((n_dirs, H, W), dtype=cp.bool_)
#
#     for k, (di, dj) in enumerate(dirs):
#         if di and dj:
#             # near diagonal (±1,±1) → classic corner-cut
#             if abs(di) == 1 and abs(dj) == 1:
#                 M[k] |= _shift(barrier_cp, 0,  dj) | _shift(barrier_cp, di, 0)
#
#             # far diagonal (±2,±2, …) → OR over origin/dest flanks + midpoint
#             elif abs(di) == abs(dj):
#                 si = 1 if di > 0 else -1
#                 sj = 1 if dj > 0 else -1
#                 origin_flanks = _shift(barrier_cp, si, 0) | _shift(barrier_cp, 0, sj)
#                 dest_flanks   = _shift(barrier_cp, di - si, dj) | _shift(barrier_cp, di, dj - sj)
#                 midpoint      = _shift(barrier_cp, di // 2, dj // 2)
#                 M[k] |= origin_flanks | dest_flanks | midpoint
#
#             # knight (2,1) / (1,2) → long-hop checks
#             else:
#                 mid_i, mid_j = di // 2, dj // 2
#                 M[k] |= (_shift(barrier_cp, mid_i, mid_j) |
#                          _shift(barrier_cp, mid_i, 0)      |
#                          _shift(barrier_cp, 0,      mid_j))
#
#
#
#         else:
#             # orthogonal long hops (±2,0)/(0,±2)
#             if abs(di) > 1 or abs(dj) > 1:
#                 mid_i, mid_j = di // 2, dj // 2
#                 M[k] |= (_shift(barrier_cp, mid_i, mid_j) |
#                          _shift(barrier_cp, mid_i, 0)      |
#                          _shift(barrier_cp, 0,      mid_j))
#
#     return M


_BLOCKED_CACHE: dict[str, cp.ndarray] = {}

def get_blocked_mask(tif_path: str,
                     barrier_cp: cp.ndarray,
                     dirs: list[tuple[int,int]]) -> cp.ndarray:
    key = str(tif_path)
    if key not in _BLOCKED_CACHE:
        _BLOCKED_CACHE[key] = _blocked_mask(barrier_cp, dirs)
    return _BLOCKED_CACHE[key]
########################################
# 1) Custom Kernel for the Fire Arrival Update
########################################

# FIRE_UPDATE_SRC = r'''
# extern "C" __global__
# void fire_update(
#     const float* __restrict__ T_old,   // shape (rows*cols)
#     float* __restrict__ T_new,         // shape (rows*cols)
#     const float* __restrict__ delays,  // shape (n_dirs*rows*cols)
#     const int* __restrict__ offsets,   // shape (n_dirs*2)
#     const int  rows,
#     const int  cols,
#     const int  n_dirs
# ) {
#     // 2D global index
#     int i = blockDim.x * blockIdx.x + threadIdx.x;
#     int j = blockDim.y * blockIdx.y + threadIdx.y;
#     if (i >= rows || j >= cols) return;
#
#     // Flattened index
#     int idx = i * cols + j;
#     float old_val = T_old[idx];
#     float best_val = old_val;  // Keep track of the minimum arrival time
#
#     // For each direction k
#     for(int k = 0; k < n_dirs; k++){
#         int di = offsets[2*k + 0];
#         int dj = offsets[2*k + 1];
#         int ni = i + di;
#         int nj = j + dj;
#         // Check bounds
#         if(ni >= 0 && ni < rows && nj >= 0 && nj < cols){
#             // neighbor's index
#             int n_idx = ni * cols + nj;
#             // delays[k, ni, nj] => flattened => k*(rows*cols) + n_idx
#             float delay_val = delays[k * rows * cols + n_idx];
#             float cand = T_old[n_idx] + delay_val;
#             if(cand < best_val){
#                 best_val = cand;
#             }
#         }
#     }
#
#     T_new[idx] = best_val;
# }
# ''';
#
# fire_update_kernel = cp.RawKernel(
#     code=FIRE_UPDATE_SRC,
#     name='fire_update'
# )
#
# # ── CUDA kernel that ignores candidates later than time_cutoff ───────────
# FIRE_UPDATE_CUTOFF_SRC = r'''
# extern "C" __global__
# void fire_update_cutoff(
#     const float* __restrict__ T_old,     // (rows*cols)
#     float*       __restrict__ T_new,     // (rows*cols)
#     const float* __restrict__ delays,    // (n_dirs*rows*cols)
#     const int*   __restrict__ offsets,   // (n_dirs*2)
#     const int    rows,
#     const int    cols,
#     const int    n_dirs,
#     const float  time_cutoff             // RELATIVE minutes
# ){
#     int i = blockDim.x * blockIdx.x + threadIdx.x;
#     int j = blockDim.y * blockIdx.y + threadIdx.y;
#     if (i >= rows || j >= cols) return;
#
#     int idx = i * cols + j;
#     float best_val = T_old[idx];           // start from current value
#
#     for (int k = 0; k < n_dirs; ++k){
#         int ni = i + offsets[2*k    ];
#         int nj = j + offsets[2*k + 1];
#         if (ni < 0 || ni >= rows || nj < 0 || nj >= cols) continue;
#
#         int n_idx  = ni * cols + nj;
#         float cand = T_old[n_idx] + delays[k*rows*cols + n_idx];
#
#         // -- keep only candidates that arrive within this phase
#         if (cand <= time_cutoff && cand < best_val){
#             best_val = cand;
#         }
#     }
#     T_new[idx] = best_val;
# }
# ''';
#
# fire_update_kernel_cutoff = cp.RawKernel(
#     code=FIRE_UPDATE_CUTOFF_SRC,
#     name='fire_update_cutoff'
# )
FUSED_SRC = r'''
extern "C" __global__
void fire_step_cutoff(
      const float* __restrict__ T_in,       // rows*cols
            float* __restrict__ T_out,      // rows*cols
      const float* __restrict__ delays,     // n_dirs*rows*cols
      const int*   __restrict__ offsets,    // n_dirs*2
      const bool*  __restrict__ burnable,   // rows*cols
      const int    rows,  const int cols,
      const int    n_dirs,
      const float  t_cutoff,                // relative minutes
            float* __restrict__ g_diff)     // single-element buffer
{
    const float INF = __int_as_float(0x7f800000);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= cols) return;

    int idx  = i * cols + j;
    float best = T_in[idx];

    // stencil sweep
    #pragma unroll
    for (int k=0; k<n_dirs; ++k) {
        int di = offsets[2*k    ];
        int dj = offsets[2*k + 1];
        int ni = i + di,  nj = j + dj;
        if (0 <= ni && ni < rows && 0 <= nj && nj < cols) {
            int nidx = ni * cols + nj;
            float cand = T_in[nidx] + delays[k*rows*cols + nidx];
            best = cand < best ? cand : best;
        }
    }

    // phase cut-off & burnability
    if (!burnable[idx] || best > t_cutoff) best = INF;

    // write result
    T_out[idx] = best;

    // local 
    float d = fabsf(best - T_in[idx]);

    // warp-reduce to max
    for (int off=16; off; off >>= 1)
        d = fmaxf(d, __shfl_down_sync(0xffffffff, d, off));

    if ((threadIdx.x & 31)==0)
        atomicMax(reinterpret_cast<int*>(g_diff), __float_as_int(d));
}
'''.replace('\t',' ');          # keep ASCII only

fire_step_cutoff = cp.RawKernel(FUSED_SRC, "fire_step_cutoff")

def launch_fused_step(T_in, T_out,
                      delays, offsets, burnable,
                      t_cutoff, d_diff):
    rows, cols = T_in.shape
    n_dirs      = offsets.shape[0]
    block       = (16,16)
    grid        = ((rows+15)//16, (cols+15)//16)

    fire_step_cutoff(grid, block,
        (T_in.ravel(),  T_out.ravel(),
         delays.ravel(), offsets.ravel(), burnable.ravel(),
         rows, cols, n_dirs,
         np.float32(t_cutoff),
         d_diff)
    )
#
# def launch_fire_update_kernel(T_old, T_new, delays, offsets):
#     """
#     Wrap the raw kernel call in Python.
#       T_old, T_new: (rows, cols) float32
#       delays: (n_dirs, rows, cols) float32
#       offsets: (n_dirs, 2) int32
#     """
#     rows, cols = T_old.shape
#     n_dirs = offsets.shape[0]
#
#     # Flatten the arrays for kernel
#     T_old_flat = T_old.ravel()
#     T_new_flat = T_new.ravel()
#     delays_flat = delays.ravel()
#     offsets_flat = offsets.ravel()
#
#     # Grid/block dims
#     block = (16, 16)  # tune if needed
#     grid = ((rows + block[0] - 1)//block[0],
#             (cols + block[1] - 1)//block[1])
#
#     fire_update_kernel(
#         grid,
#         block,
#         (
#             T_old_flat,
#             T_new_flat,
#             delays_flat,
#             offsets_flat,
#             rows,
#             cols,
#             n_dirs
#         )
#     )
#
#
# def launch_fire_update_kernel_cutoff(T_old, T_new,
#                                      delays, offsets,
#                                      time_cutoff):
#     rows, cols = T_old.shape
#     n_dirs      = offsets.shape[0]
#
#     block = (16, 16)
#     grid  = ((rows + block[0] - 1)//block[0],
#              (cols + block[1] - 1)//block[1])
#
#     fire_update_kernel_cutoff(
#         grid, block,
#         (
#             T_old.ravel(),
#             T_new.ravel(),
#             delays.ravel(),
#             offsets.ravel(),
#             rows, cols, n_dirs,
#             np.float32(time_cutoff)
#         )
#     )


########################################
# 2) Build the Delay Stack
########################################

# def build_delay_stack(ros_array, directions_list, cell_size_ft=98.4, eps=1e-6):
#     """
#     Given:
#       ros_array: shape (rows, cols, n_dirs)
#       directions_list: list of (di, dj) of length n_dirs
#     Return:
#       delays: shape (n_dirs, rows, cols),
#               where delays[k, i, j] = time (minutes) to go from cell (i,j) outward in direction k
#       offsets: shape (n_dirs, 2) (same order as directions_list)
#     By default cell_size_ft = 30m ~ 98.4ft.
#     """
#     rows, cols, n_dirs = ros_array.shape
#     directions_arr = np.array(directions_list, dtype=np.int32)
#     # The distance factor for each direction (e.g. sqrt(di^2 + dj^2))
#     multipliers = [np.sqrt((di**2 + dj**2)) for (di, dj) in directions_list]
#     multipliers_cp = cp.asarray(multipliers, dtype=cp.float32)
#
#     # We'll build delays on GPU
#     ros_cp = cp.asarray(ros_array, dtype=cp.float32)
#     delays_cp = cp.zeros((n_dirs, rows, cols), dtype=cp.float32)
#
#     for k in range(n_dirs):
#         # distance in feet for direction k
#         distance = cell_size_ft * multipliers_cp[k]
#         # delay = distance / ros
#         # ros_cp[..., k] has shape (rows, cols)
#         delay_k = distance / (ros_cp[..., k] + eps)
#         delays_cp[k, :, :] = delay_k
#
#     offsets_cp = cp.asarray(directions_arr, dtype=cp.int32)
#     return delays_cp, offsets_cp
# ---------------------------------------------------------------------
# Vectorised GPU helper – replaces the old per-direction loop
# ---------------------------------------------------------------------
def build_delay_stack(ros_cp: cp.ndarray,
                      directions: list[tuple[int, int]],
                      cell_size_ft: float = 98.4,
                      eps: float = 1e-6
                      ) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Convert the 3-D ROS tensor produced by `compute_landscape_ros`
    (shape: (rows, cols, n_dirs)) into a delay stack
    (shape: (n_dirs, rows, cols)) in a single broadcasted CuPy op.

    Parameters are **identical** to the removed helper, so external
    callers do not need to change.
    """
    # ---- 1. fixed data that only depends on the neighbour set ----------
    di_dj = cp.asarray(directions, dtype=cp.int32)            # (n_dirs, 2)
    #   √(di²+dj²) gives the grid-to-grid distance multiplier
    distance_multiplier = cp.sqrt(
        (di_dj[:, 0] ** 2 + di_dj[:, 1] ** 2)
    ).astype(cp.float32)                                      # (n_dirs,)

    # ---- 2. reshape the ROS tensor so the direction axis is first ------
    #   ROS comes out of the mega-kernel as (rows, cols, n_dirs)
    ros_dir_first = ros_cp.transpose(2, 0, 1).astype(cp.float32)  # (n_dirs, H, W)

    # ---- 3. broadcasted division: one CUDA kernel instead of 24 ----------
    distance_cp = (cell_size_ft * distance_multiplier)[:, None, None]  # (n_dirs,1,1)
    delays_cp   = distance_cp / (ros_dir_first + eps)                  # (n_dirs,H,W)

    return delays_cp, di_dj        # same return types as before



# def run_phase_gpu(
#         tiff_path: str,
#         grid_size: tuple[int, int],
#         wind_speed: float,
#         wind_dir_deg: float,
#         iteration_limit: int,
#         time_cutoff: float,
#         tol: float = 1e-4,
#         initial_T: cp.ndarray | None = None,
#         fuel_np: np.ndarray | None = None
#     ) -> cp.ndarray:
#     """
#     One propagation phase using the custom kernel + cut-off.
#     Times are *relative* inside the phase; anything > time_cutoff
#     is treated as ∞ and therefore cannot ignite subsequent phases.
#     """
#     rows, cols = grid_size
#
#     # 1. build delay stack for this wind
#     ros_cp, directions = compute_landscape_ros(
#         tiff_path,
#         wind_speed=wind_speed,
#         wind_direction_deg=wind_dir_deg,
#         wind_adjustment=1.045,
#         radius=2
#     )
#     delays_cp, offsets_cp = build_delay_stack(ros_cp, directions)
#
#     # 2. initial arrival field
#     if initial_T is None:
#         T_old = cp.full((rows, cols), cp.inf,  dtype=cp.float32)
#         T_old[rows//2, cols//2] = 0.0
#     else:
#         T_old = initial_T.astype(cp.float32)
#     T_new = T_old.copy()
#
#     # 3. burnable mask from fuel map (optional)
#     if fuel_np is not None:
#         fuel_cp = cp.asarray(fuel_np)
#         burnable_mask = ~((fuel_cp >= 90) & (fuel_cp <= 99))
#     else:
#         burnable_mask = cp.ones((rows, cols), dtype=cp.bool_)
#
#     T_old[~burnable_mask] = cp.inf
#     for it in range(iteration_limit):
#         launch_fire_update_kernel_cutoff(
#             T_old, T_new, delays_cp, offsets_cp,
#             time_cutoff
#         )
#         diff = cp.max(cp.abs(T_new - T_old))
#         T_old, T_new = T_new, T_old          # swap
#         T_old[~burnable_mask] = cp.inf       # enforce mask
#         if diff < tol:
#             break
#     return T_old

def run_phase_gpu(
        tiff_path: str,
        grid_size: tuple[int, int],
        wind_speed: float,
        wind_dir_deg: float,
        iteration_limit: int,
        time_cutoff: float,
        tol: float | None = None,
        initial_T: cp.ndarray | None = None,
        fuel_np: np.ndarray | None = None
    ) -> cp.ndarray:
    """
    Single relative-time propagation phase, GPU-only and memory-tight.

    *   Uses one fused CUDA kernel (`fire_step_cutoff`) that
        - scans neighbours,
        - enforces the burnable mask,
        - applies the phase cut-off, **and**
        - records the max |ΔT| with an in-kernel atomic.
    *   Converges when that device-side max error < `tol`.
    *   Returns the updated arrival-time field **in relative minutes**.
    """
    rows, cols = grid_size
    #1.039 for three cities
    #1.05 for esperenze
    #1.04 for marshall
    #1.075 for camp

    # 1 ── delay stack for this wind ------------------------------------
    ros_cp, directions = compute_landscape_ros(
        tiff_path,
        wind_speed          = wind_speed,
        wind_direction_deg  = wind_dir_deg,
        wind_adjustment     = 1.05,
        radius              = 2
    )
    delays_cp, offsets_cp = build_delay_stack(ros_cp, directions)

    barrier_cp = get_barrier_mask(tiff_path)  # cached CuPy 2‑D array
    blocked = get_blocked_mask(tiff_path, barrier_cp, directions)

    INF = cp.float32(1e30)
    delays_cp = cp.where(blocked, INF, delays_cp)


    # barrier_cp = get_barrier_mask(tiff_path)  # ← new
    # delays_cp = apply_barrier_rules(delays_cp, barrier_cp, directions)
    # 2 ── initial arrival map ------------------------------------------
    if initial_T is None:
        T_old = cp.full((rows, cols), cp.inf, dtype=cp.float32)
        T_old[rows // 2, cols // 2] = 0.0          # centre ignition
    else:
        T_old = initial_T.astype(cp.float32)
    T_new = T_old.copy()                           # ping-pong buffer

    # 3 ── burnable mask (True = can burn) ------------------------------
    if fuel_np is not None:
        fuel_cp = cp.asarray(fuel_np)
        burnable_mask = ~((fuel_cp >= 90) & (fuel_cp <= 99))
    else:
        burnable_mask = cp.ones((rows, cols), dtype=cp.bool_)

    # un-burnable cells start at ∞
    T_old = cp.where(burnable_mask, T_old, cp.inf)

    # 4 ── device buffer for convergence test ---------------------------
    d_diff = cp.zeros(1, dtype=cp.float32)         # holds max |ΔT|

    # 5 ── iteration loop ----------------------------------------------
    for _ in range(iteration_limit):
        d_diff.fill(0)                             # reset on GPU

        launch_fused_step(                         # one kernel launch
            T_old, T_new,
            delays_cp, offsets_cp, burnable_mask,
            time_cutoff, d_diff
        )

        # # Only early-stop (and perform the host sync) if tol > 0
        # if tol is not None and tol > 0:
        #     if float(d_diff.get()[0]) < tol:
        #         break

        T_old, T_new = T_new, T_old               # swap buffers

    return T_old


########################################
# 3) Fire Arrival Solver (iterative) using the Custom Kernel
########################################

# def run_deterministic_arrival_with_ros(
#     tiff_path, grid_size, wind_speed, wind_direction_deg, max_iter=1000, tol=1e-4
# ):
#     """
#     1. Compute the directional ROS using `compute_landscape_ros`.
#     2. Convert the ROS to a delay stack (delay_maps).
#     3. Iteratively update T using the custom CUDA kernel.
#     """
#     rows, cols = grid_size
#
#     # 1) Compute the directional ROS
#     ros_array, directions_list = compute_landscape_ros(
#         tiff_path,
#         wind_speed=wind_speed,
#         wind_direction_deg=wind_direction_deg,
#         wind_adjustment=1.05,  # example multiplier
#         radius=2
#     )
#     # 2) Build the delay stack => shape (n_dirs, rows, cols)
#     #    and offsets => shape (n_dirs, 2)
#     delays_cp, offsets_cp = build_delay_stack(ros_array, directions_list)
#
#     # 3) Initialize T
#     T_old = cp.full((rows, cols), cp.inf, dtype=cp.float32)
#     center = (rows // 2, cols // 2)
#     T_old[center] = 0.0
#     T_new = T_old.copy()
#
#     # Iterative solver
#     for it in range(max_iter):
#         # Launch custom kernel
#         launch_fire_update_kernel(T_old, T_new, delays_cp, offsets_cp)
#
#         # Measure update magnitude
#         diff = cp.max(cp.abs(T_new - T_old))
#         # Swap references instead of doing T_old = T_new.copy()
#         T_old, T_new = T_new, T_old
#
#         if diff < tol:
#             print(f"Converged after {it+1} iterations. Diff={diff}")
#             break
#     else:
#         print("Reached maximum iterations without full convergence.")
#
#     return T_old
# def run_deterministic_arrival_with_ros(
#     tiff_path, grid_size, wind_speed, wind_direction_deg, max_iter=1000, tol=1e-4, fuel_np=None
# ):
#     rows, cols = grid_size
#
#     # 1) Compute the directional ROS and directions_list as before.
#     ros_array, directions_list = compute_landscape_ros(
#         tiff_path,
#         wind_speed=wind_speed,
#         wind_direction_deg=wind_direction_deg,
#         wind_adjustment=1.045,  # example multiplier; adjust as needed
#         radius=2
#     )
#     delays_cp, offsets_cp = build_delay_stack(ros_array, directions_list)
#
#     # 2) Initialize the arrival time T
#     T_old = cp.full((rows, cols), cp.inf, dtype=cp.float32)
#     center = (rows // 2, cols // 2)
#     T_old[center] = 0.0
#
#     # 3) Compute a burnable mask from the fuel map if provided.
#     if fuel_np is not None:
#         # Unburnable cells are those with fuel model IDs between 90 and 99 (inclusive)
#         fuel_np_cp = cp.asarray(fuel_np)
#         burnable_mask = ~((fuel_np_cp >= 90) & (fuel_np_cp <= 99))
#     else:
#         burnable_mask = cp.ones((rows, cols), dtype=cp.bool_)
#
#     # Ensure unburnable cells start with infinite arrival times.
#     T_old[~burnable_mask] = cp.inf
#     T_new = T_old.copy()
#
#     # 4) Iterative solver: launch the custom kernel and then enforce the mask after each iteration.
#     for it in range(max_iter):
#         launch_fire_update_kernel(T_old, T_new, delays_cp, offsets_cp)
#         diff = cp.max(cp.abs(T_new - T_old))
#         # Swap T_old and T_new for the next iteration
#         T_old, T_new = T_new, T_old
#         # Enforce that unburnable cells never update (remain at infinity)
#         T_old[~burnable_mask] = cp.inf
#         if diff < tol:
#             print(f"Converged after {it+1} iterations. Diff={diff}")
#             break
#     else:
#         print("Reached maximum iterations without full convergence.")
#
#     return T_old

def run_deterministic_arrival_with_ros(
    tiff_path: str,
    grid_size: tuple[int, int],
    wind_speed: float,
    wind_direction_deg: float,
    max_iter: int = 1000,
    tol: float | None = None,
    fuel_np: np.ndarray | None = None,
) -> cp.ndarray:
    """
    Constant-wind solve that reuses the fused kernel.
    The cut-off is set to a very large number so it never triggers.
    """
    rows, cols = grid_size

    # build delay stack -------------------------------------------------
    ros_cp, dirs = compute_landscape_ros(
        tiff_path,
        wind_speed          = wind_speed,
        wind_direction_deg  = wind_direction_deg,
        wind_adjustment     = 1.057,
        radius              = 2,
        # dir_exp             = 2
    )
    delays_cp, offsets_cp = build_delay_stack(ros_cp, dirs)
    # ─── permanent‑barrier veto (corner‑cut + long‑hop) ────────────────
    barrier_cp = get_barrier_mask(tiff_path)  # cached on first call
    delays_cp = apply_barrier_rules(delays_cp, barrier_cp, dirs)
    # initial arrival map ----------------------------------------------
    T_old = cp.full((rows, cols), cp.inf, dtype=cp.float32)
    T_old[rows // 2, cols // 2] = 0.0
    T_new = T_old.copy()

    # burnable mask -----------------------------------------------------
    if fuel_np is not None:
        burnable = ~( (cp.asarray(fuel_np) >= 90) & (cp.asarray(fuel_np) <= 99) )
    else:
        burnable = cp.ones((rows, cols), dtype=cp.bool_)
    T_old = cp.where(burnable, T_old, cp.inf)

    # convergence buffer -----------------------------------------------
    d_diff = cp.zeros(1, dtype=cp.float32)
    STATIC_INF = np.float32(1e30)          # “∞” cut-off

    # iterations --------------------------------------------------------
    for _ in range(max_iter):
        d_diff.fill(0)
        launch_fused_step(
            T_old, T_new,
            delays_cp, offsets_cp, burnable,
            STATIC_INF,            # ← never reached
            d_diff
        )
        # if tol is not None and tol > 0:
        #     if float(d_diff.get()[0]) < tol:
        #         break
        T_old, T_new = T_new, T_old

    return T_old


def run_multi_phase_arrival_with_ros(
        tiff_path: str,
        grid_size: tuple[int, int],
        wind_schedule: list[tuple[float, float, float, float]],
        iteration_limit: int = 100,
        tol: float = 1e-4,
        fuel_np: np.ndarray | None = None
    ) -> cp.ndarray:
    """
    Identical relative-time scheme as the roth_w version, but
    implemented with the custom kernel.
    """
    rows, cols = grid_size
    T_abs = cp.full(grid_size, cp.inf, dtype=cp.float32)
    T_abs[rows//2, cols//2] = 0.0          # centre ignition
    total_time_offset = 0.0

    for t_start, t_end, ws, wd in wind_schedule:
        phase_dur = t_end - t_start
        # print(f"Phase {t_start:>6.0f}–{t_end:>6.0f}  "
        #       f"{ws:>5.1f} mph @ {wd:>3.0f}°")

        # ignition set = cells already reached
        T_rel0 = cp.where(T_abs <= total_time_offset, 0.0, cp.inf)

        # propagate for ≤ phase_dur minutes
        T_rel = run_phase_gpu(
            tiff_path, grid_size,
            ws, wd,
            iteration_limit,
            time_cutoff = phase_dur,
            tol = tol,
            initial_T = T_rel0,
            fuel_np = fuel_np
        )

        # commit to absolute timeline
        T_abs = cp.minimum(T_abs, T_rel + total_time_offset)
        total_time_offset += phase_dur

    return T_abs



def _print_schedule(label, schedule, max_rows: int = 30):
    """
    Pretty-prints a wind schedule with a short hash so you can tell
    which schedule was actually run (useful when you tweak inputs).
    """
    import hashlib, json, sys
    h = hashlib.sha1(json.dumps(schedule).encode()).hexdigest()[:8]
    PINK, RESET = "\033[95m", "\033[0m"
    print(f"\n{PINK}{label}  |  segments = {len(schedule)}  |  hash = {h}{RESET}")
    for row in schedule[:max_rows]:
        print(f"{PINK}   {row}{RESET}")
    if len(schedule) > max_rows:
        print(f"{PINK}    …{RESET}")


def compute_ros_field(T_np, cell_size_ft, clip_max=100000.0):
    """
    Compute the local rate-of-spread (ROS) from the fire arrival time field T_np.

    This version replaces infinite values with the maximum finite arrival time
    for the gradient computation, then re-masks unburned regions. For visualization,
    the ROS field is clipped to a maximum value (default: 200 ft/min).

    Parameters:
      T_np         : 2D NumPy array of arrival times (minutes)
      cell_size_ft : grid cell size (ft)
      clip_max     : maximum ROS value (ft/min) to display

    Returns:
      A 2D NumPy array of ROS (ft/min), clipped for visualization.
    """
    T_np_clean = T_np.copy()
    finite_mask = np.isfinite(T_np)
    if np.any(finite_mask):
        max_finite = np.max(T_np[finite_mask])
    else:
        max_finite = 0.0
    T_np_clean[~finite_mask] = max_finite

    # Compute gradients along y and x; np.gradient assumes spacing of cell_size_ft
    dT_dy, dT_dx = np.gradient(T_np_clean, cell_size_ft)
    grad_T = np.sqrt(dT_dx ** 2 + dT_dy ** 2)

    # Prevent division by zero by enforcing a small minimum gradient value.
    grad_T[grad_T < 1e-6] = 1e-6
    ros_field = 1.0 / grad_T  # Unit: ft/min

    # Re-mask unburned regions.
    ros_field[~finite_mask] = np.nan

    # Clip extreme ROS values for visualization purposes.
    ros_field = np.clip(ros_field, 0, clip_max)
    return ros_field


def get_perimeter_ros(T_np, time_threshold, tolerance_past=1.0, tolerance_future=5.0, cell_size_ft=98.4):
    """
    Returns the rate-of-spread only for cells that are near the fire perimeter.

    A cell is considered part of the fire perimeter if its arrival time is within ±tolerance
    of the specified time_threshold.

    Parameters:
      T_np          : 2D NumPy array of arrival times (minutes)
      time_threshold: the simulation time that defines the perimeter (minutes)
      tolerance     : the allowed deviation (in minutes) for a cell to count as on the perimeter
      cell_size_ft  : spatial resolution of each grid cell (ft)

    Returns:
      A 2D NumPy array of ROS values (ft/min) for perimeter cells (other cells set to NaN).
    """
    # Compute the complete ROS field safely.
    ros_field = compute_ros_field(T_np, cell_size_ft, clip_max=100000.0)
    # Build a mask for cells near the fire perimeter.

    # asymmetric perimeter mask
    mask = (
            (T_np >= (time_threshold - tolerance_past)) &
            (T_np <= (time_threshold + tolerance_future))
    )
    perimeter_ros = np.where(mask, ros_field, np.nan)
    return perimeter_ros

########################################
# 4) SurrogateFireModelROS (uses the new solver)
########################################

class SurrogateFireModelROS:
    """
    Surrogate Fire Model using GPU acceleration with high-fidelity ROS from rothermal_2d_wind.py
    Now uses a custom kernel in the arrival-time solver.
    """

    def __init__(self, tif_path, sim_time, wind_speed, wind_direction_deg,
                 max_iter=200, tol: float | None = None, wind_schedule: list[tuple[float,float,float,float]] | None = None, fuel_model_override: np.ndarray | None = None):
        self.tif_path = Path(tif_path).resolve()
        with rasterio.open(self.tif_path) as src:
            self._bounds = src.bounds
            self._transform = src.transform
            landscape = src.read()  # shape: (bands, height, width)

        # Assume band order: 1=Elevation, 2=Slope, 3=Aspect, 4=Fuel model
        self.elevation_np = landscape[0].astype(np.float32)
        self.slope_np = landscape[1].astype(np.float32)
        self.aspect_np = landscape[2].astype(np.float32)
        self.fuel_np = landscape[3].astype(np.int32)

        # if there’s a 10th band, treat it as “built characteristic”
        if landscape.shape[0] >= 10:
            # band index 9 → 10th band
            self.built_char_np = landscape[9].astype(np.int32)
        else:
            self.built_char_np = None

        # ── NEW: keep a CuPy copy once, so later calls never re-upload ──
        self._built_char_cp = (
            cp.asarray(self.built_char_np, dtype=cp.int32)
            if self.built_char_np is not None else None
        )


        # ── optional override supplied by the caller ─────────────────────────
        if fuel_model_override is not None:
            if fuel_model_override.shape != self.fuel_np.shape:
                raise ValueError("fuel_model_override has the wrong shape "
                                 f"(got {fuel_model_override.shape}, "
                                 f"expected {self.fuel_np.shape})")
            self.fuel_np = fuel_model_override.copy()

        self.grid_size = self.elevation_np.shape

        # Optional: load any needed JSON for fuel data, etc.
        with open("burned_area_results_long.json", "r") as f:
            fuel_data = json.load(f)
        fuel_index_np = np.ones_like(self.fuel_np, dtype=np.float32)
        for key, value in fuel_data.items():
            fuel_model = int(key)
            if 90 <= fuel_model <= 99:
                continue
            fuel_index_np[self.fuel_np == fuel_model] = float(value["Fuel_Index"])
        self.fuel_index_np = fuel_index_np

        self.sim_time = sim_time
        self.wind_speed = wind_speed
        self.wind_direction_deg = wind_direction_deg
        self.max_iter = max_iter
        self.tol = tol
        self.wind_schedule = wind_schedule
        # Solve for arrival times with our new custom-kernel approach
        self._compute_arrival_map()

        # For compatibility:
        self.fuel_model = self.fuel_np.copy()

    # def _compute_arrival_map(self):
    #     T_cp = run_deterministic_arrival_with_ros(
    #         str(self.tif_path),
    #         self.grid_size,
    #         self.wind_speed,
    #         self.wind_direction_deg,
    #         max_iter=self.max_iter,
    #         tol=self.tol
    #     )
    #     cp.cuda.Stream.null.synchronize()
    #     self.T = T_cp
    # def _compute_arrival_map(self):
    #     T_cp = run_deterministic_arrival_with_ros(
    #         str(self.tif_path),
    #         self.grid_size,
    #         self.wind_speed,
    #         self.wind_direction_deg,
    #         max_iter=self.max_iter,
    #         tol=self.tol,
    #         fuel_np=self.fuel_np  # Pass the fuel map so that unburnable cells can be handled.
    #     )
    #     cp.cuda.Stream.null.synchronize()
    #     self.T = T_cp

    def _compute_arrival_map(self):
        if self.wind_schedule is not None:
            print("Using multi-phase propagation with wind schedule.")
            T_cp = run_multi_phase_arrival_with_ros(
                str(self.tif_path),
                self.grid_size,
                self.wind_schedule,
                iteration_limit=self.max_iter,
                tol=self.tol,
                fuel_np=self.fuel_np
            )
        else:
            print("Using deterministic (constant-wind) propagation.")
            T_cp = run_deterministic_arrival_with_ros(
                str(self.tif_path),
                self.grid_size,
                self.wind_speed,
                self.wind_direction_deg,
                max_iter=self.max_iter,
                tol=self.tol,
                fuel_np=self.fuel_np
            )
        cp.cuda.Stream.null.synchronize()
        self.T = T_cp

    @property
    def ignition_pt(self):
        return ((self.bounds.left + self.bounds.right) / 2,
                (self.bounds.bottom + self.bounds.top) / 2)

    def current_fire(self, time, max_time=None):
        """
        Return an array in which cells with arrival times <= time are 0 (burned),
        else keep arrival time, optionally mask out those above max_time with NaN.
        """
        T_np = cp.asnumpy(self.T)
        fire_state = T_np.copy()
        fire_state[fire_state <= time] = 0
        if max_time is not None:
            fire_state = np.where(fire_state > max_time, np.nan, fire_state)
        return fire_state

    def current_fire_cp(self):
        """GPU version of current_fire(): returns CuPy array, no copy."""
        return self.T  # just hand back the CuPy view

    def re_run(self, new_fuel_map):
        """
        Re-run the solver if the user updates the fuel map (e.g., retardant).
        """
        self.fuel_np = new_fuel_map.copy()
        self._compute_arrival_map()
        self.fuel_model = self.fuel_np.copy()
        return cp.asnumpy(self.T)

    def fuel_editor_numpy(self, rect, old_fuel: np.ndarray, new_id: int):
        new_fuel = old_fuel.copy()
        drop_array = np.zeros_like(old_fuel)
        x_coords, y_coords = zip(*rect)
        rr, cc = polygon(y_coords, x_coords, old_fuel.shape)
        new_fuel[rr, cc] = new_id
        drop_array[rr, cc] = 1
        return new_fuel, drop_array

    def calculate_fire_score(self, time_threshold):
        burning_cells = cp.sum(self.T <= time_threshold)
        return int(burning_cells.get())

    def calculate_area_score(self, time_threshold: float) -> int:
        """Number of burned cells (arrival time ≤ threshold)."""
        burned = self.T <= time_threshold        # CuPy bool mask
        return int(cp.sum(burned).get())

    def calculate_building_score(self, time_threshold: float) -> int:
        """Number of burned residential buildings (built_char ≥ 11)."""
        if self.built_char_np is None:
            return 0
        burned = self.T <= time_threshold
        built  = cp.asarray(self.built_char_np >= 11)
        return int(cp.sum(burned & built).get())

        # ────────────────────────────────────────────────────────────────
        #  Fast per-code building breakdown  {11..25 → count}
        # ────────────────────────────────────────────────────────────────
    def calculate_building_breakdown(self, time_threshold: float) -> dict[int, int]:
        """
        GPU-side tally of every settlement-code ≥ 11 that burns by *time_threshold*.
        Returns a compact {code: count} dict (missing keys mean zero).
        Runs entirely on the GPU; the only host transfer is a tiny bincount.
        """
        if self._built_char_cp is None:
            return {}

        burned = self.T <= time_threshold
        mask = burned & (self._built_char_cp >= 11)
        if not mask.any():
            return {}

        # bincount on GPU → 0-25 array → pull back to host
        counts = cp.bincount(self._built_char_cp[mask], minlength=26).get()

        return {code: int(counts[code])
                for code in (11, 12, 13, 14, 15, 21, 22, 23, 24, 25)
                if counts[code] > 0}



    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, val):
        self._bounds = val

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, val):
        self._transform = val

    # ------------------------------------------------------------------
    #  ███►  Per-sector perimeter ROS utilities  ◄███
    # ------------------------------------------------------------------
    def aggregate_perimeter_ros_by_increments(self, angle_ranges, statistic='median',
                                              time_threshold=None, tolerance_past=1.0, tolerance_future=5.0,
                                              cell_size_ft=98.4):
        """
        Aggregate the local rate-of-spread (ROS) on the fire perimeter over specified angular ranges.

        The angular ranges are defined in the same coordinate system as used by the agents.
        Only cells on the perimeter (i.e. those with arrival times within ±tolerance of time_threshold)
        are considered.

        Parameters:
          angle_ranges: A list of tuples representing angular ranges in degrees,
                         e.g., [(0, 5), (5, 10), (10, 15)].
          statistic    : 'median' or 'mean' for the aggregation.
          time_threshold: The arrival time threshold (in minutes) that defines the fire perimeter.
                          (Required for perimeter-only analysis.)
          tolerance    : The allowed deviation (in minutes) for a cell to be considered on the perimeter.
          cell_size_ft : The spatial resolution (in feet) of each grid cell.

        Returns:
          A dictionary mapping each range (tuple) to the aggregated ROS (ft/min).
          (Cells not on the perimeter are ignored; if no cells fall in a range, the value is NaN.)
        """
        if time_threshold is None:
            raise ValueError("time_threshold must be provided for perimeter-only aggregation.")

        # Obtain the ROS field only on the fire perimeter.
        T_np = cp.asnumpy(self.T)
        ros_field = get_perimeter_ros(T_np, time_threshold, tolerance_past, tolerance_future, cell_size_ft)

        # Build a grid of cell centers using the model's affine transform.
        rows, cols = T_np.shape
        rows_idx = np.arange(rows)
        cols_idx = np.arange(cols)
        cols_grid, rows_grid = np.meshgrid(cols_idx, rows_idx)
        # For a north-up image the transform is (a, b, c, d, e, f):
        # x = c + a*(col + 0.5),  y = f + e*(row + 0.5).
        a = self.transform[0]
        c = self.transform[2]
        e = self.transform[4]
        f_val = self.transform[5]
        x_grid = c + (cols_grid + 0.5) * a
        y_grid = f_val + (rows_grid + 0.5) * e

        # Compute the polar angle (in degrees) for each cell relative to the ignition point.
        ignition_x, ignition_y = self.ignition_pt
        angle_grid = np.degrees(np.arctan2(y_grid - ignition_y, x_grid - ignition_x))
        # Normalize to [0, 360)
        angle_grid = (angle_grid + 360) % 360

        aggregated = {}
        for (lower, upper) in angle_ranges:
            # Create mask: select cells within the angular bin and with valid (finite) ROS.
            mask = (angle_grid >= lower) & (angle_grid < upper) & np.isfinite(ros_field)
            values = ros_field[mask]
            if values.size > 0:
                if statistic == 'median':
                    aggregated[(lower, upper)] = np.median(values)
                elif statistic == 'mean':
                    aggregated[(lower, upper)] = np.mean(values)

                # ─── NEW: “mean-99”  → clip the top 1 % before averaging ─────
                elif statistic in ('mean99', 'mean-99', 'mean_trimmed_1'):
                    cutoff = np.nanpercentile(values, 99.5)  # 99th-percentile
                    trimmed = values[values <= cutoff]
                    aggregated[(lower, upper)] = np.mean(trimmed)
                else:
                    raise ValueError("statistic must be either 'median' or 'mean'")
            else:
                aggregated[(lower, upper)] = np.nan
        return aggregated


    @property
    def arrival_time_grid(self) -> np.ndarray:
        """Return the full arrival-time raster as a NumPy array."""
        return cp.asnumpy(self.T)




def plot_colored_perimeter_by_sector(model, time_threshold, tolerance_past=1.0, tolerance_future=5.0,
                                     cell_size_ft=98.4,
                                     save_path="perimeter_ros_plot.png",
                                     legend_save_path="sector_legend.png"):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from skimage import measure
    from matplotlib.lines import Line2D

    angle_ranges = [(i, i + 5) for i in range(0, 360, 5)]
    aggregated_ros = model.aggregate_perimeter_ros_by_increments(
        angle_ranges=angle_ranges,
        statistic='mean',
        time_threshold=time_threshold,
        tolerance_past=tolerance_past,
        tolerance_future=tolerance_future,
        cell_size_ft=cell_size_ft
    )

    T_np = cp.asnumpy(model.T)
    fire_mask = (T_np <= time_threshold)
    contours = measure.find_contours(fire_mask.astype(float), level=0.5)
    if len(contours) == 0:
        print("No fire perimeter found.")
        return

    longest = max(contours, key=lambda x: x.shape[0])
    a = model.transform[0]
    c = model.transform[2]
    e = model.transform[4]
    f_val = model.transform[5]
    x_contour = c + (longest[:, 1] + 0.5) * a
    y_contour = f_val + (longest[:, 0] + 0.5) * e

    ignition_x, ignition_y = model.ignition_pt
    contour_angles = np.degrees(np.arctan2(y_contour - ignition_y, x_contour - ignition_x))
    contour_angles = (contour_angles + 360) % 360

    valid_ros = [v for v in aggregated_ros.values() if not np.isnan(v)]
    if not valid_ros:
        print("No valid ROS values.")
        return
    vmin, vmax = min(valid_ros), max(valid_ros)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('inferno')

    # ---------- MAIN PERIMETER PLOT ----------
    fig, ax = plt.subplots(figsize=(6, 6))  # Match scaling
    ax.set_xlabel("X Coordinate (m)", fontsize=12)
    ax.set_ylabel("Y Coordinate (m)", fontsize=12)
    ax.set_title("Fire Perimeter Colored by Aggregated ROS (5° Sectors)", fontsize=14, pad=10)
    ax.tick_params(labelsize=10)

    for (lower, upper) in angle_ranges:
        mask = (contour_angles >= lower) & (contour_angles < upper)
        if np.any(mask):
            value = aggregated_ros[(lower, upper)]
            color = cmap(norm(value)) if np.isfinite(value) else (0.5, 0.5, 0.5, 0.5)
            ax.plot(x_contour[mask], y_contour[mask], '-', lw=3, color=color)

    ax.plot(x_contour, y_contour, 'k--', lw=1, alpha=0.5)

    bounds = model.bounds
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    # ax.set_xlabel("X Coordinate")
    # ax.set_ylabel("Y Coordinate")
    # ax.set_title("Fire Perimeter Colored by Aggregated ROS (5° Sectors)", pad=20)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Aggregated ROS (ft/min)", fraction=0.046, pad=0.04)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved main perimeter plot to {save_path}")

    # ---------- SEPARATE LEGEND FIGURE ----------
    fig_legend, ax_legend = plt.subplots(figsize=(12, 6))
    ax_legend.axis('off')

    legend_handles = []
    for lower, upper in angle_ranges:
        value = aggregated_ros.get((lower, upper), np.nan)
        color = cmap(norm(value)) if np.isfinite(value) else (0.5, 0.5, 0.5, 0.5)
        label = f"{lower}-{upper}°: {value:.1f} ft/min" if np.isfinite(value) else f"{lower}-{upper}°: N/A"
        handle = Line2D([0], [0], color=color, lw=3, label=label)
        legend_handles.append(handle)

    fig_legend.legend(
        handles=legend_handles,
        loc='center',
        ncol=6,
        fontsize=7,
        title="Sector Aggregated ROS",
        frameon=False
    )

    fig_legend.tight_layout()
    fig_legend.savefig(legend_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig_legend)
    print(f"Saved sector legend to {legend_save_path}")
########################################
# 5) Example usage
########################################
def main():
    tif_path = "cali_test_big_enhanced.tif"
    sim_time = 20000

    phase_duration = 120  # Duration of each phase in minutes
    num_phases = sim_time // phase_duration  # Total number of phases
    fixed_wind_speed = 25

    # Create wind schedule: first half with wind_direction = 180, second half with wind_direction = 0.
    first_half = num_phases // 2
    wind_schedule = []
    for i in range(num_phases):
        t_start = i * phase_duration
        t_end = (i + 1) * phase_duration
        wd = 0 if i < first_half else 0
        wind_schedule.append((t_start, t_end, fixed_wind_speed, wd))

    print("Wind schedule:")
    for seg in wind_schedule:
        print(f"  t = {seg[0]} to {seg[1]}: wind_speed = {seg[2]} mph, wind_direction = {seg[3]}°")

    t0 = time.time()
    model = SurrogateFireModelROS(
        tif_path, sim_time,
        wind_speed=30, wind_direction_deg=270,
        max_iter=250, tol=1e-4,
        wind_schedule=wind_schedule
    )
    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
    print(f"CK2 multi-phase run took {t1 - t0:.3f} s")

    # Get final arrival-time map on CPU:
    T_np = cp.asnumpy(model.T)
    print("T final shape:", T_np.shape)

    # ------------------------
    # Plotting Section
    # ------------------------
    # 1) Arrival Time Map
    with rasterio.open(tif_path) as src:
        extent = (src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top)

    ##################################################################################
    # ROS STUFF
    ##################################################################################
    cell_size_ft = 98.4  # given cell dimension
    time_threshold = 2000  # example threshold (minutes) for the perimeter
    tolerance_past = 30
    tolerance_future = 200
    # Compute the complete ROS field from the arrival times
    ros_field = compute_ros_field(T_np, cell_size_ft)
    # Compute the perimeter ROS (cells within ±1 minute of the threshold)
    perimeter_ros = get_perimeter_ros(T_np, time_threshold, tolerance_past=tolerance_past,
                                      tolerance_future=tolerance_future, cell_size_ft=cell_size_ft)
    burned = T_np[np.isfinite(T_np)]
    print("Arrival time stats for burned regions:")
    print("Min:", burned.min(), "Max:", burned.max())
    print("Mean:", burned.mean())
    print("Median:", np.percentile(burned, 50))

    # Debug: Print some statistics of the computed ROS field
    finite_ros = ros_field[np.isfinite(ros_field)]
    print("ROS Field stats (ft/min):")
    print("  Min =", finite_ros.min())
    print("  Max =", finite_ros.max())
    print("  Mean =", finite_ros.mean())
    # Plotting the full ROS field
    plt.figure(figsize=(6, 6))
    vmax_ros = np.nanpercentile(ros_field, 99)  # clip out top 1% outliers
    plt.imshow(ros_field, cmap='inferno', extent=extent, origin='upper', vmin=0, vmax=vmax_ros)

    # plt.imshow(ros_field, cmap='inferno', extent=extent, origin='upper')
    plt.colorbar(label='ROS (ft/min)')
    plt.title("Local Rate-of-Spread Field")
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    # plt.show()
    plt.savefig("local_ros_field.png", dpi=300, bbox_inches='tight')

    # Plotting only the fire perimeter ROS
    plt.figure(figsize=(6, 6))
    vmax_perim = np.nanpercentile(perimeter_ros, 99)  # clip out top 1% for perimeter
    plt.imshow(perimeter_ros, cmap='inferno', extent=extent, origin='upper', vmin=0, vmax=vmax_perim)
    # plt.imshow(perimeter_ros, cmap='inferno', extent=extent, origin='upper')
    plt.colorbar(label='Perimeter ROS (ft/min)')
    plt.title(f"Fire Perimeter ROS at t = {time_threshold} min\n(Future Buffer: +{tolerance_future} min)")

    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    # plt.show()
    plt.savefig("perimeter_ros_field.png", dpi=300, bbox_inches='tight')

    # angle_ranges = [(0, 5), (5, 10), (10, 15), (15, 20)]
    angle_ranges = [(i, i + 5) for i in range(0, 360, 5)]

    # Aggregate the ROS on the perimeter using a specific time threshold.
    # (Adjust time_threshold and tolerance based on your simulation.)

    tolerance = 10  # ±10 minutes tolerance.
    tolerance_past = 5
    tolerance_future = 10

    aggregated_ros = model.aggregate_perimeter_ros_by_increments(
        angle_ranges=angle_ranges,
        statistic='mean',
        time_threshold=time_threshold,
        tolerance_past=tolerance_past,
        tolerance_future=tolerance_future,
        cell_size_ft=98.4
    )
    print("Aggregated ROS by Angle Ranges:", aggregated_ros)
    tolerance = 10  # ±10 minutes tolerance.
    cell_size_ft = 98.4  # Given cell dimension.
    # Plot the aggregated ROS overlaid on the fire perimeter.
    # (Make sure the function plot_aggregated_ros_on_perimeter is imported or defined.)
    # Now call the function to plot the colored perimeter by sector.
    plot_colored_perimeter_by_sector(model, time_threshold, tolerance_past=tolerance_past,
                                     tolerance_future=tolerance_future, cell_size_ft=cell_size_ft)

    ##################################################################################
    ##################################################################################
    ##################################################################################

    plt.figure(figsize=(6, 6))
    plt.imshow(T_np, cmap='viridis', extent=extent, origin='upper')
    plt.colorbar(label='Arrival Time (min)')
    plt.title("CK2 Multi-phase Fire Arrival Times")
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    # plt.show()
    plt.savefig("arrivaltime_field.png", dpi=300, bbox_inches='tight')

    # 2) Plot burned area at a certain threshold
    time_threshold = 2000  # example threshold
    final_state = np.zeros(model.grid_size, dtype=np.int32)
    final_state[T_np < time_threshold] = 2  # mark burned cells as 2

    plt.figure(figsize=(6, 6))
    plt.imshow(final_state, cmap='hot', extent=extent, origin='upper')
    plt.title(f"Burned Area at t = {time_threshold} minutes")
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.colorbar(label="State (0: unburned, 2: burned)")
    plt.show()

    # 3) Thresholded arrival map
    thresholded_arrival = model.current_fire(0, max_time=time_threshold)

    plt.figure(figsize=(6, 6))
    plt.imshow(thresholded_arrival, cmap='viridis', extent=extent, origin='upper')
    plt.colorbar(label='Arrival Time (min)')
    plt.title(f"Fire Arrival Times (Thresholded at t = {time_threshold})")
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.show()

    # 4) Fire score example
    time_threshold_score = 100
    fire_score = model.calculate_fire_score(time_threshold_score)
    print(f"Number of burning cells (arrival time <= {time_threshold_score}): {fire_score}")


#
if __name__ == "__main__":
    import cProfile
    import pstats

    cProfile.run("main()", "profile_fm.out")

#     tif_path = "testslope.tif"
#     wind_speed = 10
#     wind_dir_deg = 180
#     max_iter = 200
#     tol = 1e-4
#     sim_time = 100
#
#     tif_path = "testslope.tif"  # Replace with your actual TIFF file path
#     sim_time = 2000  # Total simulation time in minutes
#     phase_duration = 120  # Duration of each phase in minutes
#     num_phases = sim_time // phase_duration  # Total number of phases
#
#     # Set fixed wind speed to 10 mph for all phases.
#     fixed_wind_speed = 10
#
#     # Create wind schedule: first half of phases with wind_direction = 270, second half with wind_direction = 90.
#     first_half = num_phases // 2
#     wind_schedule = []
#     for i in range(num_phases):
#         t_start = i * phase_duration
#         t_end = (i + 1) * phase_duration
#         if i < first_half:
#             wd = 180
#         else:
#             wd = 0
#         wind_schedule.append((t_start, t_end, fixed_wind_speed, wd))
#
#     print("Wind schedule:")
#     for seg in wind_schedule:
#         print(f"  t = {seg[0]} to {seg[1]}: wind_speed = {seg[2]} mph, wind_direction = {seg[3]}°")
#
#     t_start = time.time()
#     surrogate = SurrogateFireModelROS(
#         tif_path=tif_path,
#         sim_time=sim_time,
#         wind_speed=wind_speed,
#         wind_direction_deg=wind_dir_deg,
#         max_iter=max_iter,
#         tol=tol,
#         wind_schedule=None,
#     )
#     cp.cuda.Stream.null.synchronize()
#     elapsed = time.time() - t_start
#     print(f"Simulation took {elapsed:.3f} seconds.")
#
#
#     # Plot
#     T_np = cp.asnumpy(surrogate.T)
#     with rasterio.open(tif_path) as src:
#         extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
#
#     plt.figure()
#     plt.imshow(T_np, cmap='viridis', extent=extent, origin='upper')
#     plt.colorbar(label='Arrival Time (min)')
#     plt.title("Deterministic Fire Arrival Times (Custom Kernel)")
#     plt.show()
#
# # Plot burned area based on a time threshold
#     time_threshold = 2000
#     final_state = np.zeros(surrogate.grid_size, dtype=np.int32)
#     final_state[T_np < time_threshold] = 2
#     plt.figure(figsize=(6, 6))
#     plt.imshow(final_state, cmap='hot', extent=extent, origin='upper')
#     plt.title(f"Deterministic Burned Area at t = {time_threshold}")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.colorbar(label="State (0: unburned, 2: burned)")
#     plt.show()
#
#     # Mask arrival times above the threshold.
#     thresholded_arrival = surrogate.current_fire(0, max_time=2000)
#     plt.figure(figsize=(6, 6))
#     plt.imshow(thresholded_arrival, cmap='viridis', extent=extent, origin='upper')
#     plt.colorbar(label='Arrival Time (min)')
#     plt.title(f"Fire Arrival Times (Thresholded at t = {time_threshold})")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.show()
#
#     # Example: compute fire score for a given time threshold.
#     time_threshold_score = 100
#     fire_score = surrogate.calculate_fire_score(time_threshold_score)
#     print(f"Number of burning cells (arrival time <= {time_threshold_score}): {fire_score}")
#
#     # ------------------------
#     # Plotting Section
#     # ------------------------
#     # # 1) Arrival Time Map
#     # with rasterio.open(tif_path) as src:
#     #     extent = (src.bounds.left, src.bounds.right,
#     #               src.bounds.bottom, src.bounds.top)
#
#     ##################################################################################
#     # ROS STUFF
#     ##################################################################################
#     cell_size_ft = 98.4  # given cell dimension
#     time_threshold = 2000  # example threshold (minutes) for the perimeter
#     tolerance_past = 30
#     tolerance_future = 200
#     # Compute the complete ROS field from the arrival times
#     ros_field = compute_ros_field(T_np, cell_size_ft)
#     # Compute the perimeter ROS (cells within ±1 minute of the threshold)
#     perimeter_ros = get_perimeter_ros(T_np, time_threshold, tolerance_past=tolerance_past,
#                                       tolerance_future=tolerance_future, cell_size_ft=cell_size_ft)
#     burned = T_np[np.isfinite(T_np)]
#     print("Arrival time stats for burned regions:")
#     print("Min:", burned.min(), "Max:", burned.max())
#     print("Mean:", burned.mean())
#     print("Median:", np.percentile(burned, 50))
#
#     # Debug: Print some statistics of the computed ROS field
#     finite_ros = ros_field[np.isfinite(ros_field)]
#     print("ROS Field stats (ft/min):")
#     print("  Min =", finite_ros.min())
#     print("  Max =", finite_ros.max())
#     print("  Mean =", finite_ros.mean())
#     # Plotting the full ROS field
#     plt.figure(figsize=(6, 6))
#     vmax_ros = np.nanpercentile(ros_field, 99)  # clip out top 1% outliers
#     plt.imshow(ros_field, cmap='inferno', extent=extent, origin='upper', vmin=0, vmax=vmax_ros)
#
#     # plt.imshow(ros_field, cmap='inferno', extent=extent, origin='upper')
#     plt.colorbar(label='ROS (ft/min)')
#     plt.title("Local Rate-of-Spread Field")
#     plt.xlabel("X Coordinate (m)")
#     plt.ylabel("Y Coordinate (m)")
#     # plt.show()
#     plt.savefig("local_ros_field.png", dpi=300, bbox_inches='tight')
#
#     # Plotting only the fire perimeter ROS
#     plt.figure(figsize=(6, 6))
#     vmax_perim = np.nanpercentile(perimeter_ros, 99)  # clip out top 1% for perimeter
#     plt.imshow(perimeter_ros, cmap='inferno', extent=extent, origin='upper', vmin=0, vmax=vmax_perim)
#     # plt.imshow(perimeter_ros, cmap='inferno', extent=extent, origin='upper')
#     plt.colorbar(label='Perimeter ROS (ft/min)')
#     plt.title(f"Fire Perimeter ROS at t = {time_threshold} min\n(Future Buffer: +{tolerance_future} min)")
#
#     plt.xlabel("X Coordinate (m)")
#     plt.ylabel("Y Coordinate (m)")
#     # plt.show()
#     plt.savefig("perimeter_ros_field.png", dpi=300, bbox_inches='tight')
#
#     # angle_ranges = [(0, 5), (5, 10), (10, 15), (15, 20)]
#     angle_ranges = [(i, i + 5) for i in range(0, 360, 5)]
#
#     # Aggregate the ROS on the perimeter using a specific time threshold.
#     # (Adjust time_threshold and tolerance based on your simulation.)
#
#     tolerance = 10  # ±10 minutes tolerance.
#     tolerance_past = 5
#     tolerance_future = 10
#
#     aggregated_ros = surrogate.aggregate_perimeter_ros_by_increments(
#         angle_ranges=angle_ranges,
#         statistic='mean',
#         time_threshold=time_threshold,
#         tolerance_past=tolerance_past,
#         tolerance_future=tolerance_future,
#         cell_size_ft=98.4
#     )
#     print("Aggregated ROS by Angle Ranges:", aggregated_ros)
#     tolerance = 10  # ±10 minutes tolerance.
#     cell_size_ft = 98.4  # Given cell dimension.
#     # Plot the aggregated ROS overlaid on the fire perimeter.
#     # (Make sure the function plot_aggregated_ros_on_perimeter is imported or defined.)
#     # Now call the function to plot the colored perimeter by sector.
#     plot_colored_perimeter_by_sector(surrogate, time_threshold, tolerance_past=tolerance_past,
#                                      tolerance_future=tolerance_future, cell_size_ft=cell_size_ft)
#
#
# import cProfile
# import pstats
#
# if __name__ == "__main__":
#     # Run and record the profile to profile.out
#     cProfile.run("main()", "profileCK_new.out")
