# ───────────────── forecast_provider.py ───────────────────────────
"""
Wind‑forecast generator that stays consistent with the *actual* truth
schedule being simulated.

Provides updated forecasts to the MCTS

"""

from __future__ import annotations
import math, numpy as np, pandas as pd
from typing import Sequence
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# CONSTANTS  – sync the MIN/MAX values with raws_to_sched.py
DT               = 120           # minutes per segment
TAPER_HR         = 8             # σ grows linearly for the first 6 h
SPEED_STD_MIN    = 1.0           # mph
SPEED_STD_MAX    = 5.0
DIR_STD_MIN      = 5.0           # deg
DIR_STD_MAX      = 20.0

ASSIM_HR         = 8             # inside this horizon we start nudging μ



# ── globals ------------------------------------------------------
_truth_df:      pd.DataFrame | None = None
_background_df: pd.DataFrame | None = None
# ----------------------------------------------------------------


def set_forecast_hyperparams(*,
    speed_std_min=None,
    speed_std_max=None,
    dir_std_min=None,
    dir_std_max=None,
):
    """Optionally override the σ min/max used by get_forecast()."""
    global SPEED_STD_MIN, SPEED_STD_MAX, DIR_STD_MIN, DIR_STD_MAX
    if speed_std_min is not None: SPEED_STD_MIN = float(speed_std_min)
    if speed_std_max is not None: SPEED_STD_MAX = float(speed_std_max)
    if dir_std_min is not None:   DIR_STD_MIN   = float(dir_std_min)
    if dir_std_max is not None:   DIR_STD_MAX   = float(dir_std_max)

def set_truth_schedule(schedule: Sequence) -> None:
    """
    Register the *truth* wind schedule so every forecast call uses
    the same reference.

    *schedule* can be either
        • a pandas DataFrame with cols
            start_min, end_min, speed_mean, dir_mean
        • or a list[tuple] of (start, end, speed, dir)
    """
    global _truth_df
    if isinstance(schedule, pd.DataFrame):
        _truth_df = schedule[["start_min", "end_min",
                              "speed_mean", "dir_mean"]].copy()
    else:  # assume list/tuple
        _truth_df = pd.DataFrame(schedule,
                                 columns=["start_min", "end_min",
                                          "speed_mean", "dir_mean"])
    _truth_df.sort_values("start_min", inplace=True)
    _truth_df.reset_index(drop=True, inplace=True)

def set_background_schedule(schedule: Sequence) -> None:
    """
    Register the *climatology* schedule (usually the means from
    wind_schedule_from_raws.csv).  Columns: start_min, end_min,
    speed_mean, dir_mean.
    """
    global _background_df
    if isinstance(schedule, pd.DataFrame):
        _background_df = schedule[["start_min", "end_min",
                                   "speed_mean", "dir_mean"]].copy()
    else:
        _background_df = pd.DataFrame(schedule,
                                      columns=["start_min", "end_min",
                                               "speed_mean", "dir_mean"])
    _background_df.sort_values("start_min", inplace=True)
    _background_df.reset_index(drop=True, inplace=True)

def _circular_mean(series: pd.Series) -> float:
    rad = np.deg2rad(series.to_numpy())
    return math.degrees(math.atan2(np.mean(np.sin(rad)),
                                   np.mean(np.cos(rad)))) % 360


def _angular_diff(a: float, b: float) -> float:
    """Shortest unsigned distance on a circle (deg)."""
    return abs((a - b + 180) % 360 - 180)


def _segment_means(start_m: int, end_m: int) -> tuple[float, float]:
    """
    Return climatological (μ_speed, μ_dir) for that slice.
    Falls back to the last row if the window is past the data.
    """
    if _background_df is None:
        raise RuntimeError("forecast_provider: call "
                           "set_background_schedule() before get_forecast().")

    mask = (_background_df.start_min < end_m) & (_background_df.end_min > start_m)
    slice_df = _background_df[mask]
    if slice_df.empty:
        slice_df = _background_df.iloc[[-1]]

    s_mu = slice_df.speed_mean.mean()
    d_mu = _circular_mean(slice_df.dir_mean)
    return float(s_mu), float(d_mu)



def get_forecast(*,
                 current_minute: int,
                 horizon_min: int | None = None) -> pd.DataFrame:
    """
    Build a μ/σ forecast table whose first segment **starts at or after
    *current_minute*** and extends to either *horizon_min* or the end of
    the truth schedule.

    σ follows a linear ramp:
        SPEED  : 1 → 5 mph   (0 → 6 h ahead)
        DIR    : 5 → 20 deg

    μ is a weighted blend between the template μ and the *truth* value
    for that segment; the weight goes from 0 at > ASSIM_HR lead time to
    1 at lead = 0.
    """

    print(
        "[forecast_provider] get_forecast() init",

        f"  DT={DT} min, TAPER_HR={TAPER_HR}, ASSIM_HR={ASSIM_HR}\n"
        f"  SPEED_STD_MIN={SPEED_STD_MIN}, SPEED_STD_MAX={SPEED_STD_MAX}\n"
        f"  DIR_STD_MIN={DIR_STD_MIN},   DIR_STD_MAX={DIR_STD_MAX}\n"

    )
    if _truth_df is None:
        raise RuntimeError("forecast_provider: call set_truth_schedule() "
                           "before get_forecast().")

    now = int(current_minute)

    # dynamic horizon: run until the end of the schedule if not specified
    if horizon_min is None:
        horizon_min = max(0, int(_truth_df.end_min.max()) - now)

    end = now + horizon_min
    n_bins = math.ceil((end - now) / DT)

    rows = []
    for k in range(n_bins):
        start_m = now + k * DT
        end_m = min(start_m + DT, end)

        # --- template μ (background) -------------------------------------
        mu_speed, mu_dir = _segment_means(start_m, end_m)

        # --- truth μ for the same slice ----------------------------------
        mask = (_truth_df.start_min < end_m) & (_truth_df.end_min > start_m)
        if mask.any():
            tru_slice = _truth_df[mask]
        else:
            tru_slice = _truth_df.iloc[[-1]]
        sp_truth = tru_slice.speed_mean.mean()
        dir_truth = _circular_mean(tru_slice.dir_mean)

        # ---------- σ ramp ------------------------------------------------
        lead_hr = (start_m - now) / 60.0         # hours ahead
        frac    = min(max(lead_hr, 0.0), TAPER_HR) / TAPER_HR    # 0–1
        sp_std  = SPEED_STD_MIN + (SPEED_STD_MAX - SPEED_STD_MIN) * frac
        dir_std = DIR_STD_MIN   + (DIR_STD_MAX   - DIR_STD_MIN)   * frac

        # ---------- μ assimilation ----------------------------------------
        w = 0.0 if lead_hr >= ASSIM_HR else 1.0 - lead_hr / ASSIM_HR

        mu_speed = (1 - w) * mu_speed + w * sp_truth

        # circular interpolation for direction
        a, b = np.deg2rad([mu_dir, dir_truth])
        x = (1 - w) * math.cos(a) + w * math.cos(b)
        y = (1 - w) * math.sin(a) + w * math.sin(b)
        mu_dir = (math.degrees(math.atan2(y, x)) % 360)

        rows.append(dict(start_min=start_m,
                         end_min=end_m,
                         speed_mean=round(mu_speed, 2),
                         speed_std=round(sp_std,   2),
                         dir_mean  =round(mu_dir,  2),
                         dir_std   =round(dir_std, 2)))

    return pd.DataFrame(rows)
# ──────────────────────────────────────────────────────────────────
