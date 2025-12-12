# wind_schedule_utils.py
"""
Convert wind_schedule.csv into the list of
(t_start, t_end, speed, direction) tuples expected by the surrogate model.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple   # ← add this line
# def load_wind_schedule_from_csv(path="wind_schedule.csv", seed=None):
#     """
#     Read wind_schedule.csv and return a list of
#     (t_start, t_end, wind_speed_int, wind_dir_int) tuples.
#
#     • wind_speed is drawn from N(mean, std) → rounded → clamped ≥ 0 → int
#     • wind_direction is drawn from N(mean, std) → rounded → wrapped 0-359 → int
#     """
#     rng = np.random.default_rng(seed)
#     df  = pd.read_csv(path)
#
#     schedule = []
#     for row in df.itertuples():
#         # --- sample -----------------------------------------------------
#         wspd_f = rng.normal(row.speed_mean, row.speed_std)
#         wdir_f = rng.normal(row.dir_mean,   row.dir_std)
#
#         # --- convert to clean ints -------------------------------------
#         wspd_i = int(max(round(wspd_f), 0))       # no negative wind speeds
#         wdir_i = int(round(wdir_f)) % 360         # keep in 0-359
#
#         schedule.append((
#             int(row.start_min),
#             int(row.end_min),
#             wspd_i,
#             wdir_i
#         ))
#
#     return schedule
def load_wind_schedule_from_csv_mean(path="wind_schedule.csv"):
    df = pd.read_csv(path)
    return [(int(r.start_min), int(r.end_min),
             float(r.speed_mean), float(r.dir_mean) % 360)
            for r in df.itertuples()]

def sample_schedule_from_forecast(forecast_df: pd.DataFrame, seed=None):
    """
    Draw one concrete (speed, dir) from each row’s µ/σ.
    Returns list of (start_min, end_min, speed_int, dir_int).
    """
    rng = np.random.default_rng(seed)
    out = []
    for r in forecast_df.itertuples():
        spd = int(round(rng.normal(r.speed_mean, r.speed_std)))
        d   = int(round(rng.normal(r.dir_mean,   r.dir_std))) % 360
        out.append((int(r.start_min), int(r.end_min), max(spd, 0), d))
    return out


# ─── ADD THIS NEW HELPER ────────────────────────────────────────────
def sample_future_schedule(forecast_df: pd.DataFrame,
                           current_minute: int,
                           *, rng: np.random.Generator) -> list[tuple]:
    """
    Return a list of (t_start, t_end, speed, dir) tuples **starting at
    current_minute or later**.

    Every row in forecast_df must have the cols:
        start_min, end_min, speed_mean, speed_std, dir_mean, dir_std
    """
    future = forecast_df[forecast_df["end_min"] > current_minute]

    schedule = []
    for row in future.itertuples(index=False):
        # draw exactly one sample per forecast bin
        wspd = rng.normal(row.speed_mean, row.speed_std)
        wdir = rng.normal(row.dir_mean,   row.dir_std) % 360
        schedule.append((
            int(row.start_min),
            int(row.end_min),
            float(max(wspd, 0.0)),
            float(wdir)
        ))
    return schedule



def _angular_diff(a: float, b: float) -> float:
    """Shortest signed difference between two headings (deg)."""
    d = (a - b + 180) % 360 - 180
    return abs(d)

# wind_schedule_utils.py  (replace the current helper)
def load_wind_schedule_from_csv_random(path="wind_schedule.csv", *, seed=None):
    """
    Return list[(start, end, speed, dir)].
    If *seed* is None → means only.  Otherwise draw one sample per row.
    """
    rng = None if seed is None else np.random.default_rng(seed)
    df  = pd.read_csv(path)

    if rng is None:                     # deterministic “mean” schedule
        return [(int(r.start_min), int(r.end_min),
                 float(r.speed_mean), float(r.dir_mean) % 360)
                for r in df.itertuples()]

    out = []
    for r in df.itertuples():
        spd = max(rng.normal(r.speed_mean, r.speed_std), 0.0)
        d   = rng.normal(r.dir_mean,  r.dir_std) % 360.0
        out.append((int(r.start_min), int(r.end_min), spd, d))
    return out



def load_wind_schedule_from_csv_sigma(
    path: str = "wind_schedule.csv",
    *,
    sigma: float = 3.0,
    seed: int | None = None
) -> List[Tuple[int, int, float, float]]:
    """
    Return the **truth** schedule with every (speed, dir) pushed exactly
    ± sigma·std away from the mean.

    Parameters
    ----------
    path   : CSV with the usual columns: start_min, end_min,
             speed_mean, speed_std, dir_mean, dir_std
    sigma  : How many standard deviations away from the mean
    seed   : If given, results are reproducible (changes the ± sign pattern)

    Notes
    -----
    • Speed is clamped at ≥ 0 after the offset.
    • Direction is wrapped into 0–359 deg after the offset.
    """
    rng = np.random.default_rng(seed)
    df  = pd.read_csv(path)

    out = []
    for r in df.itertuples(index=False):
        # decide sign independently for speed and direction
        sign_spd = rng.choice((-1, 1))
        sign_dir = rng.choice((-1, 1))

        spd = r.speed_mean + sign_spd * sigma * r.speed_std
        spd = max(spd, 0.0)                          # no negative wind speed

        d   = (r.dir_mean + sign_dir * sigma * r.dir_std) % 360.0

        out.append((int(r.start_min), int(r.end_min), float(spd), float(d)))
    return out


def _print_schedule(
    schedule: List[Tuple[int, int, float, float]],
    title: str = "Truth schedule"
) -> None:
    """
    Nicely print a list of (t_start, t_end, speed, dir) tuples.
    """
    df = pd.DataFrame(schedule, columns=["start", "end", "speed", "dir"])
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))