#!/usr/bin/env python3
"""

Build a 120-minute-bin forecast/uncertainty table from RAWS data.

Output columns :
    start_min, end_min, speed_mean, speed_std, dir_mean, dir_std
"""

from __future__ import annotations
from pathlib import Path
import math, sys
import numpy as np
import pandas as pd

# ───────────── USER-TUNABLE SETTINGS ────────────────────────────────
RAWS_PATH        = "data_esper_high.xls"   # ← put your uploaded Crazy-Peak file here
END_MIN          = 3000        # simulation length (minutes)
DT               = 120          # bin size (minutes) – keep at 120
TAPER_HR         = 8            # σ stops growing after this many hours

# Initial & max σ (speed in mph, direction in deg)
# for high esper&marshall & medium camp
SPEED_STD_MIN    = 1
SPEED_STD_MAX    = 5.0

#for medium esper&marshall
# SPEED_STD_MIN    = 1
# SPEED_STD_MAX    = 2.5

#for high camp
# SPEED_STD_MIN    = 1
# SPEED_STD_MAX    = 15.0

# #medium
# DIR_STD_MIN      = 10.0
# DIR_STD_MAX      = 30.0

#high
DIR_STD_MIN      = 20.0
DIR_STD_MAX      = 60.0
# ────────────────────────────────────────────────────────────────────


# RAWS_PATH        = "data_two.xls"   # ← put your uploaded Crazy-Peak file here
# END_MIN          = 2000         # simulation length (minutes)
# DT               = 120          # bin size (minutes) – keep at 120
# TAPER_HR         = 12            # σ stops growing after this many hours
#
# # Initial & max σ (speed in mph, direction in deg)
# SPEED_STD_MIN    = 0.1
# SPEED_STD_MAX    = 3.0
# DIR_STD_MIN      = 10.0
# DIR_STD_MAX      = 60.0
# # ────────────────────────────────────────────────────────────────────


def _parse_raws_txt(path: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns timestamp, wind_speed_mph, wind_dir_deg.
    """
    rows: list[tuple[pd.Timestamp, float, float]] = []
    with path.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith(":"):
                continue                       # skip headers / blanks
            parts = ln.split()
            if len(parts) < 4:
                continue                       # incomplete row, skip
            ts_raw = parts[0]                  # e.g. 2507010000
            try:
                ts = pd.to_datetime(ts_raw, format="%y%m%d%H%M")
            except ValueError:
                print(f"[WARN] bad timestamp {ts_raw!r}, skipping")
                continue
            speed = float(parts[2])
            direc = float(parts[3]) % 360
            rows.append((ts, speed, direc))

    if not rows:
        sys.exit("[ERROR] No RAWS data rows found – check the file path.")
    df = pd.DataFrame(rows, columns=["timestamp",
                                     "wind_speed_mph",
                                     "wind_dir_deg"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _circular_mean(deg_series: pd.Series) -> float:
    """Mean direction on a circle (deg)."""
    rad = np.deg2rad(deg_series.to_numpy())
    return math.degrees(math.atan2(np.mean(np.sin(rad)),
                                   np.mean(np.cos(rad)))) % 360


def build_schedule(raws: pd.DataFrame) -> pd.DataFrame:
    """
    Construct DT-minute bins starting at the first RAWS record (t=0)
    and ending at END_MIN.  Means come from the RAWS data; σ follows
    the linear-until-TAPER ramp.
    """
    t0 = raws["timestamp"].iloc[0]
    raws["lead_min"] = (raws["timestamp"] - t0).dt.total_seconds() / 60.0

    n_bins = END_MIN // DT
    rows = []

    for k in range(n_bins + 1):          # +1 so last (possibly short) bin 1920-2000 is included
        start_m = k * DT
        end_m   = min(start_m + DT, END_MIN)

        mask = (raws["lead_min"] >= start_m) & (raws["lead_min"] < end_m)
        slice_df = raws[mask]

        if slice_df.empty:
            # If RAWS ends early, use the latest available obs
            slice_df = raws.iloc[[-1]]

        speed_mean = slice_df["wind_speed_mph"].mean()
        dir_mean   = _circular_mean(slice_df["wind_dir_deg"])

        lead_hr = start_m / 60.0
        frac    = min(lead_hr, TAPER_HR) / TAPER_HR
        speed_std = SPEED_STD_MIN + (SPEED_STD_MAX - SPEED_STD_MIN) * frac
        dir_std   = DIR_STD_MIN   + (DIR_STD_MAX   - DIR_STD_MIN) * frac

        rows.append({
            "start_min": start_m,
            "end_min"  : end_m,
            "speed_mean": round(speed_mean, 2),
            "speed_std" : round(speed_std , 2),
            "dir_mean"  : round(dir_mean  , 2),
            "dir_std"   : round(dir_std   , 2),
        })

    return pd.DataFrame(rows)


def main() -> None:
    raws_path = Path(RAWS_PATH)
    if not raws_path.exists():
        sys.exit(f"[ERROR] {raws_path} not found.")

    raws_df = _parse_raws_txt(raws_path)
    sched   = build_schedule(raws_df)

    out_path = Path("wind_schedule_esper_high.csv")
    sched.to_csv(out_path, index=False)
    print(f"[OK] wrote schedule → {out_path.resolve()}")


if __name__ == "__main__":
    main()
