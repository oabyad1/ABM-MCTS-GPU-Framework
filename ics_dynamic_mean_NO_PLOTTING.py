#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ics_dynamic_mean_NO_PLOTTING.py
──────────────────────────────────────────────────────────────────────────────
Dynamic ICS baseline that **makes every decision from the forecast *mean*
(μ) schedule** while the wildfire itself still evolves under the true sampled
schedule provided at construction.

•  Sector priorities / budgets are identical to the original deterministic
   ICS script.
•  Only two functional changes:
     1. _latest_wind_direction() reads dir_mean from `model.latest_forecast_df`
        instead of peeking at the truth schedule.
     2. The main simulation loop refreshes that forecast DataFrame at every
        decision slice.

Launch with:  `python ics_dynamic_mean_NO_PLOTTING.py`
"""

# ───────────────────────────────────────────────────────────────
# Imports & globals
# ───────────────────────────────────────────────────────────────
import numpy as np
import datetime
import time as t

import dashboard as dash
from mcts import simulate_in_place, ordinal_map

DECISION_INTERVAL = 120        # minutes between allocation rounds

# ───────────────────────────────────────────────────────────────
# Helper functions (mostly unchanged from the baseline script)
# ───────────────────────────────────────────────────────────────
def _sectors_for_angle(angle_deg, sector_ranges, *, eps=1e-6):
    angle_deg %= 360
    hits = [idx for idx, (lo, hi) in enumerate(sector_ranges)
            if lo - eps <= angle_deg < hi + eps]
    return hits or [0]

def _weights_per_sector(open_set, head_secs, heel_secs, flank_secs):
    BUDGET = dict(head=0.98, flank=0.01, heel=0.01)
    roles  = dict(
        head=[s for s in head_secs  if s in open_set],
        heel=[s for s in heel_secs  if s in open_set],
        flank=[s for s in flank_secs if s in open_set],
    )
    w = {}
    for role, secs in roles.items():
        if secs:
            share = BUDGET[role] / len(secs)
            for s in secs:
                w[s] = share
    tot = sum(w.values()) or 1.0
    for s in w:                                       # renormalise
        w[s] /= tot
    return w

def _draw_for_assets(rng, open_set, weights, n):
    sectors = np.array(list(open_set))
    probs   = np.array([weights[s] for s in sectors])
    probs  /= probs.sum()
    if len(sectors) >= n:
        picks = rng.choice(sectors, size=n, replace=False, p=probs)
    else:                              # cycle if not enough unique sectors
        picks = list(rng.choice(sectors, size=len(sectors),
                                replace=False, p=probs))
        while len(picks) < n:
            picks.extend(picks)
        picks = np.array(picks[:n])
    return [int(s) + 1 for s in picks]

# ───────────────────────────────────────────────────────────────
# Forecast-aware wind helper  **(THIS IS THE KEY CHANGE!)**
# ───────────────────────────────────────────────────────────────
def _latest_wind_direction(model) -> float:
    """
    Decision-wind direction priority:
      1. dir_mean in model.latest_forecast_df covering ‘now’
      2. last row of that DataFrame (if we’re past its end)
      3. direction of the *truth* schedule segment currently active
      4. static model.wind_direction
    """
    df = getattr(model, "latest_forecast_df", None)
    if df is not None and not df.empty:
        now = model.time
        row = df[(df["start_min"] <= now) & (now < df["end_min"])]
        if not row.empty:
            return float(row.iloc[0]["dir_mean"])
        return float(df.iloc[-1]["dir_mean"])

    # --- fallbacks ----------------------------------------------------
    if model.wind_schedule is None:
        return float(model.wind_direction)
    for s, e, _spd, wdir in model.wind_schedule:
        if s <= model.time < e:
            return float(wdir)
    return float(model.wind_schedule[-1][3])

# ───────────────────────────────────────────────────────────────
# Angle→sector helpers & allocator (unchanged logic, new wind)
# ───────────────────────────────────────────────────────────────
def _angle_to_sector(angle_deg, sector_ranges):
    angle_deg %= 360
    for idx, (lo, hi) in enumerate(sector_ranges):
        if lo <= angle_deg < hi:
            return idx
    return 0

def dynamic_ics_allocation(model, open_secs):
    rng        = np.random.default_rng()
    open_set   = set(open_secs)
    model.update_sector_splits()
    ranges     = model.sector_angle_ranges

    raw_wind   = _latest_wind_direction(model)
    head_ang   = (180 + raw_wind) % 360
    heel_ang   = (head_ang + 180) % 360
    left_ang   = (head_ang + 90) % 360
    right_ang  = (head_ang - 90) % 360

    head_secs  = _sectors_for_angle(head_ang,  ranges)
    heel_secs  = _sectors_for_angle(heel_ang,  ranges)
    flank_secs = [s for s in open_set if s not in head_secs and s not in heel_secs]

    print(f"[ICS-MEAN] wind={raw_wind:6.1f}°  head={head_secs}  "
          f"flank={flank_secs}  heel={heel_secs}")

    weights   = _weights_per_sector(open_set, head_secs, heel_secs, flank_secs)
    crew_pref = [*heel_secs, *flank_secs, *head_secs]

    actions = {}
    for atype, uid_list in ordinal_map(model).items():
        if not uid_list:
            continue
        if atype == "GroundCrewAgent":
            chosen, idx = [], 0
            for _ in uid_list:
                for k in range(len(crew_pref)):
                    sec = crew_pref[(idx + k) % len(crew_pref)]
                    if sec in open_set:
                        chosen.append(sec + 1)
                        idx = (idx + k + 1) % len(crew_pref)
                        break
                else:                              # fallback
                    chosen.append(rng.choice(list(open_set)) + 1)
            actions[atype] = tuple(chosen)
        else:
            actions[atype] = tuple(
                _draw_for_assets(rng, open_set, weights, len(uid_list))
            )
    return actions

# ───────────────────────────────────────────────────────────────
# End-of-sim helper
# ───────────────────────────────────────────────────────────────
def _end_sim(model):
    from groundcrew import GroundCrewAgent
    for ag in model.schedule.agents:
        if isinstance(ag, GroundCrewAgent):
            ag.flush_clear_buffer()
    # sc = model.fire.calculate_fire_score(model.time)
    # print(f"[SIM] Finished at t={model.time:.0f} min – final fire-score = {sc:.2f}")
    import json
    # final metrics
    area_tot = model.fire.calculate_fire_score(model.time)
    try:
        bldg_tot = model.fire.calculate_building_score(model.time)
        bldg_map = model.fire.calculate_building_breakdown(model.time)
    except AttributeError:  # tif had <10 bands
        bldg_tot, bldg_map = 0, {}

    print(
        f"[SIM] Finished at t={model.time:.0f} min – "
        f"final fire-score = {area_tot:.2f}  "
        f"final buildings-destroyed = {bldg_tot}  "
        f"final buildings-breakdown = {json.dumps(bldg_map, separators=(',', ':'))}"
    )

# ───────────────────────────────────────────────────────────────
# Main simulation loop
# ───────────────────────────────────────────────────────────────
def simulation_loop(model):
    """
    Same pacing as original script, but each slice:
      – refresh forecast → model.latest_forecast_df
      – allocate via dynamic_ics_allocation() using *mean* wind
      – simulate for DECISION_INTERVAL minutes
    """
    schedule_enabled = model.wind_schedule is not None
    next_decision    = model.time
    limit            = model.overall_time_limit

    if schedule_enabled:
        model.latest_forecast_df = dash.get_forecast(
            current_minute=int(model.time))
        print(f"[SIM] Initial forecast updated at t={model.time:.0f} min")

    while model.time < limit:
        if model.time >= limit - DECISION_INTERVAL:
            print("[SIM] Less than one full slice left – terminating.")
            _end_sim(model)
            break

        if model.time >= next_decision:
            # --- forecast refresh so means stay up-to-date
            if schedule_enabled:
                model.latest_forecast_df = dash.get_forecast(
                    current_minute=int(model.time))

            # open_secs = [s for s in range(4) if not model.is_sector_contained(s)]
            num_sectors = getattr(model, "num_sectors",
                                  len(getattr(model, "sector_angle_ranges", [])) or 4)
            open_secs = [s for s in range(num_sectors) if not model.is_sector_contained(s)]
            if not open_secs:
                print("[SIM] All sectors contained – done.")
                _end_sim(model)
                break

            action = dynamic_ics_allocation(model, open_secs)
            simulate_in_place(model, action, duration=DECISION_INTERVAL)
            print(f"Applied action {action} at t={model.time:.0f} min")
            next_decision += DECISION_INTERVAL
            continue

        if not model.step():          # simple advance
            break

    print("Simulation complete.")

# ───────────────────────────────────────────────────────────────
# Build model & run (headless)
# ───────────────────────────────────────────────────────────────
def main():
    print("Starting ICS-MEAN simulation (headless)…")

    dash.pause_event.clear()
    dash.simulation_stop_event.clear()

    model = dash.WildfireModel(
        airtanker_counts=dict(C130J=0, FireHerc=1, Scooper=0,
                              AT802F=0, Dash8_400MRE=0),
        wind_speed=0, wind_direction=220,
        base_positions=[(20000, 20000)],
        lake_positions=[(5000, 5000)],
        time_step=1, debug=False,
        start_time=datetime.datetime.strptime("00:00", "%H:%M"),
        case_folder="ICS_Mean_Case",
        overall_time_limit=2000,
        fire_spread_sim_time=2000,
        operational_delay=0,
        enable_plotting=False,
        groundcrew_count=0, groundcrew_speed=30,
        elapsed_minutes=240,
        groundcrew_sector_mapping=[0],
        second_groundcrew_sector_mapping=None,
        wind_schedule=dash.load_wind_schedule_from_csv_random("wind_schedule_natural.csv"),
        fuel_model_override=None
    )

    simulation_loop(model)

if __name__ == "__main__":
    main()
