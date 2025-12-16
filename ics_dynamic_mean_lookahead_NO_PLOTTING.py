#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ics_dynamic_mean_lookahead_NO_PLOTTING.py
──────────────────────────────────────────────────────────────────────────────
Dynamic ICS baseline that **decides on the next slice’s mean wind**:

At every allocation boundary (every DECISION_INTERVAL minutes) we:
  • Look `LOOKAHEAD = DECISION_INTERVAL` minutes *into the future*,
    pull dir_mean of that forecast bin, and base the head / flank / heel
    classification on that *future* wind.
  • Apply the chosen actions immediately for the coming slice,
    while the fire itself continues to evolve under the truth schedule.

Why?  This mimics an ICS planner that always works one step ahead,
anticipating where the head will have shifted *at the end* of the slice.

Launch with:  `python ics_dynamic_mean_lookahead_NO_PLOTTING.py`
"""

# ───────────────────────────────────────────────────────────────
# Imports & globals
# ───────────────────────────────────────────────────────────────
import numpy as np
import datetime
import time as t

import dashboard as dash
from mcts import simulate_in_place, ordinal_map

DECISION_INTERVAL = 120        # [min]  length of one allocation slice
LOOKAHEAD         = DECISION_INTERVAL   # how far we peek into the future

# ───────────────────────────────────────────────────────────────
# Helpers copied (verbatim) from the “mean” script except where noted
# ───────────────────────────────────────────────────────────────
def _sectors_for_angle(angle_deg, sector_ranges, *, eps=1e-6):
    angle_deg %= 360
    return ([idx for idx, (lo, hi) in enumerate(sector_ranges)
             if lo - eps <= angle_deg < hi + eps] or [0])

def _weights_per_sector(open_set, head, heel, flank):
    BUDGET = dict(head=0.98, flank=0.01, heel=0.01)
    roles  = dict(
        head=[s for s in head if s in open_set],
        heel=[s for s in heel if s in open_set],
        flank=[s for s in flank if s in open_set],
    )
    w = {}
    for role, secs in roles.items():
        if secs:
            sh = BUDGET[role] / len(secs)
            for s in secs:
                w[s] = sh
    tot = sum(w.values()) or 1.0
    for s in w: w[s] /= tot
    return w

def _draw_for_assets(rng, open_set, weights, n):
    secs  = np.array(list(open_set))
    probs = np.array([weights[s] for s in secs]); probs /= probs.sum()
    if len(secs) >= n:
        picks = rng.choice(secs, size=n, replace=False, p=probs)
    else:
        picks = list(rng.choice(secs, size=len(secs), replace=False, p=probs))
        while len(picks) < n: picks.extend(picks)
        picks = np.array(picks[:n])
    return [int(s)+1 for s in picks]

# ───────────────────────────────────────────────────────────────
# LOOK-AHEAD wind helper  ← **NEW LOGIC**
# ───────────────────────────────────────────────────────────────
def _future_wind_direction(model) -> float:
    """
    Return dir_mean for the *next* slice (time + LOOKAHEAD).

    Fallback order identical to the mean script if forecast is missing.
    """
    df   = getattr(model, "latest_forecast_df", None)
    t_now = model.time + LOOKAHEAD
    if df is not None and not df.empty:
        row = df[(df["start_min"] <= t_now) & (t_now < df["end_min"])]
        if not row.empty:                 # found a future bin
            return float(row.iloc[0]["dir_mean"])
        return float(df.iloc[-1]["dir_mean"])  # beyond last bin → use last
    # ---- fallbacks on truth / static ----
    if model.wind_schedule is None:
        return float(model.wind_direction)
    for s, e, _spd, wdir in model.wind_schedule:
        if s <= t_now < e:
            return float(wdir)
    return float(model.wind_schedule[-1][3])

# ───────────────────────────────────────────────────────────────
# Dynamic allocator (same math, but uses _future_wind_direction)
# ───────────────────────────────────────────────────────────────
def _angle_to_sector(angle_deg, sector_ranges):
    angle_deg %= 360
    for idx, (lo, hi) in enumerate(sector_ranges):
        if lo <= angle_deg < hi:
            return idx
    return 0

def dynamic_ics_lookahead(model, open_secs):
    rng       = np.random.default_rng()
    open_set  = set(open_secs)
    model.update_sector_splits()
    ranges    = model.sector_angle_ranges

    raw_wind  = _future_wind_direction(model)
    head_ang  = (180 + raw_wind) % 360
    heel_ang  = (head_ang + 180) % 360
    left_ang  = (head_ang + 90) % 360
    right_ang = (head_ang - 90) % 360

    head_secs  = _sectors_for_angle(head_ang,  ranges)
    heel_secs  = _sectors_for_angle(heel_ang,  ranges)
    flank_secs = [s for s in open_set if s not in head_secs and s not in heel_secs]

    print(f"[ICS-LOOK] now={model.time:.0f} → lookahead wind={raw_wind:5.1f}°  "
          f"head={head_secs} flank={flank_secs} heel={heel_secs}")

    weights   = _weights_per_sector(open_set, head_secs, heel_secs, flank_secs)
    crew_pref = [*heel_secs, *flank_secs, *head_secs]

    actions = {}
    for atype, uid_list in ordinal_map(model).items():
        if not uid_list: continue
        if atype == "GroundCrewAgent":
            chosen, idx = [], 0
            for _ in uid_list:
                for k in range(len(crew_pref)):
                    sec = crew_pref[(idx+k) % len(crew_pref)]
                    if sec in open_set:
                        chosen.append(sec+1); idx = (idx+k+1) % len(crew_pref); break
                else:
                    chosen.append(rng.choice(list(open_set))+1)
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
# Simulation loop
# ───────────────────────────────────────────────────────────────
def simulation_loop(model):
    limit         = model.overall_time_limit
    next_decision = model.time

    model.latest_forecast_df = dash.get_forecast(current_minute=int(model.time))

    while model.time < limit:
        if model.time >= limit - DECISION_INTERVAL:
            print("[SIM] < slice left – quitting."); _end_sim(model); break

        if model.time >= next_decision:
            model.latest_forecast_df = dash.get_forecast(current_minute=int(model.time))

            # open_secs = [s for s in range(4) if not model.is_sector_contained(s)]
            num_sectors = getattr(model, "num_sectors",
                                  len(getattr(model, "sector_angle_ranges", [])) or 4)
            open_secs = [s for s in range(num_sectors) if not model.is_sector_contained(s)]
            if not open_secs:
                print("[SIM] All sectors contained – done."); _end_sim(model); break

            action = dynamic_ics_lookahead(model, open_secs)
            simulate_in_place(model, action, duration=DECISION_INTERVAL)
            print(f"Applied {action} at t={model.time:.0f}")
            next_decision += DECISION_INTERVAL
            continue

        if not model.step(): break

    print("Simulation complete.")

# ───────────────────────────────────────────────────────────────
# Build model & run
# ───────────────────────────────────────────────────────────────
def main():
    dash.pause_event.clear(); dash.simulation_stop_event.clear()
    print("Starting ICS-LOOKAHEAD simulation…")

    model = dash.WildfireModel(
        airtanker_counts=dict(C130J=0, FireHerc=1, Scooper=0,
                              AT802F=0, Dash8_400MRE=0),
        wind_speed=0, wind_direction=220,
        base_positions=[(20000, 20000)],
        lake_positions=[(5000, 5000)],
        time_step=1, debug=False,
        start_time=datetime.datetime.strptime("00:00", "%H:%M"),
        case_folder="ICS_Lookahead_Case",
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
