#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ics_dynamic_truth_lookahead_NO_PLOTTING.py
──────────────────────────────────────────────────────────────────────────────
Dynamic ICS baseline that **looks one decision slice into the future, but uses
the *truth* wind schedule (not the forecast means) to decide where the head /
flank / heel will be**.

•  The fire itself already evolves under that truth schedule, so this strategy
   is effectively an “oracle” that can anticipate the exact wind shift for the
   coming slice.
•  Ground-crew and airtanker budgets / priorities are unchanged.

Launch with:  `python ics_dynamic_truth_lookahead_NO_PLOTTING.py`
"""

# ───────────────────────────────────────────────────────────────
# Imports & globals
# ───────────────────────────────────────────────────────────────
import numpy as np
import datetime
import time as t

import dashboard as dash
from mcts import simulate_in_place, ordinal_map

DECISION_INTERVAL = 120      # minutes per allocation slice
LOOKAHEAD         = DECISION_INTERVAL  # horizon we peek into

# ───────────────────────────────────────────────────────────────
# Shared helpers (unchanged from other variants)
# ───────────────────────────────────────────────────────────────
def _sectors_for_angle(angle_deg, sector_ranges, *, eps=1e-6):
    angle_deg %= 360
    return ([idx for idx, (lo, hi) in enumerate(sector_ranges)
             if lo - eps <= angle_deg < hi + eps] or [0])

def _weights_per_sector(open_set, head, heel, flank):
    BUDGET = dict(head=0.98, flank=0.01, heel=0.01)
    roles  = dict(
        head=[s for s in head  if s in open_set],
        heel=[s for s in heel  if s in open_set],
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
# **Truth-schedule look-ahead wind helper**
# ───────────────────────────────────────────────────────────────
def _truth_wind_direction_lookahead(model) -> float:
    """
    Returns the wind direction (deg) *LOOKAHEAD* minutes in the future,
    based solely on the model's truth schedule.
    """
    t_future = model.time + LOOKAHEAD

    if model.wind_schedule is None:               # static run
        return float(model.wind_direction)

    for start, end, _spd, wdir in model.wind_schedule:
        if start <= t_future < end:
            return float(wdir)
    return float(model.wind_schedule[-1][3])      # beyond last bin → last dir

# ───────────────────────────────────────────────────────────────
# Dynamic allocator (identical budgets, different wind helper)
# ───────────────────────────────────────────────────────────────
def _angle_to_sector(angle_deg, sector_ranges):
    angle_deg %= 360
    for idx, (lo, hi) in enumerate(sector_ranges):
        if lo <= angle_deg < hi:
            return idx
    return 0

def dynamic_ics_truth_lookahead(model, open_secs):
    rng       = np.random.default_rng()
    open_set  = set(open_secs)
    model.update_sector_splits()
    ranges    = model.sector_angle_ranges

    raw_wind  = _truth_wind_direction_lookahead(model)
    head_ang  = (180 + raw_wind) % 360
    heel_ang  = (head_ang + 180) % 360
    left_ang  = (head_ang + 90) % 360
    right_ang = (head_ang - 90) % 360

    head_secs  = _sectors_for_angle(head_ang,  ranges)
    heel_secs  = _sectors_for_angle(heel_ang,  ranges)
    flank_secs = [s for s in open_set if s not in head_secs and s not in heel_secs]

    print(f"[ICS-ORACLE] now={model.time:.0f}  lookahead wind={raw_wind:5.1f}°  "
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
    # print(f"[SIM] Finished t={model.time:.0f} min – final fire-score = {sc:.2f}")
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

    while model.time < limit:
        if model.time >= limit - DECISION_INTERVAL:
            print("[SIM] < slice left – quitting."); _end_sim(model); break

        if model.time >= next_decision:
            # open_secs = [s for s in range(4) if not model.is_sector_contained(s)]
            num_sectors = getattr(model, "num_sectors",
                                  len(getattr(model, "sector_angle_ranges", [])) or 4)
            open_secs = [s for s in range(num_sectors) if not model.is_sector_contained(s)]
            if not open_secs:
                print("[SIM] All sectors contained – done."); _end_sim(model); break

            action = dynamic_ics_truth_lookahead(model, open_secs)
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
    print("Starting ICS-TRUTH-LOOKAHEAD simulation…")

    model = dash.WildfireModel(
        airtanker_counts=dict(C130J=0, FireHerc=1, Scooper=0,
                              AT802F=0, Dash8_400MRE=0),
        wind_speed=0, wind_direction=220,
        base_positions=[(20000, 20000)],
        lake_positions=[(5000, 5000)],
        time_step=1, debug=False,
        start_time=datetime.datetime.strptime("00:00", "%H:%M"),
        case_folder="ICS_Truth_Lookahead_Case",
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
