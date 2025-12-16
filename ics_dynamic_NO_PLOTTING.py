#!/usr/bin/env python
"""
dashboard_ics_dynamic_headless.py – deterministic ICS baseline (command‐line version)

Compared with the original dashboard_ics_dynamic.py the Panel/UI elements
have been removed. This script builds a WildfireModel with default parameters
and runs a headless simulation loop that applies dynamic ICS allocation.

Launch with:  python dashboard_ics_dynamic_headless.py
"""

import time as t
import numpy as np
import datetime

# Import the original “dashboard” module for its WildfireModel and forecast functions.
import dashboard as dash
from mcts import simulate_in_place, ordinal_map

# Decision interval (minutes) used in the simulation loop.
DECISION_INTERVAL = 120




# ───────────────────────────────────────────────────────────────
# Helper functions for ICS allocation
# ───────────────────────────────────────────────────────────────
def _sectors_for_angle(angle_deg: float, sector_ranges: list[tuple[float, float]], *, eps: float = 1e-6) -> list[int]:
    """
    Return the list of sector indices whose range touches angle_deg.
    If the angle lies exactly on a boundary (within eps) you get two adjacent sectors.
    """
    angle_deg %= 360
    hits = []
    for idx, (lo, hi) in enumerate(sector_ranges):
        if lo - eps <= angle_deg < hi + eps:
            hits.append(idx)
    return hits or [0]  # safety fallback


def _weights_per_sector(open_set: set[int],
                        head_secs: list[int],
                        heel_secs: list[int],
                        flank_secs: list[int]) -> dict[int, float]:
    """
    For airtankers allocate probability weights according to the following budgets:
       head  → 0.98
       flank → 0.01
       heel  → 0.01
    These budgets are divided among sectors of that role that remain open.
    The result always sums to 1.0.
    """
    BUDGET = dict(head=0.98, flank=0.01, heel=0.01)
    roles = dict(
        head=[s for s in head_secs if s in open_set],
        heel=[s for s in heel_secs if s in open_set],
        flank=[s for s in flank_secs if s in open_set],
    )
    w = {}
    for role, secs in roles.items():
        if secs:
            share = BUDGET[role] / len(secs)
            for s in secs:
                w[s] = share
    # Renormalize in case some roles are missing
    total = sum(w.values())
    for s in w:
        w[s] /= total
    return w


def _draw_for_assets(rng, open_set, weights, n):
    """
    Draw n distinct sectors (without replacement) according to the supplied
    probability weights. If there are fewer unique sectors than n, cycle the draws.
    Convert indices to 1‐based numbering for simulate_in_place().
    """
    sectors = np.array(list(open_set))
    probs = np.array([weights[s] for s in sectors])
    probs = probs / probs.sum()

    if len(sectors) >= n:
        picks = rng.choice(sectors, size=n, replace=False, p=probs)
    else:  # not enough unique sectors
        picks = list(rng.choice(sectors, size=len(sectors), replace=False, p=probs))
        while len(picks) < n:
            picks.extend(picks)
        picks = np.array(picks[:n])
    return [int(s) + 1 for s in picks]


def _latest_wind_direction(model) -> float:
    """
    Return the current wind direction (in degrees) from the model.
    If a dynamic wind schedule is active, returns the direction of the active segment.
    """
    if model.wind_schedule is None:  # static wind
        return float(model.wind_direction)
    now = model.time
    for start, end, _spd, wdir in model.wind_schedule:
        if start <= now < end:
            return float(wdir)
    return float(model.wind_schedule[-1][3])


def _angle_to_sector(angle_deg: float, sector_ranges: list[tuple[float, float]]) -> int:
    """
    Given an absolute angle (in degrees) return the index of the sector containing it.
    """
    angle_deg %= 360
    for idx, (lo, hi) in enumerate(sector_ranges):
        if lo <= angle_deg < hi:
            return idx
    return 0  # safety fallback


def _priority_lists(model):
    """
    Build two ordered lists of sector indices based on the current wind:
       - For airtankers: [head, left flank, right flank, heel]
       - For ground crews: [heel, left flank, right flank, head]
    """
    # Update sectors and get the sector angle ranges.
    model.update_sector_splits()
    ranges = model.sector_angle_ranges

    # Compute wind-based angles.
    raw_wind = _latest_wind_direction(model)
    # Convert to model coordinates: fire-head is opposite the wind.
    head_ang = (180 + raw_wind) % 360
    heel_ang = (head_ang + 180) % 360
    left_ang = (head_ang + 90) % 360  # left (looking downwind)
    right_ang = (head_ang - 90) % 360

    print(
        f"[ICS] wind={raw_wind:5.1f}°  → head={head_ang:.1f}°, heel={heel_ang:.1f}°, L={left_ang:.1f}°, R={right_ang:.1f}°")

    head_sec = _angle_to_sector(head_ang, ranges)
    heel_sec = _angle_to_sector(heel_ang, ranges)
    lflk_sec = _angle_to_sector(left_ang, ranges)
    rflk_sec = _angle_to_sector(right_ang, ranges)

    # Remove duplicates if wind lies on a boundary.
    def uniq(lst):
        seen = set()
        out = []
        for s in lst:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    tanker_order = uniq([head_sec, lflk_sec, rflk_sec, heel_sec])
    crew_order = uniq([heel_sec, lflk_sec, rflk_sec, head_sec])
    return tanker_order, crew_order


def dynamic_ics_allocation(model, open_secs):
    """
    Determines ICS allocation as follows:
      • For ground-crew agents use a deterministic priority:
            heel → flank(s) → head(s)
      • For airtankers use a weighted random draw with budgets:
            head:   0.60 (effective via _weights_per_sector with head allocation)
            flank:  0.30 (shared equally)
            heel:   0.10
      Only sectors that are still open (not yet contained) are considered.
    Returns an action dict mapping asset types to a tuple of 1-based sector indices.
    """
    open_set = set(open_secs)
    rng = np.random.default_rng()

    # Update sector geometry information.
    model.update_sector_splits()
    ranges = model.sector_angle_ranges  # e.g. [(0,90), (90,180), ...]

    raw_wind = _latest_wind_direction(model)
    head_ang = (180 + raw_wind) % 360  # fire-head bearing in model coords
    heel_ang = (head_ang + 180) % 360

    head_secs = _sectors_for_angle(head_ang, ranges)
    heel_secs = _sectors_for_angle(heel_ang, ranges)
    flank_secs = [s for s in open_set if s not in head_secs and s not in heel_secs]

    print(
        f"[ICS] wind={raw_wind:6.1f}°  head={head_secs}  flank={flank_secs}  heel={heel_secs}  open={sorted(open_set)}")

    # Compute probability weights for the airtankers.
    weights = _weights_per_sector(open_set,
                                  head_secs=head_secs,
                                  heel_secs=heel_secs,
                                  flank_secs=flank_secs)

    # Determine deterministic order for ground-crew agents.
    crew_pref = [*heel_secs, *flank_secs, *head_secs]

    actions = {}
    for atype, uid_list in ordinal_map(model).items():
        if not uid_list:
            continue

        if atype == "GroundCrewAgent":
            chosen = []
            pref_idx = 0
            for _ in uid_list:
                for k in range(len(crew_pref)):
                    sec = crew_pref[(pref_idx + k) % len(crew_pref)]
                    if sec in open_set:
                        chosen.append(sec + 1)  # convert to 1-based indexing
                        pref_idx = (pref_idx + k + 1) % len(crew_pref)
                        break
                else:
                    # Fallback: pick a random open sector.
                    sec = rng.choice(list(open_set))
                    chosen.append(sec + 1)
            actions[atype] = tuple(chosen)
        else:
            # For airtankers, perform weighted random draw.
            picks = _draw_for_assets(rng, open_set, weights, len(uid_list))
            actions[atype] = tuple(picks)

    return actions


# ───────────────────────────────────────────────────────────────
# End-of-simulation helper (headless version)
# ───────────────────────────────────────────────────────────────
def end_simulation(model):
    """
    Finalizes the simulation by flushing any pending ground-crew work and printing
    the final fire score.
    """
    from groundcrew import GroundCrewAgent
    for ag in model.schedule.agents:
        if isinstance(ag, GroundCrewAgent):
            ag.flush_clear_buffer()
    # score = model.fire.calculate_fire_score(model.time)
    # print(f"[SIM] Finished at t={model.time:.0f} min – final fire-score = {score:.2f}")
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
# Simulation Loop (Headless & Command-Line)
# ───────────────────────────────────────────────────────────────
def simulation_loop_ics(model):
    """
    Headless simulation loop for dynamic ICS allocation.
    At each decision boundary (every DECISION_INTERVAL minutes), the loop
    determines the open sectors, computes the dynamic allocation, applies the
    chosen action via simulate_in_place(), and advances the simulation.
    """
    schedule_enabled = model.wind_schedule is not None
    next_decision_time = model.time
    overall_limit = model.overall_time_limit

    if schedule_enabled:
        forecast_df = dash.get_forecast(current_minute=int(model.time))
        model.latest_forecast_df = forecast_df
        print(f"[SIM] Initial forecast updated at t={model.time} min")

    while model.time < overall_limit:
        if model.time >= overall_limit - DECISION_INTERVAL:
            print("[SIM] Less than one full decision slice left – terminating.")
            end_simulation(model)
            break

        if model.time >= next_decision_time:
            # open_secs = [s for s in range(4) if not model.is_sector_contained(s)]
            # NEW
            num_sectors = getattr(model, "num_sectors",
                                  len(getattr(model, "sector_angle_ranges", [])) or 4)
            open_secs = [s for s in range(num_sectors) if not model.is_sector_contained(s)]
            if not open_secs:
                print("[SIM] All sectors contained – ending simulation.")
                end_simulation(model)
                break

            action = dynamic_ics_allocation(model, open_secs)
            simulate_in_place(model, action, duration=DECISION_INTERVAL)
            print(f"Applied action {action} at t={model.time} min")
            next_decision_time += DECISION_INTERVAL
            continue
        else:
            if not model.step():
                break

    print("Simulation complete.")


# ───────────────────────────────────────────────────────────────
# Main entry point
# ───────────────────────────────────────────────────────────────
def main():
    print("Starting deterministic ICS simulation (headless mode)...")

    # Build a WildfireModel instance with default parameters.
    # (Parameters are set to match the baseline simulation; adjust as needed.)
    model = dash.WildfireModel(
        airtanker_counts={
            "C130J": 0,
            "FireHerc": 1,
            "Scooper": 0,
            "AT802F": 0,
            "Dash8_400MRE": 0,
        },
        wind_speed=0,
        wind_direction=220,
        base_positions=[(20000, 20000)],
        lake_positions=[(5000, 5000)],
        time_step=1,
        debug=False,
        start_time=datetime.datetime.strptime("00:00", "%H:%M"),
        case_folder="Dashboard_Case",
        overall_time_limit=2000,
        fire_spread_sim_time=2000,
        operational_delay=0,
        enable_plotting=False,  # disable plotting for command-line use
        groundcrew_count=0,
        groundcrew_speed=30,
        elapsed_minutes=240,
        groundcrew_sector_mapping=[0],
        second_groundcrew_sector_mapping=None,
        wind_schedule=dash.load_wind_schedule_from_csv_random("wind_schedule_natural.csv"),
        fuel_model_override=None
    )

    simulation_loop_ics(model)


if __name__ == "__main__":
    main()