
"""
dashboard_ics_buildings_headless.py – building-priority ICS (command-line)

This variant ranks/weights sectors by the total number of buildings *in that sector
wedge across the whole map* (not how many have burned).

• Ground crews: deterministic, highest-building-count sector first, then 2nd, etc.
• Airtankers (and any non-GroundCrew assets): weighted random by each sector's
  share of buildings; weights are renormalized over *open* sectors each decision.
  Example: if one sector contains 90% of all buildings on the map, it will be chosen
  ~90% of the decision points (until that sector becomes contained).

Launch with:  python dashboard_ics_buildings_headless.py
"""

import time as t
import numpy as np
import datetime
from typing import List, Tuple, Dict

# Import the original “dashboard” module for its WildfireModel and forecast functions.
import dashboard as dash
from mcts import simulate_in_place, ordinal_map

# Decision interval (minutes) used in the simulation loop.
DECISION_INTERVAL = 120


# ───────────────────────────────────────────────────────────────
# Helpers to compute per-sector building totals (static, once)
# ───────────────────────────────────────────────────────────────
def _sector_angle_ranges(model) -> List[Tuple[float, float]]:
    """Ensure sector geometry exists and return [(lo, hi), ...] in degrees."""
    model.update_sector_splits()
    return list(model.sector_angle_ranges)


def _grid_cell_centers_xy(transform, shape):
    """
    Affine transform to per-cell centers in model coords (meters).
    x = c + a*(col + 0.5),   y = f + e*(row + 0.5)
    """
    rows, cols = shape
    a = transform[0]; c = transform[2]
    e = transform[4]; f = transform[5]
    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))
    x = c + (jj + 0.5) * a
    y = f + (ii + 0.5) * e
    return x, y


def _angles_from_ignition(x, y, ignition_xy) -> np.ndarray:
    """Polar angle (deg in [0,360)) of each cell center relative to ignition point."""
    ix, iy = ignition_xy
    ang = np.degrees(np.arctan2(y - iy, x - ix)) % 360.0
    return ang.astype(np.float32)


def _buildings_mask_from_model(model) -> np.ndarray:
    """
    Boolean mask (rows, cols) of “building” cells. Uses the fire model's
    built_char band if present: built_char >= 11 → building.
    If no band exists, returns an all-False mask.
    """
    fire = getattr(model, "fire", None)
    built_char = getattr(fire, "built_char_np", None)
    if built_char is None:
        return np.zeros(model.grid_size, dtype=bool)
    return (built_char >= 11)


def _count_buildings_per_sector(model) -> Tuple[np.ndarray, int]:
    """
    Return (counts_per_sector [num_sectors], total_buildings).
    Bins *all* building cells by the model's sector wedges (full-map extent).
    """
    ranges = _sector_angle_ranges(model)               # [(lo,hi), ...]
    num_sectors = getattr(model, "num_sectors", len(ranges) or 4)

    # Geometry
    fire = model.fire
    x, y = _grid_cell_centers_xy(fire.transform, fire.grid_size)
    ang = _angles_from_ignition(x, y, fire.ignition_pt)

    # Buildings mask
    bmask = _buildings_mask_from_model(model)
    if not np.any(bmask):
        return np.zeros(num_sectors, dtype=np.int64), 0

    ang_b = ang[bmask]

    # Bin by sector ranges
    counts = np.zeros(num_sectors, dtype=np.int64)
    for s, (lo, hi) in enumerate(ranges):
        # include lo, exclude hi (consistent with sector indexing elsewhere)
        sel = (ang_b >= lo) & (ang_b < hi)
        counts[s] = int(np.count_nonzero(sel))

    return counts, int(counts.sum())


# ───────────────────────────────────────────────────────────────
# Allocation logic driven by per-sector building totals
# ───────────────────────────────────────────────────────────────
def _weights_from_buildings(open_set: set[int],
                            counts: np.ndarray) -> Dict[int, float]:
    """
    Probability weights for airtanker-type assets.
    w[s] ∝ total_buildings_in_sector[s], renormalized over open sectors.
    If no buildings (sum=0), use uniform over open sectors.
    """
    if not open_set:
        return {}

    base = {s: float(counts[s]) for s in open_set}
    total = sum(base.values())
    if total <= 0:
        # uniform fallback
        u = 1.0 / len(open_set)
        return {s: u for s in open_set}
    return {s: base[s] / total for s in open_set}


def _draw_for_assets(rng, open_set, weights, n):
    """
    Draw n distinct sectors (without replacement) according to supplied weights.
    If there are fewer unique sectors than n, cycle.
    Convert to 1-based numbering for simulate_in_place().
    """
    sectors = np.array(list(open_set))
    probs = np.array([weights.get(s, 0.0) for s in sectors], dtype=float)
    if probs.sum() <= 0:
        probs[:] = 1.0 / len(sectors)
    else:
        probs /= probs.sum()

    if len(sectors) >= n:
        picks = rng.choice(sectors, size=n, replace=False, p=probs)
    else:
        picks = list(rng.choice(sectors, size=len(sectors), replace=False, p=probs))
        while len(picks) < n:
            picks.extend(picks)
        picks = np.array(picks[:n])
    return [int(s) + 1 for s in picks]  # 1-based


def _groundcrew_order_from_buildings(open_set: set[int],
                                     counts: np.ndarray) -> List[int]:
    """
    Deterministic priority list for ground crews:
    sort *open* sectors by building count desc, ties → lower index first.
    Returns sector indices (0-based).
    """
    ranked = sorted(open_set, key=lambda s: (-counts[s], s))
    return ranked if ranked else list(open_set)


def building_priority_allocation(model,
                                 open_secs: List[int],
                                 counts: np.ndarray) -> Dict[str, Tuple[int, ...]]:
    """
    Determine ICS allocation based on total buildings per sector.
    GroundCrewAgent → deterministic high→low.
    Others (airtankers) → weighted random by building share on *open* sectors.

    Returns action dict mapping asset type to tuple of 1-based sector indices.
    """
    open_set = set(open_secs)
    rng = np.random.default_rng()

    # Weights for aircraft (and any non-GC assets)
    weights = _weights_from_buildings(open_set, counts)
    # Deterministic order for ground crews
    crew_pref = _groundcrew_order_from_buildings(open_set, counts)

    # Debug print
    print("[ICS] building counts per sector (all-map):", counts.tolist())
    tot_b = int(counts.sum())
    if tot_b > 0:
        shares = [round(100.0 * (counts[i] / tot_b), 1) for i in range(len(counts))]
        print("[ICS] building share per sector (% of map):", shares)

    actions = {}
    for atype, uid_list in ordinal_map(model).items():
        if not uid_list:
            continue

        if atype == "GroundCrewAgent":
            # walk the deterministic ranking, cycling if more crews than sectors
            chosen = []
            pref_idx = 0
            for _ in uid_list:
                if not crew_pref:
                    # fallback: random open sector
                    sec = rng.choice(list(open_set))
                    chosen.append(sec + 1)
                    continue
                sec = crew_pref[pref_idx % len(crew_pref)]
                # ensure it's still open; if not, find the next open one
                if sec not in open_set:
                    alt = [s for s in crew_pref if s in open_set]
                    if alt:
                        sec = alt[0]
                    else:
                        # all closed? pick any original open as a safety
                        sec = next(iter(open_set))
                chosen.append(sec + 1)
                pref_idx += 1
            actions[atype] = tuple(chosen)
        else:
            picks = _draw_for_assets(rng, open_set, weights, len(uid_list))
            actions[atype] = tuple(picks)

    return actions


# ───────────────────────────────────────────────────────────────
# End-of-simulation helper (headless version)
# ───────────────────────────────────────────────────────────────
def end_simulation(model):
    """
    Finalizes the simulation by flushing any pending ground-crew work and printing
    the final fire & building scores.
    """
    from groundcrew import GroundCrewAgent
    for ag in model.schedule.agents:
        if isinstance(ag, GroundCrewAgent):
            ag.flush_clear_buffer()

    import json
    area_tot = model.fire.calculate_fire_score(model.time)
    try:
        bldg_tot = model.fire.calculate_building_score(model.time)
        bldg_map = model.fire.calculate_building_breakdown(model.time)
    except AttributeError:
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
def simulation_loop_building(model):
    """
    Headless loop for building-driven ICS allocation.
    At each decision boundary (every DECISION_INTERVAL minutes), the loop
    recomputes the open sectors, applies the building-based allocation, and
    advances the simulation via simulate_in_place().
    """
    schedule_enabled = model.wind_schedule is not None
    next_decision_time = model.time
    overall_limit = model.overall_time_limit

    # Not strictly required here, but keep parity with the other script
    if schedule_enabled:
        forecast_df = dash.get_forecast(current_minute=int(model.time))
        model.latest_forecast_df = forecast_df
        print(f"[SIM] Initial forecast updated at t={model.time} min")

    # ── Compute per-sector building totals ONCE (whole-map, static) ──
    #     They do NOT depend on fire state; “sector wedge extended to map”.
    counts, total_b = _count_buildings_per_sector(model)
    print(f"[ICS] Total buildings on map (codes ≥ 11): {total_b}")
    if total_b == 0:
        print("[ICS] No building band found or zero buildings; falling back to uniform weighting.")

    while model.time < overall_limit:
        if model.time >= overall_limit - DECISION_INTERVAL:
            print("[SIM] Less than one full decision slice left – terminating.")
            end_simulation(model)
            break

        if model.time >= next_decision_time:
            num_sectors = getattr(model, "num_sectors",
                                  len(getattr(model, "sector_angle_ranges", [])) or 4)
            open_secs = [s for s in range(num_sectors) if not model.is_sector_contained(s)]
            if not open_secs:
                print("[SIM] All sectors contained – ending simulation.")
                end_simulation(model)
                break

            action = building_priority_allocation(model, open_secs, counts)
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
    print("Starting building-priority ICS simulation (headless mode)...")

    # Same defaults as your deterministic ICS headless script
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
        enable_plotting=False,  # headless
        groundcrew_count=0,
        groundcrew_speed=30,
        elapsed_minutes=240,
        groundcrew_sector_mapping=[0],
        second_groundcrew_sector_mapping=None,
        wind_schedule=dash.load_wind_schedule_from_csv_random("wind_schedule_natural.csv"),
        fuel_model_override=None
    )

    simulation_loop_building(model)


if __name__ == "__main__":
    main()
