#!/usr/bin/env python
"""
ics_truth_burned_buildings_NO_PLOTTING.py
──────────────────────────────────────────────────────────────────────────────
Strategy:
  • Precompute, ONCE, the number of buildings that would burn by the final
    time in each sector under the TRUTH wind schedule (no suppression).
  • Ground crews are assigned deterministically in descending order of those
    per-sector burned-building totals (skipping contained sectors).
  • Aircraft are assigned via weighted draws where the probability weight of a
    sector ∝ its burned-building total (restricted to open sectors).

Notes:
  • This script is compatible with run_all_strategies_batch.py. The wrapper
    patches model.WildfireModel.__init__ to inject the TRUTH schedule and the
    run’s output folder; we simply build a model and run the loop.
  • We do NOT recompute the truth fire at each decision slice.
"""

import numpy as np
import datetime
import cupy as cp

# Use the same WildfireModel / simulate helpers your other strategies use
import dashboard as dash                          # imports WildfireModel from model.py
from mcts import simulate_in_place, ordinal_map   # action application & asset map

from FIRE_MODEL_CUDA import SurrogateFireModelROS

# minutes between decision points
DECISION_INTERVAL = 120


# ───────────────────────────────────────────────────────────────
# Utility: 1-based weighted draws for aircraft, skipping contained sectors
# ───────────────────────────────────────────────────────────────
def _draw_for_assets(rng, open_set, weights_dict, n):
    """
    Draw n sectors (without replacement when possible) according to weights in
    weights_dict[s] (s are 0-based sector indices). Return 1-based sector ids.
    """
    sectors = np.array(sorted(open_set))
    w = np.array([max(float(weights_dict.get(s, 0.0)), 0.0) for s in sectors], dtype=float)
    if w.sum() <= 0:
        # fallback: uniform when all weights are zero among the open set
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()

    if len(sectors) >= n:
        picks = rng.choice(sectors, size=n, replace=False, p=w)
    else:
        picks = list(rng.choice(sectors, size=len(sectors), replace=False, p=w))
        while len(picks) < n:
            picks.extend(picks)
        picks = np.array(picks[:n])
    return [int(s) + 1 for s in picks]


# ───────────────────────────────────────────────────────────────
# Precompute: per-sector buildings burned by final time under TRUTH schedule
# ───────────────────────────────────────────────────────────────
def _tif_path_from_model(model) -> str:
    """Best effort to get the landscape tif path from the live model."""
    try:
        return str(model.fire.tif_path)
    except Exception:
        # Fallback: most of your batch jobs use this file name; tweak if needed
        return "camp_fire_three_enhanced.tif"


def _compute_burned_buildings_per_sector_truth(model) -> list[int]:
    """
    Build a fresh surrogate model using the TRUTH schedule and no suppression,
    then count how many *buildings* (built_char ≥ 11) have arrival_time ≤ final
    time inside each sector wedge.
    """
    # sector geometry (degrees, 0 ≤ θ < 360), 0-based sector indices
    model.update_sector_splits()
    sector_ranges: list[tuple[float, float]] = model.sector_angle_ranges
    num_sectors = len(sector_ranges) or getattr(model, "num_sectors", 4)

    tif_path = _tif_path_from_model(model)
    final_time = float(model.fire_spread_sim_time)

    # Build a *new* surrogate so suppression during the run doesn't affect it.
    dummy_speed = model.wind_schedule[-1][2] if model.wind_schedule else float(model.wind_speed)
    dummy_dir   = model.wind_schedule[-1][3] if model.wind_schedule else float(model.wind_direction)

    fm_truth = SurrogateFireModelROS(
        tif_path=tif_path,
        sim_time=int(final_time),
        wind_speed=float(dummy_speed),                 # ignored when wind_schedule provided
        wind_direction_deg=float(dummy_dir),           # ignored when wind_schedule provided
        max_iter=250,
        tol=1e-3,
        wind_schedule=model.wind_schedule
    )
    cp.cuda.Stream.null.synchronize()

    # masks on CPU
    T_np = cp.asnumpy(fm_truth.T)                              # arrival times (minutes)
    built = fm_truth.built_char_np
    if built is None:
        print("[PRECOMPUTE] No building band present; all sector weights = 0")
        return [0]*num_sectors

    # burned buildings mask at final time
    burned_buildings = np.isfinite(T_np) & (T_np <= final_time) & (built >= 11)

    if not burned_buildings.any():
        print("[PRECOMPUTE] No buildings burn under the truth schedule.")
        return [0]*num_sectors

    # Compute polar angle from ignition → each burned building cell
    rows_idx, cols_idx = np.where(burned_buildings)

    # cell-center coordinates using the affine transform
    a = fm_truth.transform[0]     # pixel width (x scale)
    c = fm_truth.transform[2]     # x origin
    e = fm_truth.transform[4]     # pixel height (y scale, typically negative)
    f = fm_truth.transform[5]     # y origin

    xs = c + (cols_idx + 0.5) * a
    ys = f + (rows_idx + 0.5) * e
    ign_x, ign_y = fm_truth.ignition_pt

    angles = (np.degrees(np.arctan2(ys - ign_y, xs - ign_x)) + 360.0) % 360.0

    # Bin into your model’s sector wedges
    counts = [0] * num_sectors
    for s, (lo, hi) in enumerate(sector_ranges):
        # include lower bound, exclude upper (consistent with model code)
        counts[s] = int(np.sum((angles >= lo) & (angles < hi)))

    total = int(np.sum(counts))
    print(f"[PRECOMPUTE] Truth burned buildings per sector (final t={final_time:.0f}): "
          f"{counts}  | total={total}")
    return counts


# ───────────────────────────────────────────────────────────────
# Allocation policy
# ───────────────────────────────────────────────────────────────
def truth_burned_buildings_allocation(model, open_secs, burned_counts):
    """
    Ground crews:
        deterministic priority = sectors sorted by burned_counts DESC.
    Aircraft:
        weighted random draw where P(sector) ∝ burned_counts[sector]
        (restricted to open sectors; uniform if all weights are zero).
    """
    open_set = set(open_secs)
    rng = np.random.default_rng()

    # deterministic order for crews (skip contained sectors)
    order = sorted(open_set, key=lambda s: (burned_counts[s], -s), reverse=True)

    # aircraft weights among open sectors
    weights = {s: float(burned_counts[s]) for s in open_set}
    if sum(weights.values()) <= 0:
        # uniform fallback if nothing is predicted to burn
        eq = 1.0 / max(len(open_set), 1)
        weights = {s: eq for s in open_set}

    actions = {}
    for atype, uid_list in ordinal_map(model).items():
        if not uid_list:
            continue

        if atype == "GroundCrewAgent":
            chosen = []
            if order:
                # cycle through the priority list for multiple crews
                idx = 0
                for _ in uid_list:
                    # skip any sector that might have been contained since last tick
                    for k in range(len(order)):
                        s = order[(idx + k) % len(order)]
                        if s in open_set:
                            chosen.append(s + 1)   # 1-based
                            idx = (idx + k + 1) % len(order)
                            break
                    else:
                        # all closed → random open sector
                        chosen.append(int(rng.choice(list(open_set))) + 1)
            else:
                # no open sectors (shouldn’t happen here)
                chosen = [1] * len(uid_list)
            actions[atype] = tuple(chosen)
        else:
            picks = _draw_for_assets(rng, open_set, weights, len(uid_list))
            actions[atype] = tuple(picks)

    # for visibility in logs
    if open_set:
        ws = {s: round(float(weights.get(s, 0.0)), 3) for s in sorted(open_set)}
        print(f"[ICS] open={sorted(open_set)}  crew_order={order}  weights={ws}")
    return actions


# ───────────────────────────────────────────────────────────────
# End-of-simulation helper (same print signature the batch parser expects)
# ───────────────────────────────────────────────────────────────
def end_simulation(model):
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
# Simulation loop
# ───────────────────────────────────────────────────────────────
def simulation_loop(model):
    """
    At each decision boundary:
      • determine open sectors
      • choose actions using the fixed truth-burned-buildings ranking
      • apply via simulate_in_place()
    """
    next_decision_time = model.time
    overall_limit = model.overall_time_limit

    # PRECOMPUTE once
    burned_counts = _compute_burned_buildings_per_sector_truth(model)

    while model.time < overall_limit:
        # graceful early stop when less than one full slice remains
        if model.time >= overall_limit - DECISION_INTERVAL:
            print("[SIM] Less than one full decision slice left – terminating.")
            end_simulation(model)
            break

        if model.time >= next_decision_time:
            # sectors still not contained
            num_sectors = getattr(model, "num_sectors",
                                  len(getattr(model, "sector_angle_ranges", [])) or 4)
            open_secs = [s for s in range(num_sectors) if not model.is_sector_contained(s)]
            if not open_secs:
                print("[SIM] All sectors contained – ending simulation.")
                end_simulation(model)
                break

            action = truth_burned_buildings_allocation(model, open_secs, burned_counts)
            simulate_in_place(model, action, duration=DECISION_INTERVAL)
            print(f"Applied action {action} at t={model.time} min")
            next_decision_time += DECISION_INTERVAL
            continue
        else:
            if not model.step():
                break

    print("Simulation complete.")


# ───────────────────────────────────────────────────────────────
# Main (parameters here are placeholders; the batch wrapper overrides them)
# ───────────────────────────────────────────────────────────────
def main():
    print("Starting TRUTH-burned-buildings ICS simulation (headless)…")

    # The batch wrapper will patch WildfireModel.__init__ to:
    #   • inject the TRUTH wind_schedule
    #   • set case_folder to the run’s output dir
    #   • set num_sectors from CLI
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
        overall_time_limit=3000,
        fire_spread_sim_time=3000,
        operational_delay=0,
        enable_plotting=False,
        groundcrew_count=1,
        groundcrew_speed=30,
        elapsed_minutes=240,
        groundcrew_sector_mapping=[0],
        second_groundcrew_sector_mapping=None,
        wind_schedule=None,               # will be replaced by the wrapper
        fuel_model_override=None
    )

    simulation_loop(model)


if __name__ == "__main__":
    main()
