#!/usr/bin/env python
"""
ics_mean_burned_buildings_NO_PLOTTING.py
──────────────────────────────────────────────────────────────────────────────
Strategy:
  • Precompute ONCE the number of buildings that would burn by the final time
    in each sector under the **MEAN** wind schedule (no suppression).
  • Ground crews: deterministic priority by that per-sector count (DESC).
  • Aircraft: weighted draws where P(sector) ∝ that per-sector count among open
    sectors (uniform fallback if all zero).

Strict requirement:
  • If the MEAN (background) schedule is not registered, the script raises a
    RuntimeError and aborts. There is NO fallback to truth or CSV.

Notes:
  • The simulation itself still runs under the TRUTH schedule (the batch wrapper
    patches WildfireModel.__init__). We only *derive* priorities/weights from
    the MEAN schedule, once up-front.
"""

import numpy as np
import datetime
import cupy as cp
import sys

import dashboard as dash                          # WildfireModel comes from model.py
from mcts import simulate_in_place, ordinal_map   # apply actions & list assets

from FIRE_MODEL_CUDA import SurrogateFireModelROS

DECISION_INTERVAL = 120  # minutes


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def _draw_for_assets(rng, open_set, weights_dict, n):
    """
    Draw n sectors (without replacement when possible) according to weights in
    weights_dict[s] (s are 0-based sector indices). Return 1-based sector ids.
    """
    sectors = np.array(sorted(open_set))
    w = np.array([max(float(weights_dict.get(s, 0.0)), 0.0) for s in sectors], dtype=float)
    if w.sum() <= 0:
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


def _tif_path_from_model(model) -> str:
    """Best effort to get the landscape tif path from the live model."""
    try:
        return str(model.fire.tif_path)
    except Exception:
        # keep the same default used elsewhere in your batch baseline
        return "camp_fire_three_enhanced.tif"


def _must_get_mean_schedule(model) -> list[tuple]:
    import os, json
    import wind_schedule_utils as _wsu
    """
    STRICT: return the MEAN (background) schedule.
    Order:
      1) model._background_schedule (injected by wrapper)
      2) forecast_provider (if registered)
      3) env var BACKGROUND_SCHEDULE_JSON (JSON-encoded list of tuples)
      4) env var BACKGROUND_SCHEDULE_PATH → load_wind_schedule_from_csv_mean
    If none found → raise RuntimeError. Never fall back to truth.
    """
    # 1) from the live model (most reliable, avoids import-order issues)
    sched = getattr(model, "_background_schedule", None)
    if sched:
        return sched

    # 2) registered provider (if available later)
    try:
        import forecast_provider as fp
        if hasattr(fp, "get_background_schedule"):
            s = fp.get_background_schedule()
            if s:
                return s
        for attr in ("BACKGROUND_SCHEDULE", "_BACKGROUND_SCHEDULE", "background_schedule", "BACKGROUND_SCHED"):
            if hasattr(fp, attr):
                s = getattr(fp, attr)
                if s:
                    return s
    except Exception:
        pass

    # 3) environment JSON (set by wrapper)
    env_json = os.environ.get("BACKGROUND_SCHEDULE_JSON")
    if env_json:
        try:
            s = json.loads(env_json)
            if s:
                return s
        except Exception:
            pass

    # 4) environment path → read means from CSV
    env_path = os.environ.get("BACKGROUND_SCHEDULE_PATH")
    if env_path:
        s = _wsu.load_wind_schedule_from_csv_mean(env_path)
        if s:
            return s

    # Strict failure
    raise RuntimeError(
        "MEAN schedule unavailable. Looked on model._background_schedule, "
        "forecast_provider, BACKGROUND_SCHEDULE_JSON, BACKGROUND_SCHEDULE_PATH."
    )

# ───────────────────────────────────────────────────────────────
# Precompute: per-sector buildings burned by final time under MEAN schedule
# ───────────────────────────────────────────────────────────────
def _compute_burned_buildings_per_sector_mean(model) -> list[int]:
    """
    Build a surrogate model using the **mean** schedule (no suppression),
    then count how many *buildings* (built_char ≥ 11) have arrival_time ≤ final
    time inside each sector wedge.
    """
    # sector geometry from the live model
    model.update_sector_splits()
    sector_ranges: list[tuple[float, float]] = model.sector_angle_ranges
    num_sectors = len(sector_ranges) or getattr(model, "num_sectors", 4)

    mean_sched = _must_get_mean_schedule(model)  # ← STRICT: raises if missing
    print(mean_sched)
    tif_path = _tif_path_from_model(model)
    final_time = float(model.fire_spread_sim_time)

    # dummy wind only used if wind_schedule is None
    dummy_speed = float(mean_sched[-1][2])
    dummy_dir = float(mean_sched[-1][3])

    fm_mean = SurrogateFireModelROS(
        tif_path=tif_path,
        sim_time=int(final_time),
        wind_speed=dummy_speed,
        wind_direction_deg=dummy_dir,
        max_iter=250,
        tol=1e-3,
        wind_schedule=mean_sched
    )
    cp.cuda.Stream.null.synchronize()

    T_np = cp.asnumpy(fm_mean.T)
    built = fm_mean.built_char_np
    if built is None:
        print("[PRECOMPUTE] WARNING: No building band present; all sector weights = 0")
        return [0]*num_sectors

    burned_buildings = np.isfinite(T_np) & (T_np <= final_time) & (built >= 11)
    if not burned_buildings.any():
        print("[PRECOMPUTE] Under mean schedule, no buildings burn by final time.")
        return [0]*num_sectors

    # angles for burned building cells
    rows_idx, cols_idx = np.where(burned_buildings)
    a = fm_mean.transform[0]; c = fm_mean.transform[2]
    e = fm_mean.transform[4]; f = fm_mean.transform[5]
    xs = c + (cols_idx + 0.5) * a
    ys = f + (rows_idx + 0.5) * e
    ign_x, ign_y = fm_mean.ignition_pt
    angles = (np.degrees(np.arctan2(ys - ign_y, xs - ign_x)) + 360.0) % 360.0

    # bin into the model’s sector wedges
    counts = [0] * num_sectors
    for s, (lo, hi) in enumerate(sector_ranges):
        counts[s] = int(np.sum((angles >= lo) & (angles < hi)))

    total = int(np.sum(counts))
    print(f"[PRECOMPUTE] MEAN-schedule burned buildings per sector (final t={final_time:.0f}): "
          f"{counts}  | total={total}")
    return counts


# ───────────────────────────────────────────────────────────────
# Allocation policy
# ───────────────────────────────────────────────────────────────
def mean_burned_buildings_allocation(model, open_secs, burned_counts):
    """
    Ground crews:
        deterministic priority = sectors sorted by burned_counts DESC.
    Aircraft:
        weighted random draw where P(sector) ∝ burned_counts[sector]
        (restricted to open sectors; uniform if all weights are zero).
    """
    open_set = set(open_secs)
    rng = np.random.default_rng()

    order = sorted(open_set, key=lambda s: (burned_counts[s], -s), reverse=True)
    weights = {s: float(burned_counts[s]) for s in open_set}
    if sum(weights.values()) <= 0:
        eq = 1.0 / max(len(open_set), 1)
        weights = {s: eq for s in open_set}

    actions = {}
    for atype, uid_list in ordinal_map(model).items():
        if not uid_list:
            continue

        if atype == "GroundCrewAgent":
            chosen = []
            if order:
                idx = 0
                for _ in uid_list:
                    for k in range(len(order)):
                        s = order[(idx + k) % len(order)]
                        if s in open_set:
                            chosen.append(s + 1)   # 1-based
                            idx = (idx + k + 1) % len(order)
                            break
                    else:
                        chosen.append(int(rng.choice(list(open_set))) + 1)
            else:
                chosen = [1] * len(uid_list)
            actions[atype] = tuple(chosen)
        else:
            picks = _draw_for_assets(rng, open_set, weights, len(uid_list))
            actions[atype] = tuple(picks)

    if open_set:
        ws = {s: round(float(weights.get(s, 0.0)), 3) for s in sorted(open_set)}
        print(f"[ICS] open={sorted(open_set)}  crew_order={order}  weights={ws}")
    return actions


# ───────────────────────────────────────────────────────────────
# End-of-simulation helper (matches the batch parser’s regex)
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
      • choose actions using the fixed MEAN-schedule burned-building ranking
      • apply via simulate_in_place()
    """
    next_decision_time = model.time
    overall_limit = model.overall_time_limit
    # PRECOMPUTE once (mean schedule) — raises if missing
    try:
        burned_counts = _compute_burned_buildings_per_sector_mean(model)
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        sys.exit(2)

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

            action = mean_burned_buildings_allocation(model, open_secs, burned_counts)
            simulate_in_place(model, action, duration=DECISION_INTERVAL)
            print(f"Applied action {action} at t={model.time} min")
            next_decision_time += DECISION_INTERVAL
            continue
        else:
            if not model.step():
                break

    print("Simulation complete.")


# ───────────────────────────────────────────────────────────────
# Main (wrapped by run_all_strategies_batch.py)
# ───────────────────────────────────────────────────────────────
def main():
    print("Starting STRICT-MEAN-burned-buildings ICS simulation (headless)…")

    # The batch wrapper patches WildfireModel.__init__:
    #   • wind_schedule ← TRUTH
    #   • case_folder   ← run output dir
    #   • num_sectors   ← CLI --sectors
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
        wind_schedule=None,  # will be replaced by wrapper (TRUTH)
        fuel_model_override=None
    )

    simulation_loop(model)


if __name__ == "__main__":
    main()
