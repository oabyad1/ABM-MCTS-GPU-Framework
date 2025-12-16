#!/usr/bin/env python
"""
ics_ros_weighted_headless.py – ROS-weighted “greedy” baseline (headless version)

• Airtankers are allocated proportionally to the current perimeter ROS of every still‐open sector:
      weight_s = ROS_s / (Σ ROS_open)
• Ground-crews are assigned deterministically in descending order of ROS.
• All UI/dashboard functionality has been removed so that the simulation runs from the command line.

Launch with:  python ics_ros_weighted_headless.py
"""

import numpy as np
import datetime
import time as t

import dashboard as dash
import ics_dynamic as dyn  # Reused helper code from the dynamic ICS module
from mcts import ordinal_map

# Reuse the helper from dyn: draw N unique sectors by probability.
_draw_for_assets = dyn._draw_for_assets


# ───────────────────────────────────────────────────────────────
# 1) Per-sector aggregated perimeter ROS (ft/min)
# ───────────────────────────────────────────────────────────────
def _sector_ros(model, open_set, *, tol_past=5, tol_fut=120, cell_ft=98.4) -> dict:
    """
    Return a dictionary mapping sector indices to aggregated perimeter ROS
    (ft/min) evaluated at the current simulation minute (model.time). Any NaNs or
    non-positive values are set to zero.
    """
    ros = {}
    time_now = model.time
    for s in open_set:
        lo, hi = model.sector_angle_ranges[s]  # degrees
        ar = (lo, hi)
        ros_val = model.fire.aggregate_perimeter_ros_by_increments(
            [ar],
            statistic='mean99',
            time_threshold=time_now,
            tolerance_past=tol_past,
            tolerance_future=tol_fut,
            cell_size_ft=cell_ft
        )[ar]
        if np.isnan(ros_val) or ros_val <= 0.0:
            ros_val = 0.0
        ros[s] = float(ros_val)
    return ros


def _weights_from_ros(ros: dict) -> dict:
    """
    Normalize the raw ROS values to produce probability weights.
    If the total ROS is 0, return uniform weights.
    """
    tot = sum(ros.values())
    if tot <= 0.0:
        n = len(ros)
        return {s: 1.0 / n for s in ros}
    return {s: v / tot for s, v in ros.items()}


# ───────────────────────────────────────────────────────────────
# 2) ROS-weighted allocation function
# ───────────────────────────────────────────────────────────────
def ros_weighted_allocation(model, open_secs):
    """
    Determine asset allocation based on perimeter ROS:
      • Ground-crews: deterministic allocation in descending order of ROS.
      • Airtankers: weighted random draw with probability ∝ ROS.
    Returns an action dictionary mapping asset types to a tuple of 1-based sector indices.
    """

    open_set = set(open_secs)
    rng = np.random.default_rng()

    # 1) Compute raw perimeter ROS for every still-open sector.
    ros = _sector_ros(model, open_set)

    # 2) Compute probability weights for airtankers.
    weights = _weights_from_ros(ros)

    # 3) Determine a deterministic order for ground-crews (descending ROS).
    crew_pref = sorted(open_set, key=lambda s: ros[s], reverse=True)

    # ---------- DEBUG DUMP ---------------------------------------
    print(f"\n[ROS-ALLOC]  t = {model.time:.0f} min")
    for s in sorted(open_set):
        print(f"  • sector {s}:  ROS = {ros[s]:7.1f}  ->  weight = {weights[s]:.3f}")
    print("  Ground-crew ranking  =>", crew_pref)
    # -------------------------------------------------------------

    # 4) Build the action dictionary.
    actions = {}
    for atype, uid_list in ordinal_map(model).items():
        if not uid_list:
            continue

        # Ground-crews: deterministic selection.
        if atype == "GroundCrewAgent":
            chosen, pref_idx = [], 0
            for _ in uid_list:
                for k in range(len(crew_pref)):
                    sec = crew_pref[(pref_idx + k) % len(crew_pref)]
                    if sec in open_set:
                        chosen.append(sec + 1)  # Convert to 1-based index.
                        pref_idx = (pref_idx + k + 1) % len(crew_pref)
                        break
                else:
                    sec = rng.choice(list(open_set))
                    chosen.append(sec + 1)
            actions[atype] = tuple(chosen)
        # Airtankers: weighted random draw.
        else:
            picks = _draw_for_assets(rng, open_set, weights, len(uid_list))
            actions[atype] = tuple(picks)
    return actions


# ───────────────────────────────────────────────────────────────
# 3) Plug the new allocator into the existing simulation loop.
# ───────────────────────────────────────────────────────────────
# Monkey-patch the dynamic allocation strategy in the dynamic ICS module.
dyn.dynamic_ics_allocation = ros_weighted_allocation


def simulation_loop_ros(model):
    """
    Wrapper that invokes the original simulation_loop_ics from the dynamic ICS module
    using our ROS-weighted allocation strategy.
    """
    return dyn.simulation_loop_ics(model)


# Optionally, override the simulation_loop in the dashboard module.
dash.simulation_loop = simulation_loop_ros


# ───────────────────────────────────────────────────────────────
# 4) Main entry point – Headless simulation run from the command line.
# ───────────────────────────────────────────────────────────────
def main():
    print("Starting ROS-weighted simulation (headless mode)...")

    # IMPORTANT: Clear pause and stop events to ensure the simulation loop runs.
    dash.pause_event.clear()
    dash.simulation_stop_event.clear()

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
        enable_plotting=False,  # disable plotting for headless mode
        groundcrew_count=0,
        groundcrew_speed=30,
        elapsed_minutes=240,
        groundcrew_sector_mapping=[0],
        second_groundcrew_sector_mapping=None,
        wind_schedule=dash.load_wind_schedule_from_csv_random("wind_schedule_natural.csv"),
        fuel_model_override=None
    )

    simulation_loop_ros(model)


if __name__ == "__main__":
    main()