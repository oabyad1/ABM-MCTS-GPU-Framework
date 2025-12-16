"""
DOE simulation script with no plotting enabled

This version removes all Panel/Plotly user-interface and plotting code,
but keeps the original simulation/MCTS/forecast behavior intact.
It is meant for running design-of‐experiments studies.
"""

import datetime
import math
import time as t
import numpy as np
import pandas as pd
import networkx as nx
import cupy as cp
import os

from forecast_provider import get_forecast
# from wind_schedule_utils import load_wind_schedule_from_csv
from model import WildfireModel
from mcts import clone_simulation, mcts, simulate_in_place, hierarchy_pos_tree, count_expandable_assets

# ───────────────────────────────────────────────────────────────
# Debug/Helper Functions
# ───────────────────────────────────────────────────────────────
def _print_schedule_tuples(schedule: list[tuple], *, title=""):
    """
    Pretty-print a list of (start, end, speed, dir) tuples.
    """
    if title:
        print(f"\n{title}")
    df = pd.DataFrame(schedule, columns=["start", "end", "speed", "dir"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

def _debug_dump_schedule(model, *, header=""):
    """
    Pretty-print the entire wind schedule of the model.
    """
    cols = ["start", "end", "speed", "direction"]
    df = pd.DataFrame(model.wind_schedule, columns=cols)
    print(f"\n{header}  —  t = {model.time:.0f} min")
    print(df.to_string(index=False))

def _debug_dump_forecast(fore_df: pd.DataFrame, *, now: int, hdr=""):
    """
    Pretty-print forecast μ/σ info (for every future segment) before sampling.
    """
    if hdr:
        print(f"\n{hdr}  —  t = {now} min")
    show = fore_df.copy()
    show["lead"] = show.start_min - now
    cols = ["lead", "start_min", "end_min", "speed_mean", "speed_std", "dir_mean", "dir_std"]
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

def _make_baseline_schedule(truth_model, forecast_df):
    """
    Return full 0-to-end schedule made of past records, plus future means.
    """
    past = [seg for seg in truth_model.wind_schedule if seg[1] <= truth_model.time]
    future_mu = [
        (int(r.start_min), int(r.end_min), float(r.speed_mean), float(r.dir_mean))
        for r in forecast_df.itertuples() if r.end_min > truth_model.time
    ]
    return past + future_mu

def count_nodes(node) -> int:
    """
    Recursively count the number of descendants (including the root node).
    """
    return 1 + sum(count_nodes(c) for c in getattr(node, "children", []))

# ───────────────────────────────────────────────────────────────
# Global simulation parameters (defaults)
# ───────────────────────────────────────────────────────────────
lake_positions = [(5000, 5000)]
base_positions = [(20000, 20000)]
start_time = datetime.datetime.strptime("00:00", "%H:%M")
case_folder = "DOE_Case"
groundcrew_sector_mapping = [0]
second_groundcrew_sector_mapping = None

# MCTS and simulation configuration
decision_interval   = 120   # minutes between replanning
mcts_iterations     = 50
mcts_max_depth      = 3
exploration_constant= 1 / math.sqrt(2)
building_weight     = 2

auto_iterations       = True     # turn on to mirror dashboard's auto-iterations
iters_for_1_asset     = 50
iters_for_2_assets    = 100
iters_for_3_assets    = 300


# Simulation and fire parameters
elapsed_minutes     = 240
fire_sim_time       = 2000
overall_time_limit  = 2000
groundcrew_count    = 0

# Airtanker counts (using one type for brevity)
c130j_count         = 0
fireherc_count      = 1
scooper_count       = 0
at802f_count        = 0
dash8_count         = 0

time_step           = 1
operational_delay   = 0
groundcrew_speed    = 30
wind_speed          = 0
wind_dir            = 220

#Commented out so that it doesnt overwrite the schedule set by the batch script

# # Load the wind schedule (from CSV)
# from forecast_provider import (
#     get_forecast,
#     set_background_schedule,
#     set_truth_schedule,
# )
# from wind_schedule_utils import (
#     load_wind_schedule_from_csv_mean,
#     load_wind_schedule_from_csv_random,   # or _sigma if you prefer
# )
#
# # --- load background (means) ------------------------------------
# _BACKGROUND_SCHED = load_wind_schedule_from_csv_mean("wind_schedule_natural.csv")
#
# # --- load truth (random draw for standalone runs) ---------------
# TRUTH_SEED = 3
# truth_schedule = load_wind_schedule_from_csv_random(
#     "wind_schedule_natural.csv",
#     seed=TRUTH_SEED)
#
# # register with forecast_provider
# set_background_schedule(_BACKGROUND_SCHED)
# set_truth_schedule(truth_schedule)

truth_schedule = None


# ───────────────────────────────────────────────────────────────
# Model construction
# ───────────────────────────────────────────────────────────────
def build_model() -> WildfireModel:
    """
    Construct a fresh WildfireModel using the preset parameter values.
    The enable_plotting flag is set to False so that all plotting-related
    functions inside the model are disabled (for DOE experiments).
    """
    model = WildfireModel(
        airtanker_counts={
            "C130J":          c130j_count,
            "FireHerc":       fireherc_count,
            "Scooper":        scooper_count,
            "AT802F":         at802f_count,
            "Dash8_400MRE":   dash8_count,
        },
        wind_speed           = wind_speed,
        wind_direction       = wind_dir,
        base_positions       = base_positions,
        lake_positions       = lake_positions,
        time_step            = time_step,
        debug                = False,
        start_time           = start_time,
        case_folder          = case_folder,
        overall_time_limit   = overall_time_limit,
        fire_spread_sim_time = fire_sim_time,
        operational_delay    = operational_delay,
        enable_plotting      = False,  # disable all plotting/panel features
        groundcrew_count     = groundcrew_count,
        groundcrew_speed     = groundcrew_speed,
        elapsed_minutes      = elapsed_minutes,
        groundcrew_sector_mapping        = groundcrew_sector_mapping,
        second_groundcrew_sector_mapping = second_groundcrew_sector_mapping,
        wind_schedule        = truth_schedule,
        fuel_model_override  = None
    )
    return model

# ───────────────────────────────────────────────────────────────
# End-of-simulation helper
# ───────────────────────────────────────────────────────────────
def end_simulation(model: WildfireModel):
    """
    Finalize the run: flush any pending ground-crew work, calculate and print
    the final fire score. (Plotting is only performed if enabled.)
    """
    from groundcrew import GroundCrewAgent
    for ag in model.schedule.agents:
        if isinstance(ag, GroundCrewAgent):
            ag.flush_clear_buffer()



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




    # score = model.fire.calculate_fire_score(model.time)
    # print(f"[SIM] Finished at t={model.time:.0f} min – final fire-score = {score:.2f}")

    if model.enable_plotting:
        model.plot_fire()
        from pathlib import Path
        out = Path(model.case_folder) / "images"
        out.mkdir(exist_ok=True)
        fname = out / f"final_{int(model.time)}.png"
        model.plot_fig.write_image(str(fname), scale=4)
        print(f"[SIM] Saved final frame → {fname}")

# ───────────────────────────────────────────────────────────────
# Simulation Loop (headless version)
# ───────────────────────────────────────────────────────────────
def simulation_loop(model: WildfireModel):
    """
    Core simulation loop that drives the wildfire simulation and invokes MCTS
    to choose actions. All UI/panel updates have been removed so that the code
    is suitable for design-of-experiments runs.
    """
    schedule_enabled = model.wind_schedule is not None
    next_decision_time = model.time
    overall_limit = model.overall_time_limit

    # Initial forecast update (if enabled)
    if schedule_enabled:
        forecast_df = get_forecast(current_minute=int(model.time))
        model.latest_forecast_df = forecast_df
        _debug_dump_forecast(forecast_df, now=int(model.time), hdr="Initial Forecast")
    else:
        forecast_df = None

    while model.time < overall_limit:
        # If there is less than one full decision slice left, terminate.
        if model.time >= overall_limit - decision_interval:
            print("[SIM] Less than one full decision slice left – terminating simulation.")
            end_simulation(model)
            break

        # At decision boundaries, run MCTS for a new action.
        if model.time >= next_decision_time:
            if schedule_enabled:
                forecast_df = get_forecast(current_minute=int(model.time))
                model.latest_forecast_df = forecast_df
                _debug_dump_forecast(forecast_df, now=int(model.time), hdr="Forecast before MCTS")
                _debug_dump_schedule(model, header="Schedule before MCTS")

            # open_sectors = [s for s in range(4) if not model.is_sector_contained(s)]

            # derive sector count from the model
            num_sectors = getattr(model, "num_sectors", len(getattr(model, "sector_angle_ranges", [])) or 4)
            open_sectors = [s for s in range(num_sectors) if not model.is_sector_contained(s)]
            if not open_sectors:
                print("[SIM] All sectors contained – ending simulation.")
                end_simulation(model)
                break

            clone = clone_simulation(model)
            if schedule_enabled:
                clone.latest_forecast_df = forecast_df

            cp.cuda.Stream.null.synchronize()
            t0 = t.time()
            # Run MCTS without a UI update callback
            # root, _ = mcts(
            #     clone,
            #     iterations=mcts_iterations,
            #     max_depth=mcts_max_depth,
            #     duration=decision_interval,
            #     exploration_constant=exploration_constant,
            #     building_weight=building_weight,
            #     track_progress=False,
            #     on_iteration=None
            # )
            # ── decide how many iterations to run ────────────────────────
            if auto_iterations:
                n_assets = count_expandable_assets(model)
                mapping = {
                    1: iters_for_1_asset,
                    2: iters_for_2_assets,
                    3: iters_for_3_assets,
                }
                mcts_iterations_effective = mapping.get(n_assets, mcts_iterations)
            else:
                mcts_iterations_effective = mcts_iterations
            ROLLOUT_DEPTH_ADJUSTMENT = int(os.environ["ROLLOUT_DEPTH_ADJUSTMENT"])

            root, _ = mcts(
                clone,
                iterations=mcts_iterations_effective,
                max_depth=mcts_max_depth,
                duration=decision_interval,
                exploration_constant=exploration_constant,
                building_weight=building_weight,
                rollout_depth_adjustment=ROLLOUT_DEPTH_ADJUSTMENT,
                track_progress=False,
                on_iteration=None,
            )

            cp.cuda.Stream.null.synchronize()
            mcts_duration = t.time() - t0

            if not root.children:
                print("[SIM] MCTS produced no children – skipping ahead.")
                next_decision_time += decision_interval
                continue

            total_nodes = count_nodes(root)
            best_child = max(root.children, key=lambda c: (c.reward / c.visits) if c.visits else -float('inf'))
            avg_best = (best_child.reward / best_child.visits) if best_child.visits else 0.0
            rewards = [(c.reward / c.visits) if c.visits else 0 for c in root.children]

            print(f"\n--- Decision at t = {model.time:.0f} min ---")
            print(f"Total tree nodes: {total_nodes}")
            print(f"Iterations: {mcts_iterations_effective}, Max Depth: {mcts_max_depth}")
            print(f"Chosen Action: {best_child.node_name}")
            print(f"Average Reward: {avg_best:.2f}, Best: {max(rewards):.2f}, Worst: {min(rewards):.2f}")
            print(f"MCTS Computation Time: {mcts_duration:.2f} s")

            # Apply the chosen MCTS action to the simulation.
            simulate_in_place(model, best_child.action, duration=decision_interval)
            next_decision_time += decision_interval
        else:
            # No decision due yet: advance simulation one time step.
            if not model.step():
                break

    print("[SIM] Simulation complete.")

# ───────────────────────────────────────────────────────────────
# Main entry point
# ───────────────────────────────────────────────────────────────
def main():
    # Build the model using the preset parameters.
    model = build_model()
    print("Starting simulation with the following parameters:")
    print(f"  Overall Time Limit: {model.overall_time_limit} min")
    print(f"  Decision Interval:  {decision_interval} min")

    # Compute a baseline fire score (if wind schedule/forecast is enabled).
    if model.wind_schedule is not None:
        forecast_df = get_forecast(current_minute=0)
        baseline_sched = _make_baseline_schedule(model, forecast_df)
        _print_schedule_tuples(baseline_sched, title="Baseline Schedule")
        baseline_model = clone_simulation(model, new_wind_schedule=baseline_sched)
        model.baseline_fire_score = baseline_model.fire.calculate_fire_score(
            baseline_model.fire_spread_sim_time
        )
        print("Baseline FIRE SCORE (forecast-based):", model.baseline_fire_score)
    else:
        model.baseline_fire_score = model.fire.calculate_fire_score(model.fire_spread_sim_time)
        print("Baseline FIRE SCORE (truth-based):", model.baseline_fire_score)

    simulation_loop(model)

if __name__ == "__main__":
    main()