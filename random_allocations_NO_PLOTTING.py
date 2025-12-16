#!/usr/bin/env python
"""
dashboard_random_headless.py – Headless random allocation baseline

This version replaces MCTS with a uniform‑random allocation baseline.
All Panel/Plotly UI, widgets, and layout updates have been removed so that the
simulation runs headlessly from the command line. For each decision slice, every asset
is assigned to a uniformly random open sector.

Launch with:  python dashboard_random_headless.py
"""

import time as t
import numpy as np
import datetime

# Import everything from the original dashboard.
# Because the original file guards the pn.serve() call under __main__,
# importing it here will NOT start a second server.
import dashboard as dash
from mcts import simulate_in_place, ordinal_map  # helper utilities only


def random_allocation(model, open_secs):
    """
    Return an action dict that assigns every asset to a random open sector.
    Sectors are converted to 1-based indices as expected by simulate_in_place().
    """
    rng = np.random.default_rng()
    actions = {}
    for atype, uid_list in ordinal_map(model).items():
        if not uid_list:
            continue  # No assets of this type
        actions[atype] = tuple(int(rng.choice(open_secs) + 1) for _ in uid_list)
    return actions


def simulation_loop_random(model):
    """
    Headless simulation loop using a uniform random allocation baseline.
    At each decision boundary (every dash.decision_interval minutes), a random action
    is computed for each asset and applied to the simulation.
    """
    schedule_enabled = (model.wind_schedule is not None)
    next_decision_time = model.time
    overall_limit = model.overall_time_limit

    # If forecasts are enabled, pull one right away so the simulation has some
    # forecast data (for debugging/logging purposes).
    if schedule_enabled:
        forecast_df = dash.get_forecast(current_minute=int(model.time))
        model.latest_forecast_df = forecast_df
        print(f"[SIM] Initial forecast updated at t = {model.time} min")

    while model.time < overall_limit:
        if model.time >= overall_limit - dash.decision_interval:
            print("[SIM] < 1 decision slice left – terminating simulation.")
            dash.end_simulation(model)
            break

        if model.time >= next_decision_time:
            # open_secs = [s for s in range(4) if not model.is_sector_contained(s)]

            # NEW
            num_sectors = getattr(model, "num_sectors",
                                  len(getattr(model, "sector_angle_ranges", [])) or 4)
            open_secs = [s for s in range(num_sectors) if not model.is_sector_contained(s)]

            if not open_secs:
                print("[SIM] All sectors contained – ending simulation.")
                dash.end_simulation(model)
                break

            # Compute a random allocation action for the current open sectors.
            action = random_allocation(model, open_secs)
            simulate_in_place(model, action, duration=dash.decision_interval)
            print(f"[SIM] At t = {model.time:.0f} min, applied random action: {action}")
            next_decision_time += dash.decision_interval
            continue  # Skip the ordinary one‑minute tick this loop

        # Advance one time step.
        if not model.step():
            break

    print("Simulation complete.")


def main():
    print("Starting random allocation simulation (headless mode)...")
    # Build the model using default parameters (adjust as needed).
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
        enable_plotting=False,  # Disable plotting in headless mode.
        groundcrew_count=0,
        groundcrew_speed=30,
        elapsed_minutes=240,
        groundcrew_sector_mapping=[0],
        second_groundcrew_sector_mapping=None,
        wind_schedule=dash.load_wind_schedule_from_csv_random("wind_schedule_natural.csv"),
        fuel_model_override=None
    )
    simulation_loop_random(model)


if __name__ == "__main__":
    main()