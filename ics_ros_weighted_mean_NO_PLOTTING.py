"""
ics_ros_weighted_mean_NO_PLOTTING.py
──────────────────────────────────────────────────────────────────────────────
ROS-weighted “greedy” baseline that **computes its ROS weights from the forecast
*means* (µ) schedule**, while the fire itself still evolves under the sampled
“truth” schedule supplied when the model is constructed.

Key points
──────────
•  At every decision slice we ask the dashboard for the latest forecast table
   (start_min, end_min, speed_mean, speed_std, dir_mean, dir_std …).

•  We convert that table to a *pure-mean* schedule and temporarily inject it
   into the Fire object **only for the ROS query**.  The simulation state is
   restored immediately afterwards, so physics never change.

•  Airtankers receive probability weights ∝ ROS; ground-crews get a deterministic
   ranking by descending ROS.


"""

# ───────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────
import numpy as np
import datetime
import time as t

import dashboard as dash
import ics_dynamic as dyn                 # re-use helpers
from mcts import ordinal_map
from surrogate_plotting import plot_fire  # only used for optional snapshots
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────
DECISION_INTERVAL = 120       # minutes between allocation rounds

# Draw-helper reused from the dynamic ICS module.
_draw_for_assets = dyn._draw_for_assets

# ───────────────────────────────────────────────────────────────
# Helper – build a “mean” schedule list from the forecast DataFrame
# ───────────────────────────────────────────────────────────────
def _mean_schedule_from_forecast(df):
    """[(start, end, speed_mean, dir_mean)]   or   None if df is empty."""
    if df is None or df.empty:
        return None
    return [
        (int(r.start_min), int(r.end_min),
         float(r.speed_mean), float(r.dir_mean) % 360)
        for r in df.itertuples()
    ]

# ───────────────────────────────────────────────────────────────
# Helper – temporarily evaluate ROS under an *alternate* schedule
# ───────────────────────────────────────────────────────────────
def _sector_ros(model, open_set,
                *, use_schedule=None,
                tol_past=5, tol_fut=120, cell_ft=98.4) -> dict[int,float]:
    """
    Aggregate perimeter ROS (ft / min) for each open sector at `model.time`.

    If `use_schedule` is supplied (a list of tuples) it is patched into both
    `model` and `model.fire` *just for the duration of the query*.
    """
    # 1) Patch schedule if requested
    if use_schedule is not None:
        _old_mod_sched  = model.wind_schedule
        _old_fire_sched = model.fire.wind_schedule
        model.wind_schedule       = use_schedule
        model.fire.wind_schedule  = use_schedule

    # 2) Compute ROS
    ros = {}
    now = model.time
    for s in open_set:
        lo, hi = model.sector_angle_ranges[s]
        ar = (lo, hi)
        val = model.fire.aggregate_perimeter_ros_by_increments(
            [ar], statistic="mean99",
            time_threshold=now,
            tolerance_past=tol_past, tolerance_future=tol_fut,
            cell_size_ft=cell_ft)[ar]
        ros[s] = max(float(val), 0.0) if not np.isnan(val) else 0.0

    # 3) Restore the original schedule
    if use_schedule is not None:
        model.wind_schedule       = _old_mod_sched
        model.fire.wind_schedule  = _old_fire_sched

    return ros

# ───────────────────────────────────────────────────────────────
# Helper – normalise ROS → probability weights
# ───────────────────────────────────────────────────────────────
def _weights_from_ros(ros: dict[int,float]) -> dict[int,float]:
    total = sum(ros.values())
    if total <= 0.0:
        n = len(ros)
        return {s: 1.0 / n for s in ros}
    return {s: v / total for s, v in ros.items()}

# ───────────────────────────────────────────────────────────────
# Allocation function – **uses mean-schedule ROS**
# ───────────────────────────────────────────────────────────────
def ros_weighted_allocation(model, open_secs):
    """
    • Ground-crews: deterministic order by descending ROS (mean schedule).
    • Airtankers : random draw  p(s) ∝ ROS_s  (mean schedule).
    """
    open_set = set(open_secs)
    rng = np.random.default_rng()

    # 1) Build the mean schedule from the latest forecast
    mean_sched = _mean_schedule_from_forecast(
        getattr(model, "latest_forecast_df", None)
    )
    print(mean_sched)

    # 2) ROS under that *mean* schedule
    ros = _sector_ros(model, open_set, use_schedule=mean_sched)

    # 3) Convert to weights / crew ranking
    weights   = _weights_from_ros(ros)
    crew_pref = sorted(open_set, key=lambda s: ros[s], reverse=True)

    # ---- DEBUG PRINT ---------------------------------------------------
    print(f"\n[ROS-MEAN] t={model.time:.0f} min  (forecast-mean schedule)")
    for s in sorted(open_set):
        print(f"  sector {s}: ROS={ros[s]:7.1f}  → weight={weights[s]:.3f}")
    print("  crew order  →", crew_pref)
    # -------------------------------------------------------------------

    # 4) Build the action dict
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
                        chosen.append(sec + 1)      # to 1-based
                        idx = (idx + k + 1) % len(crew_pref)
                        break
                else:  # fallback random
                    chosen.append(rng.choice(list(open_set)) + 1)
            actions[atype] = tuple(chosen)
        else:        # airtankers
            actions[atype] = tuple(
                _draw_for_assets(rng, open_set, weights, len(uid_list))
            )
    return actions

# ───────────────────────────────────────────────────────────────
# End-simulation helper
# ───────────────────────────────────────────────────────────────
def _end_sim(model):
    """Flush GC buffers and print final score (no plotting)."""
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
# Main simulation loop (stand-alone, no dependencies on dyn)
# ───────────────────────────────────────────────────────────────
def simulation_loop(model):
    """
    Decision slices every DECISION_INTERVAL minutes.
    Each slice:
      – refresh forecast → model.latest_forecast_df
      – allocate assets with ros_weighted_allocation()
      – simulate in place for the slice duration
    Terminates when all sectors are contained or overall limit reached.
    """
    from mcts import simulate_in_place

    next_dec = model.time
    limit    = model.overall_time_limit

    # initial forecast
    model.latest_forecast_df = dash.get_forecast(current_minute=int(model.time))

    while model.time < limit:
        if model.time >= limit - DECISION_INTERVAL:
            print("[SIM] Less than one full slice left – stopping.")
            _end_sim(model)
            break

        if model.time >= next_dec:
            # refresh forecast means
            model.latest_forecast_df = dash.get_forecast(
                current_minute=int(model.time)
            )

            # open_secs = [s for s in range(4) if not model.is_sector_contained(s)]
            num_sectors = getattr(model, "num_sectors",
                                  len(getattr(model, "sector_angle_ranges", [])) or 4)
            open_secs = [s for s in range(num_sectors) if not model.is_sector_contained(s)]
            if not open_secs:
                print("[SIM] All sectors contained – done.")
                _end_sim(model)
                break

            action = ros_weighted_allocation(model, open_secs)
            simulate_in_place(model, action, duration=DECISION_INTERVAL)
            print(f"Applied action {action} at t={model.time:.0f} min")
            next_dec += DECISION_INTERVAL
            continue

        # Otherwise just step the model
        if not model.step():
            break

    print("Simulation complete.")

# ───────────────────────────────────────────────────────────────
# Main entry point – build model & run
# ───────────────────────────────────────────────────────────────
def main():
    print("Starting ROS-weighted-MEAN simulation (headless)…")

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
        case_folder="ROS_Mean_Case",
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

    # OPTIONAL – save initial & final PNGs
    def _snap(tag):
        fig, ax = plot_fire(model.fire, time=model.time,
                            max_time=model.fire_spread_sim_time)
        ax.set_title(f"ROS-MEAN – {tag}  (t={model.time:.0f} min)")
        fig.savefig(f"ros_mean_{tag}.png", dpi=300)
        plt.close(fig)
    _snap("initial")

    simulation_loop(model)

    _snap("final")

if __name__ == "__main__":
    main()
