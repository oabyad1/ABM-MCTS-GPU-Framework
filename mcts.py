"""
mcts.py

This script implements a Monte Carlo Tree Search (MCTS) for a wildfire simulation.
The objective is to minimize the fire score (area burned), so the reward is defined as -fire_score.
Additionally, it records detailed timing stats (expansion time, rollout time, node execution time, and iteration)
which are exported in a JSON file.
"""

import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import random
import math
import networkx as nx  # For plotting the MCTS tree structure
import cProfile
import pstats
from model import WildfireModel
from surrogate_plotting import plot_fire  # plot_fire(surrogate, ...) function
import json
import time
from wind_schedule_utils import sample_future_schedule
from groundcrew import GroundCrewAgent
from airtankeragent import AirtankerAgent

from forecast_provider import get_forecast
from wind_schedule_utils import sample_schedule_from_forecast



from surrogate_fire_model_CK2_multi_phase import SurrogateFireModelROS_CK2Multi
from FIRE_MODEL_CUDA import SurrogateFireModelROS
import hashlib, json, pathlib
DUMP_SCHEDULES = False

#For Naming
from functools import lru_cache

def _roster_signature(model) -> tuple:
    """Return an immutable signature of the current roster."""
    return tuple(sorted((ag.__class__.__name__, ag.unique_id)
                        for ag in model.schedule.agents))

@lru_cache(maxsize=None)
def _ordinal_map_cached(sig: tuple) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for atype, uid in sig:                       # already sorted
        mapping.setdefault(atype, []).append(uid)
    return mapping

def ordinal_map(model):
    return _ordinal_map_cached(_roster_signature(model))

#ROLLOUT SAVING STUFF

from pathlib import Path
import sys

def count_expandable_assets(model) -> int:
    """
    Return how many **asset INSTANCES** currently have at least one
    legal move *in the coming decision slice*.
    • airtankers   → always counted
    • ground-crew  → counted only if state == 'Standby'
    """
    from groundcrew import GroundCrewAgent
    from airtankeragent import AirtankerAgent

    c = 0
    for ag in model.schedule.agents:
        if isinstance(ag, GroundCrewAgent):
            if ag.state == "Standby":
                c += 1
        elif isinstance(ag, AirtankerAgent):
            c += 1
    return min(c, 3)            # we only map up to three


def _save_rollout_snapshot(sim_state, *, tag=""):
    """
    Dump a high-res PNG of the fire at sim_state.time.

    The file lands in <repo>/MCTS_Rollout_Images and is named
      rollout_<tag>_<minute>_<nanos>.png
    so you get a unique file per rollout regardless of how often the
    function is called.
    """
    # 1) pick a stable, script-relative output directory
    script_dir = Path(getattr(sys.modules[__name__], "__file__", Path.cwd())).resolve().parent
    out_dir    = script_dir / "MCTS_Rollout_Images"
    out_dir.mkdir(exist_ok=True)

    # 2) build a unique, human-readable filename
    ts   = int(time.time_ns())            # guarantees uniqueness
    mins = int(sim_state.time)
    fname = f"rollout_{tag}_{mins}min_{ts}.png"

    # 3) render and save
    fig, ax = plot_fire(
        sim_state.fire,
        time=sim_state.time,
        max_time=sim_state.fire_spread_sim_time
    )
    fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ROLL-SNAP] saved {out_dir / fname}")
# ─────────────────────────────────────────────────────────────────────────────




# ---------------------------------------------------------------------------
# Clean baseline for a given schedule
# ---------------------------------------------------------------------------
def _clean_baseline_score(sim_state, schedule):
    """
    Build a completely fresh surrogate fire-model, feed it *schedule*,
    **no agents, no drops, no fire-line edits**, and return its fire-score.
    """
    # Pick any speed/dir for the constructor – they are ignored once
    # wind_schedule is supplied.
    dummy_speed, dummy_dir = schedule[-1][2], schedule[-1][3]

    # fm = SurrogateFireModelROS_CK2Multi(
    #         tif_path   = sim_state.fire.tif_path,
    #         sim_time   = sim_state.fire_spread_sim_time,
    #         wind_speed = dummy_speed,
    #         wind_direction_deg = dummy_dir,
    #         max_iter   = sim_state.fire.max_iter,
    #         tol        = sim_state.fire.tol,
    #         wind_schedule = schedule
    # )

    fm = SurrogateFireModelROS(
        tif_path=sim_state.fire.tif_path,
        sim_time=sim_state.fire_spread_sim_time,
        wind_speed=dummy_speed,
        wind_direction_deg=dummy_dir,
        max_iter=sim_state.fire.max_iter,
        tol=sim_state.fire.tol,
        wind_schedule=schedule
    )

    # fire-score at full simulation span
    print("Fm.simtime: ", fm.sim_time)
    fs = fm.calculate_fire_score(fm.sim_time)
    area = fm.calculate_area_score(fm.sim_time)
    try:
        bldg = fm.calculate_building_score(fm.sim_time)  # counts residential (>=11) already
    except Exception:
        bldg = 0.0

    return (fs, area, bldg)
    # return fm.calculate_fire_score(fm.sim_time)


def _make_baseline_schedule(truth_model, forecast_df):
    """Return full 0-→end schedule made of past truth + future means."""
    past = [seg for seg in truth_model.wind_schedule            # end ≤ now
            if seg[1] <= truth_model.time]

    future_mu = [
        (int(r.start_min), int(r.end_min),
         float(r.speed_mean), float(r.dir_mean))
        for r in forecast_df.itertuples()
        if r.end_min > truth_model.time
    ]

    return past + future_mu


def schedule_hash(schedule):
    """
    Return a short hex digest that uniquely identifies an ordered list
    of (start,end,speed,dir) tuples.
    """
    m = hashlib.sha256()
    for seg in schedule:
        m.update(json.dumps(seg, sort_keys=True).encode())
    return m.hexdigest()[:10]            # first 10 chars is plenty


import re, unicodedata

def slugify(text, maxlen=40):
    """
    Turn arbitrary text into a filesystem-safe slug.
    Keeps alphanumerics, underscores and dashes.
    Cuts off at `maxlen` characters.
    """
    text = unicodedata.normalize("NFKD", str(text))
    text = re.sub(r"[^\w\-]+", "_", text)          # replace bad chars
    return text[:maxlen]

# ── schedule ON / OFF -------------------------------------------------------
def _schedule_enabled(model_or_state):
    """True if we’re supposed to use wind-schedules / forecasts."""
    return getattr(model_or_state, "wind_schedule", None) is not None


# --- Define the MCTS node structure ---
class MCTSNode:
    def __init__(self, sim_state, parent=None, action=None, node_name="root", depth=0):
        """
        sim_state: a snapshot of the simulation state (WildfireModel instance).
        parent: parent MCTSNode.
        action: the action taken from the parent to reach this node.
        node_name: a string representing the sequence of actions.
        depth: depth of the node in the tree.
        """
        self.sim_state = sim_state
        self.parent = parent
        self.action = action
        self.node_name = node_name
        self.depth = depth
        self.children = []
        self.visits = 0
        self.reward = 0.0  # cumulative reward (reward = -fire_score)
        self.untried_actions = []  # List of (sector_actions, action_code) pairs.
        self.expansion_time = 0.0         # Time taken when this node was expanded.
        self.total_rollout_time = 0.0       # Cumulative rollout time for this node.
        self.rollout_count = 0            # Count of rollouts performed from this node.
        self.exec_time = 0.0              # Execution time (relative to MCTS start) at node expansion.
        self.iteration = None             # The iteration number when this node was created.


    def is_terminal(self, max_depth: int) -> bool:
        """
        A node is terminal when either:
          • we reached the configured search depth **or**
          • the underlying simulation has reached its overall_time_limit
        """
        return (
                self.depth >= max_depth or
                self.sim_state.time >= self.sim_state.overall_time_limit
        )

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def uct_value(self, child, exploration_constant=math.sqrt(2)):
        # UCT value: average reward + exploration term.
        if child.visits == 0:
            return float('inf')
        return (child.reward / child.visits) + exploration_constant * math.sqrt(math.log(self.visits) / child.visits)

    def best_child(self, exploration_constant=math.sqrt(2)):
        # Select the child with the highest UCT value.
        best = None
        best_value = -float('inf')
        for child in self.children:
            uct = self.uct_value(child, exploration_constant)
            if uct > best_value:
                best_value = uct
                best = child
        return best


def get_asset_shorthand(asset_type):
    mapping = {
        "FireHerc": "FH",
        "C130J": "C130",
        "AT802F": "AT802F",
        "Dash8_400MRE": "D8",
        "Scooper": "SC",
        "GroundCrewAgent": "GC",
    }
    return mapping.get(asset_type, asset_type[:3].upper())



# ---------------------------------------------------------------------------
# clone_simulation — handles GroundCrewAgent *and* AirtankerAgent cloning along
# with the cloning of the whole environment
# ---------------------------------------------------------------------------
def clone_simulation(parent_model, new_wind_schedule=None):
    """
    Create a child WildfireModel with full runtime state copied over,
    using .deep_clone() for GroundCrewAgent and AirtankerAgent.
    All CUDA data stay on the GPU; only plain-Python objects are copied.
    """

    # 0) flush write-behind buffers so the fire grid is up-to-date
    for ag in parent_model.schedule.agents:
        if isinstance(ag, GroundCrewAgent):
            ag.flush_clear_buffer()

    # 1) build the shell model (this will spawn placeholder agents)
    child = WildfireModel(
        airtanker_counts     = parent_model.agent_counts,
        wind_speed           = parent_model.wind_speed,
        wind_direction       = parent_model.wind_direction,
        base_positions       = parent_model.base_positions,
        lake_positions       = parent_model.lake_positions,
        time_step            = parent_model.time_step,
        debug                = parent_model.debug,
        start_time           = parent_model.start_time,
        case_folder          = parent_model.case_folder,
        overall_time_limit   = parent_model.overall_time_limit,
        fire_spread_sim_time = parent_model.fire_spread_sim_time,
        operational_delay    = parent_model.operational_delay,
        enable_plotting      = False,
        groundcrew_count     = parent_model.groundcrew_count,
        groundcrew_speed     = parent_model.groundcrew_speed,
        elapsed_minutes      = parent_model.time,
        groundcrew_sector_mapping        = parent_model.groundcrew_sector_mapping,
        second_groundcrew_sector_mapping = parent_model.second_groundcrew_sector_mapping,
        wind_schedule        = (new_wind_schedule
                                if new_wind_schedule is not None
                                else parent_model.wind_schedule),
        fuel_model_override  = parent_model.fire.fuel_model,
        num_sectors=parent_model.num_sectors,
    )

    # 2) top-level scalar / array state
    child.time                = parent_model.time
    child.current_time        = parent_model.current_time
    child.fire.fuel_model     = parent_model.fire.fuel_model.copy()
    child.baseline_fire_score = parent_model.baseline_fire_score
    child.contained_sectors   = parent_model.contained_sectors.copy()
    child.retarded_sectors    = parent_model.retarded_sectors.copy()

    child._pair_sync_done = getattr(parent_model, "_pair_sync_done", False)
    child._pair_sync_list = []  # never reuse parent’s transient queue


    if hasattr(parent_model, "fixed_center"):
        child.fixed_center = parent_model.fixed_center
        child.fixed_angle_ranges = parent_model.fixed_angle_ranges

        child.sector_center = parent_model.sector_center
        child.sector_angle_ranges = parent_model.sector_angle_ranges
        child.sector_boundaries = parent_model.sector_boundaries.copy()

    # ---- copy per-boundary anchor history ------------------
    if hasattr(parent_model, "boundary_anchors"):
        child.boundary_anchors = {
            b: {role: tup for role, tup in roles.items()}
            for b, roles in parent_model.boundary_anchors.items()
        }

    else:
        child.boundary_anchors = {}

    # 3) make sure helper lists exist even if WildfireModel never declared them
    if not hasattr(child, "ground_crews"):
        child.ground_crews = []
    if not hasattr(child, "aircraft"):
        child.aircraft = []

    # 4) remove the placeholder crews/aircraft the ctor spawned
    for ag in list(child.schedule.agents):
        if isinstance(ag, (GroundCrewAgent, AirtankerAgent)):
            child.schedule.remove(ag)
    child.ground_crews.clear()
    child.aircraft.clear()

    # 5) rebuild every agent from the parent
    for old in parent_model.schedule.agents:

        if isinstance(old, GroundCrewAgent):
            new_gc = old.deep_clone(child)
            child.ground_crews.append(new_gc)
            child.schedule.add(new_gc)

        elif isinstance(old, AirtankerAgent):
            new_at = old.deep_clone(child)
            child.aircraft.append(new_at)
            child.schedule.add(new_at)

        else:
            import copy
            new_agent = copy.copy(old)
            new_agent.model = child
            child.schedule.add(new_agent)

    return child


def simulate_in_place(model, sector_actions, duration=120):
    """
    Like simulate_branch but does NOT clone the model.
    It assigns the given sector_actions to the model's agents
    and then steps the model in place for the desired duration.
    """
    remaining = model.overall_time_limit - model.time
    if remaining <= 0:
        return                          # nothing left to simulate
    duration = min(duration, remaining) # clamp slice length
    # ---------------------------------------------------------------------

    print('simulating in place')
    # Assign each agent to its chosen sector
    assignment_counters = {asset: 0 for asset in sector_actions.keys()}
    # --- assign sectors -------------------------------------------------
    assignment_counters = {k: 0 for k in sector_actions.keys()}
    for agent in sorted(model.schedule.agents,
                        key=lambda ag: (ag.__class__.__name__, ag.unique_id)):
    # for agent in model.schedule.agents:
        atype = agent.__class__.__name__
        if isinstance(agent, GroundCrewAgent):
            if (atype in sector_actions) and agent.state == "Standby":
                idx = assignment_counters[atype]
                if idx < len(sector_actions[atype]):
                    raw_sector = sector_actions[atype][idx]  # 1-based
                    # ignore sectors that are already contained
                    if model.is_sector_contained(raw_sector - 1):
                        continue
                    assignment_counters[atype] += 1
                    # one-liner: computes new start/finish internally
                    agent.assign_to_sector(raw_sector - 1)  # 0-based
            # crews that are busy keep their current task
            continue

        # ── 3b. All other assets (airtankers, etc.) ─────────────────────
        if atype in sector_actions:
            idx = assignment_counters[atype]
            chosen_sector = sector_actions[atype][idx]
            assignment_counters[atype] += 1
        else:
            chosen_sector = 1  # fallback sector

        agent.assigned_sector = (f"Sector {chosen_sector}", chosen_sector - 1)

    # for agent in model.schedule.agents:
    #     asset_type = agent.__class__.__name__
    #     if asset_type in sector_actions:
    #         idx = assignment_counters[asset_type]
    #         chosen_sector = sector_actions[asset_type][idx]
    #         assignment_counters[asset_type] += 1
    #     else:
    #         chosen_sector = 1  # fallback sector if not found
    #     agent.assigned_sector = (f"Sector {chosen_sector}", chosen_sector - 1)

    # Step forward in time for 'duration' minutes.
    start_time = model.time
    while model.time - start_time < duration:
        # model.step() returns False if we hit the time limit or containment
        if not model.step():
            break



# ------------------------------------------------------------------
def apply_action_with_random_wind(parent_state,
                                  sector_actions,
                                  *,
                                  forecast_df,
                                  duration=120):
    """
    Clone *parent_state*, sample ONE new wind schedule for the next
    120-min slot, apply the sector_actions, advance `duration` minutes
    and return the *new* state.  *Does not* modify the tree node.
    """
    return simulate_branch(parent_state,
                           sector_actions,
                           forecast_df=forecast_df,
                           duration=duration)
# ------------------------------------------------------------------

def simulate_branch(parent_model, sector_actions, *, forecast_df=None, duration=120):
    """
    Clone `parent_model`, assign `sector_actions`, advance
    the clock by `duration` minutes and return the new model.

    ── Two operating modes ─────────────────────────────────────────
    1. **Schedules ON**  → parent_model.wind_schedule is **not None**
       ▸ sample a fresh future schedule from `forecast_df`
       ▸ prepend the past schedule segments
       ▸ clone with that combined list

    2. **Schedules OFF** → parent_model.wind_schedule is **None**
       ▸ clone straight away (new_wind_schedule=None)
       ▸ skip every forecast / sampling step
    """
    ####################################################################
    ####################################################################
    #code to fix it spilling over past 2000
    remaining = parent_model.overall_time_limit - parent_model.time
    if remaining <= 0:
        # no time left – just hand back a clone frozen at the limit
        return clone_simulation(parent_model,
                                new_wind_schedule=parent_model.wind_schedule)

    duration = min(duration, remaining)

    ####################################################################
    # ─────────────────────────────────────────────────────────────
    schedules_enabled = parent_model.wind_schedule is not None

    # ── Clone the sim state ──────────────────────────────────────
    if schedules_enabled:
        if forecast_df is None:
            raise ValueError("simulate_branch(): forecast_df must be supplied "
                             "when schedules are enabled")

        rng = np.random.default_rng()
        future_sched = sample_future_schedule(
            forecast_df,
            current_minute=int(parent_model.time),
            rng=rng)

        # keep the past exactly as it was in the parent
        past_sched = [seg for seg in parent_model.wind_schedule
                      if seg[1] <= parent_model.time]

        forecast_sched = past_sched + future_sched
        branch_model = clone_simulation(
            parent_model,
            new_wind_schedule=forecast_sched)

        branch_model._wind_schedule_used = forecast_sched
    else:
        # schedules disabled → no forecasting, no past segments
        branch_model = clone_simulation(
            parent_model,
            new_wind_schedule=None)

    # ── Assign sectors to all agents ─────────────────────────────
    assignment_counters = {k: 0 for k in sector_actions.keys()}


    # for agent in branch_model.schedule.agents:
    #     atype = agent.__class__.__name__
    #     if atype in sector_actions:
    #         idx = assignment_counters[atype]
    #         sector = sector_actions[atype][idx]
    #         assignment_counters[atype] += 1
    #     else:
    #         sector = 1                      # fallback
    #     agent.assigned_sector = (f"Sector {sector}", sector - 1)
    for agent in sorted(branch_model.schedule.agents,
                        key=lambda ag: (ag.__class__.__name__, ag.unique_id)):


    # for agent in branch_model.schedule.agents:
        atype = agent.__class__.__name__

        # ground-crew
        if isinstance(agent, GroundCrewAgent):
            if (atype in sector_actions) and agent.state == "Standby":
                idx = assignment_counters[atype]
                if idx < len(sector_actions[atype]):
                    raw_sector = sector_actions[atype][idx]
                    if branch_model.is_sector_contained(raw_sector - 1):
                        continue
                    assignment_counters[atype] += 1
                    agent.assign_to_sector(raw_sector - 1)
            continue

        if atype in sector_actions:
            idx = assignment_counters[atype]
            chosen_sector = sector_actions[atype][idx]
            assignment_counters[atype] += 1
        else:
            chosen_sector = 1
        agent.assigned_sector = (f"Sector {chosen_sector}", chosen_sector - 1)

    start_time = branch_model.time
    while branch_model.time - start_time < duration:
        if not branch_model.step():         # stops on time-limit / containment
            break

    return branch_model


# --- Function to generate all possible actions from a simulation state ---

def get_possible_actions(current_model, duration=120):
    """
    Build every combination of sector assignments for:
      • all airtankers
      • ONLY those GroundCrewAgent instances that are currently in 'Standby'

    Returns a list of tuples: (sector_actions, action_code)
      sector_actions = {"FireHerc": ( … ), "GroundCrewAgent": ( … ), …}
    """
    remaining = current_model.overall_time_limit - current_model.time
    if remaining <= (duration+1):
        return []  # <- nothing left to plan



    n = getattr(current_model, "num_sectors", len(current_model.sector_boundaries))
    sectors = [s for s in range(1, n + 1)
               if not current_model.is_sector_contained(s - 1)]
    asset_counts   = {}
    standby_gc = 0

    # ── scan the roster ─────────────────────────────────────────────
    for ag in current_model.schedule.agents:
        if isinstance(ag, GroundCrewAgent):
            if ag.state == "Standby":
                standby_gc += 1
        else:
            atype = ag.__class__.__name__
            asset_counts[atype] = asset_counts.get(atype, 0) + 1

    if standby_gc:
        asset_counts["GroundCrewAgent"] = standby_gc

    asset_assignments = {
        atype: list(itertools.product(sectors, repeat=n))
        for atype, n in asset_counts.items()
    }
    # asset_types_order = list(asset_counts.keys())
    asset_types_order = sorted(asset_counts.keys())

    overall_combos = itertools.product(
        *(asset_assignments[at] for at in asset_types_order)
    )


    actions = []
    for combo in overall_combos:
        sector_actions = {}
        bits = []
        for atype, assignment in zip(asset_types_order, combo):
            sector_actions[atype] = assignment

            # static UID order for *every* type
            uid_list = ordinal_map(current_model)[atype]

            for idx_in_tuple, sec in enumerate(assignment):
                uid = uid_list[idx_in_tuple]
                ord_ = idx_in_tuple + 1  # 1-based position
                sc = get_asset_shorthand(atype)
                bits.append(f"{sc}{ord_}-{sec}")
        actions.append((sector_actions, "-".join(bits)))
    return actions



def rollout(sim_state, forecast_df, rollout_depth, duration=120, building_weight=1.0):
    """
    One Monte-Carlo rollout from `sim_state`.

    • If wind-schedules are **enabled** (sim_state.wind_schedule ≠ None)
      → draw ONE future schedule from `forecast_df`, keep it fixed,
        then simulate_in_place() for every random action.

    • If wind-schedules are **disabled**
      → clone once (new_wind_schedule=None) and ignore every
        forecast / sampling / dumping step.
    """

    schedules_enabled = sim_state.wind_schedule is not None

    # ── schedules OFF  ─────────────────────────────────────────────
    if not schedules_enabled:
        current_state = clone_simulation(sim_state, new_wind_schedule=None)

        for _ in range(rollout_depth):
            poss = get_possible_actions(current_state, duration)
            if not poss:
                #No valid actions available; advance simulation time with a no-op.
                simulate_in_place(current_state, {}, duration)
                break
            actions, _ = random.choice(poss)
            simulate_in_place(current_state, actions, duration)

        fire_score = current_state.fire.calculate_fire_score(
            current_state.fire_spread_sim_time
        )

        print("Time at which firescore is calculated: ", current_state.fire_spread_sim_time)
        print("Fire score: ", fire_score)
        normalized_reward = -fire_score / sim_state.baseline_fire_score

        # _save_rollout_snapshot(
        #     current_state,
        #     tag=getattr(sim_state, "node_name", "root")
        # )

        return normalized_reward

    # ── schedules ON  ──────────────────────────────────────────────
    rng = np.random.default_rng()
    future_sched = sample_future_schedule(
        forecast_df,
        current_minute=int(sim_state.time),
        rng=rng
    )

    past_sched = [seg for seg in sim_state.wind_schedule if seg[1] <= sim_state.time]
    sched = past_sched + future_sched

    current_state = clone_simulation(sim_state, new_wind_schedule=sched)
    current_state._wind_schedule_used = sched

    # optional debug dump
    if DUMP_SCHEDULES:
        h = schedule_hash(sched)
        origin = getattr(sim_state, "node_name", "root_clone")
        fname = f"rollout_{slugify(origin)}_{h}_{time.time_ns()}.json"
        pathlib.Path("schedule_dumps").mkdir(exist_ok=True)
        with open(f"schedule_dumps/{fname}", "w") as f:
            json.dump({
                "phase": "rollout",
                "origin": origin,
                "hash": h,
                "schedule": sched
            }, f, indent=2)

    i=0
    # random actions
    for _ in range(rollout_depth):

        i = i + 1
        print(f"rollout: {i}")
        poss = get_possible_actions(current_state, duration)
        if not poss:
            simulate_in_place(current_state, {}, duration)
            break
        actions, _ = random.choice(poss)
        simulate_in_place(current_state, actions, duration)

    fire_score = current_state.fire.calculate_fire_score(
        current_state.fire_spread_sim_time
    )

    #old reward system

    # print("Time at which firescore is calculated: ", current_state.fire_spread_sim_time)
    # print("Fire score: ", fire_score)
    #
    # baseline_score = _clean_baseline_score(sim_state, sched)
    # print("Baseline score: ", baseline_score)
    # normalized_reward = -fire_score / baseline_score
    # print("Normalized reward: ", normalized_reward)
    # return normalized_reward
    # after you advance to `current_state` at end of rollout:



    # _save_rollout_snapshot(
    #     current_state,
    #     tag=getattr(sim_state, "node_name", "root")
    # )

    # return normalized_reward

    #NEW REWARD SYSTEM

   # ---- Measure current outcomes ------------------------------------
    # (Area) always available
    area_cur = current_state.fire.calculate_area_score(current_state.fire_spread_sim_time)

    # (Buildings) may be absent; tolerate gracefully
    try:
        bldg_cur = current_state.fire.calculate_building_score(current_state.fire_spread_sim_time)
    except Exception:
        bldg_cur = None  # signals "not available"

    # ---- Build matching baseline with the SAME schedule decision -----
    #     (or None if schedules are OFF)
    _, area_base, bldg_base = _clean_baseline_score(sim_state, sched)

    # ---- Normalize and mix  ------------------------------------------
    a_norm = (area_cur / area_base) if area_base > 0 else 0.0

    use_buildings = (bldg_cur is not None) and (bldg_base is not None) and (bldg_base > 0)
    if use_buildings:
        b_norm = bldg_cur / bldg_base
        mixed = (a_norm + building_weight * b_norm) / (1.0 + building_weight)
    else:
        mixed = a_norm  # fallback to area-only if building layer missing or baseline=0

    # Keep inside [0, 1] to ensure reward ∈ [-1, 0]
    mixed = max(0.0, min(1.0, mixed))
    reward = -mixed

    # Debug prints (optional)
    print(f"[Rollout] area_cur={area_cur:.3f}, area_base={area_base:.3f}, a_norm={a_norm:.3f}")
    if use_buildings:
        print(f"[Rollout] bldg_cur={bldg_cur:.3f}, bldg_base={bldg_base:.3f}, b_norm={b_norm:.3f}, weight={building_weight}")
    else:
        print("[Rollout] Building layer missing or baseline buildings = 0 → area-only.")
    print(f"[Rollout] Mixed ratio={mixed:.3f} → reward={reward:.3f}")

    return reward



def mcts(root_state,
         *,
         iterations            = 100,
         max_depth             = 2,
         duration              = 120,
         exploration_constant  = 1/math.sqrt(2),
         building_weight      = 1.0,
         rollout_depth_adjustment: int = 22,

         track_progress        = False,
         on_iteration          = None):
    """
    Run MCTS starting from *root_state*.

    •  Every edge traversal (whether during SELECTION or EXPANSION) calls
       `simulate_branch()` which     → draws a NEW wind schedule for the next
       120-min slice and advances the simulation.

    •  Thus each node’s value is an average over many independent wind draws;
       no single unlucky sample can poison the estimate.

    All original debug prints, schedule dumps, timing, and UI callbacks are
    kept intact.
    """
    print("starting MCTS …")
    # ⬇ HEADER (bold white)
    print(
        f"\033[1;37m>>> MCTS START: iterations={iterations}, max_depth={max_depth}, C={exploration_constant}\033[0m")  # ⬅ COLORED

    # print(exploration_constant)

    # ── root node & helpers ────────────────────────────────────────────────
    root_node                 = MCTSNode(sim_state=root_state,
                                         parent=None,
                                         action=None,
                                         node_name="root",
                                         depth=0)
    root_node.untried_actions = get_possible_actions(root_state, duration)
    forecast_df               = getattr(root_state, "latest_forecast_df", None)

    # try with and without
    # -----------------------------------------------------------------
    # If there are no valid actions from the root, bail out right away.
    if not root_node.untried_actions:
        return root_node, []  # empty child list signals “no work”
    # -----------------------------------------------------------------
    # optional tracking
    tracking_start_time = datetime.datetime.now() if track_progress else None
    node_exec_times     = []  if track_progress else []
    # ---------------------------------------------------------------------------------
    # Ensure we have a baseline to normalise rewards against
    # ---------------------------------------------------------------------------------
    if getattr(root_state, "baseline_fire_score", None) is None:

        if (root_state.wind_schedule is not None  # ⇢ schedules enabled
                and root_state.latest_forecast_df is not None):  # ⇢ forecast on hand

            # --- deterministic *truth-so-far + forecast-means* schedule ------------

            baseline_sched = _make_baseline_schedule(root_state,
                                                     root_state.latest_forecast_df)

            # --- clone the model with that schedule, measure its fire score --------
            baseline_model = clone_simulation(root_state,
                                              new_wind_schedule=baseline_sched)
            root_state.baseline_fire_score = baseline_model.fire.calculate_fire_score(
                root_state.fire_spread_sim_time)
            root_state.baseline_area = baseline_model.fire.calculate_area_score(
                root_state.fire_spread_sim_time
            )
            root_state.baseline_building = baseline_model.fire.calculate_building_score(
                root_state.fire_spread_sim_time
            )

        else:
            # schedules OFF  →  baseline = “let it burn” in the current (truth) run
            root_state.baseline_fire_score = root_state.fire.calculate_fire_score(
                root_state.fire_spread_sim_time)

            root_state.baseline_area = root_state.fire.calculate_area_score(
                root_state.fire_spread_sim_time
            )
            root_state.baseline_building = root_state.fire.calculate_building_score(
                root_state.fire_spread_sim_time
            )

    print("[MCTS] baseline fire-score =", f"{root_state.baseline_fire_score:.2f}")

    # ---------------------------------------------------------------------
    for i in range(iterations):
        # ⬇ ITERATION HEADER (bold white)
        print(f"\n\033[1;37m--- ITERATION {i + 1}/{iterations} ---\033[0m")  # ⬅ COLORED

        if track_progress:
            elapsed = datetime.datetime.now() - tracking_start_time
            print(f"[{elapsed}] Starting iteration {i+1}/{iterations}")

        # ❶ fresh working copy we will mutate along the path
        node        = root_node
        working_sim = clone_simulation(root_state)          # cheap CuPy copy

        # ── SELECTION ────────────────────────────────────────────────────
        while (not node.is_terminal(max_depth)
               and node.is_fully_expanded()
               and node.children):
            # ⬇ SELECTION LOG (cyan)
            print(
                f"\033[36m[Selection]\033[0m at '{node.node_name}' (depth={node.depth}, visits={node.visits})")  # ⬅ COLORED

            # choose best child
            child = node.best_child(exploration_constant)

            # ⬇ TRAVERSE LOG (yellow)
            print(f"\033[33m[Traverse]\033[0m → '{node.node_name}' → '{child.node_name}'")  # ⬅ COLORED

            # ▶ resample wind & advance 120 min for THAT action
            working_sim = simulate_branch(working_sim,
                                          child.action,
                                          forecast_df=forecast_df,
                                          duration=duration)

            node = child

        # ── EXPANSION ────────────────────────────────────────────────────
        if (not node.is_terminal(max_depth)) and node.untried_actions:
            idx                = random.randrange(len(node.untried_actions))
            action, action_code= node.untried_actions.pop(idx)

            # ⬇ EXPANSION LOG (magenta)
            print(f"\033[35m[Expansion]\033[0m at '{node.node_name}' using action '{action_code}'")  # ⬅ COLORED

            # timing start
            t_exp_start = time.perf_counter()

            # resample & advance
            working_sim = simulate_branch(working_sim,
                                          action,
                                          forecast_df=forecast_df,
                                          duration=duration)

            expansion_time = time.perf_counter() - t_exp_start

            # build child node (template state only)
            child_name = f"{node.node_name}-{working_sim.time}-{action_code}"

            # ⬇ NEW NODE LOG (green)
            print(
                f"\033[32m[New Node]\033[0m created '{child_name}' (depth={node.depth + 1}, expand_time={expansion_time:.3f}s)")  # ⬅ COLORED

            child_node = MCTSNode(sim_state=working_sim,
                                  parent=node,
                                  action=action,
                                  node_name=child_name,
                                  depth=node.depth + 1)
            child_node.expansion_time = expansion_time

            # >>> DEBUG / dump sampled schedule ---------------------------
            if DUMP_SCHEDULES:
                sched = working_sim._wind_schedule_used
                h     = schedule_hash(sched)
                pathlib.Path("schedule_dumps").mkdir(exist_ok=True)
                fname = f"exp_iter{i+1}_{slugify(child_name)}_{h}.json"
                with open(f"schedule_dumps/{fname}", "w") as f:
                    json.dump({"phase"      : "expansion",
                               "iteration"  : i+1,
                               "parent_node": node.node_name,
                               "node_name"  : child_name,
                               "depth"      : child_node.depth,
                               "hash"       : h,
                               "schedule"   : sched},
                              f, indent=2)
            # <<< ---------------------------------------------------------

            # progress tracking
            if track_progress:
                exec_time = (datetime.datetime.now()
                             - tracking_start_time).total_seconds()
                child_node.exec_time = exec_time
                child_node.iteration = i + 1
                node_exec_times.append((i + 1, exec_time))
                print(f"[{datetime.datetime.now() - tracking_start_time}] "
                      f"Expanded {child_node.node_name}  "
                      f"(D:{child_node.depth} It:{child_node.iteration} "
                      f"t={exec_time:.3f}s)")

            # initialise its untried-action list
            if child_node.depth < max_depth:
                child_node.untried_actions = get_possible_actions(working_sim,
                                                                  duration)

            node.children.append(child_node)
            node = child_node    # search continues from the new node

        # ── SIMULATION (roll-out) ────────────────────────────────────────
        #6 is normal value not 12
        # rollout_depth   = (22 + max_depth) - node.depth

        rollout_depth = (int(rollout_depth_adjustment) + max_depth) - node.depth

        # ⬇ ROLLOUT START (blue)
        print(
            f"\033[34m[Rollout]\033[0m from '{node.node_name}' (depth={node.depth}, steps={rollout_depth})")  # ⬅ COLORED

        t_roll_start    = time.perf_counter()
        reward          = rollout(working_sim,       # not node.sim_state !
                                  forecast_df,
                                  rollout_depth,
                                  duration,
                                  building_weight=building_weight)
        rollout_time    = time.perf_counter() - t_roll_start
        # ⬇ ROLLOUT END (blue)
        print(f"\033[34m[Rollout]\033[0m done (time={rollout_time:.3f}s) → reward={reward:.2f}")  # ⬅ COLORED

        node.total_rollout_time += rollout_time
        node.rollout_count      += 1

        # ── BACK-PROPAGATION ─────────────────────────────────────────────
        # ⬇ BACKPROP PUSH (red)
        print(f"\033[31m[Backprop]\033[0m pushing reward={reward:.2f} up from '{node.node_name}'")  # ⬅ COLORED

        cur = node
        while cur is not None:
            cur.visits += 1
            cur.reward += reward
            # ⬇ BACKPROP UPDATE (red)
            print(
                f"    \033[31m[Backprop]\033[0m '{cur.node_name}' visits={cur.visits}, total_reward={cur.reward:.2f}")  # ⬅ COLORED
            cur = cur.parent
        # ⬇ ITERATION COMPLETE (bold white)
        print(f"\033[1;37m--- Iteration {i + 1}/{iterations} complete ---\033[0m")  # ⬅ COLORED
        if track_progress:
            print(f"[{datetime.datetime.now() - tracking_start_time}] "
                  f"Completed iteration {i+1}/{iterations}")

        # ── UI callback ─────────────────────────────────────────────────
        if on_iteration:
            on_iteration(root_node, i + 1)

    return root_node, node_exec_times

# --- Function to traverse the MCTS tree and print information ---
def print_mcts_tree(node, indent=0):
    avg_reward = node.reward / node.visits if node.visits > 0 else 0
    print(" " * indent + f"Depth: {node.depth}, Visits: {node.visits}, Avg Reward: {avg_reward:.3f}, Name: '{node.node_name}'")
    for child in node.children:
        print_mcts_tree(child, indent + 4)


# --- Function to save simulation plots for each node (unchanged) ---
def save_plots(node, folder="MCTS_Node_Images"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig, ax = plot_fire(node.sim_state.fire, time=node.sim_state.time, max_time=node.sim_state.fire_spread_sim_time)
    filepath = os.path.join(folder, f"{node.node_name}.png")
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot for node '{node.node_name}' at simulation time {node.sim_state.time} min.")
    for child in node.children:
        save_plots(child, folder)


# --- Function to write MCTS node statistics to a JSON file ---
def write_mcts_stats_to_json(root, filename="mcts_stats.json"):
    """
    Traverses the MCTS tree and writes out stats for each node to a JSON file.
    Each node's stats include: iteration (if available), node name, depth, visits, total reward,
    average reward, expansion time, rollout count, total rollout time, average rollout time,
    and execution time.
    """
    stats = []

    def traverse(node):
        avg_reward = node.reward / node.visits if node.visits > 0 else 0.0
        avg_rollout_time = node.total_rollout_time / node.rollout_count if node.rollout_count > 0 else 0.0
        stats.append({
            "iteration": node.iteration,  # The iteration number when the node was created.
            "node_name": node.node_name,
            "depth": node.depth,
            "visits": node.visits,
            "total_reward": node.reward,
            "average_reward": avg_reward,
            "expansion_time": node.expansion_time,
            "rollout_count": node.rollout_count,
            "total_rollout_time": node.total_rollout_time,
            "average_rollout_time": avg_rollout_time,
            "exec_time": node.exec_time
        })
        for child in node.children:
            traverse(child)

    traverse(root)
    with open(filename, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved MCTS node stats to {filename}")


# --- Functions to build and plot a graph of the MCTS tree ---
def get_leaf_count(G, root):
    """
    Recursively count the number of leaf nodes in the subtree rooted at 'root'.
    """
    children = list(G.successors(root))
    if not children:
        return 1
    return sum(get_leaf_count(G, child) for child in children)


def hierarchy_pos_tree(G, root, x=0, y=0, dx=1.0, dy=1.0, pos=None):
    """
    Compute positions for a tree layout.

    - x: x-coordinate for the current root.
    - y: y-coordinate for the current root.
    - dx: horizontal distance unit.
    - dy: vertical distance between levels.

    Returns a dictionary mapping node id to (x,y) positions.
    """
    if pos is None:
        pos = {}
    pos[root] = (x, y)
    children = list(G.successors(root))
    if children:
        total_width = sum(get_leaf_count(G, child) for child in children)
        cur_x = x - dx * total_width / 2.0
        for child in children:
            child_leaves = get_leaf_count(G, child)
            child_x = cur_x + dx * child_leaves / 2.0
            pos = hierarchy_pos_tree(G, child, x=child_x, y=y - dy, dx=dx, dy=dy, pos=pos)
            cur_x += dx * child_leaves
    return pos


def build_graph_from_tree(node, G, parent_id=None):
    """
    Recursively adds nodes and edges from the MCTS tree into the NetworkX DiGraph G.
    Each node is labeled with its name, depth, visits, and average reward.
    """
    node_id = id(node)
    avg_reward = node.reward / node.visits if node.visits > 0 else 0.0
    label = f"{node.node_name}\nD:{node.depth} V:{node.visits} R:{avg_reward:.2f}"
    G.add_node(node_id, label=label)
    if parent_id is not None:
        G.add_edge(parent_id, node_id)
    for child in node.children:
        build_graph_from_tree(child, G, node_id)


def plot_mcts_tree_graph(root, filename="MCTS_Tree_Graph.png",
                         node_size=1000, font_size=8, dx=2.0, dy=2.0, scale_factor=2.0):
    """
    Build a tree-layout graph of the MCTS tree using a custom hierarchical layout,
    and save an image that dynamically scales figure size and font size to maintain readability.
    """
    # Build the graph from the tree.
    G = nx.DiGraph()
    build_graph_from_tree(root, G)
    root_id = id(root)
    pos = hierarchy_pos_tree(G, root_id, x=0, y=0, dx=dx, dy=dy)

    x_coords = [p[0] for p in pos.values()]
    y_coords = [p[1] for p in pos.values()]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    width = max_x - min_x
    height = max_y - min_y

    fig_width = max(8, width * scale_factor)
    fig_height = max(6, height * scale_factor)

    plt.figure(figsize=(fig_width, fig_height))
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos,
            labels=labels,
            with_labels=True,
            node_size=node_size,
            node_color="lightblue",
            font_size=font_size,
            arrows=True,
            edge_color="gray")
    plt.title("MCTS Tree Structure (Custom Hierarchical Layout)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved MCTS tree graph as '{filename}'.")


# --- Main script execution ---
def main():
    # Set up initial simulation parameters.
    airtanker_counts = {
        "FireHerc": 1,  # Modify as needed.
        # "C130J": 2,
        # "AT802F": 0,
    }
    wind_speed = 0
    wind_direction = 180
    base_positions = [(0, 0)]
    lake_positions = []
    time_step = 1
    debug = False
    start_time = datetime.datetime.strptime("11:00", "%H:%M")
    case_folder = "case1"
    overall_time_limit = 10000
    fire_spread_sim_time = 10000
    operational_delay = 0
    groundcrew_sector_mapping = [0]
    second_groundcrew_sector_mapping = None

    # Create the root simulation state.
    root_model = WildfireModel(
        airtanker_counts,
        wind_speed,
        wind_direction,
        base_positions,
        lake_positions,
        time_step,
        debug,
        start_time,
        case_folder,
        overall_time_limit,
        fire_spread_sim_time,
        operational_delay,
        enable_plotting=False,
        groundcrew_count=0,  # spin up 3 ground-crew agents
        groundcrew_speed=30,
        elapsed_minutes=2000,
        groundcrew_sector_mapping=None,
        second_groundcrew_sector_mapping =None
    )


    # Run MCTS with specified iterations and depth.
    iterations = 50  # Increase this value for a more thorough search.
    max_depth = 3   # Maximum decision steps.
    duration = 120    # Simulation duration per branch decision.
    # Enable progress tracking to record node execution times.
    # mcts_tree, node_exec_times = mcts(root_model, iterations, max_depth, duration, exploration_constant= 2500, track_progress=True)
    mcts_tree, node_exec_times = mcts(
        root_model,
        iterations=iterations,
        max_depth=max_depth,
        duration=duration,
        exploration_constant=2500,
        track_progress=True
    )
    # Plot node execution times if available.
    if node_exec_times:
        node_numbers, exec_times = zip(*node_exec_times)
        plt.figure()
        plt.plot(node_numbers, exec_times, marker='o')
        plt.xlabel("Iteration Number")
        plt.ylabel("Execution Time (seconds)")
        plt.title("Node Execution Times")
        plt.savefig("node_execution_times.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved node execution times plot as 'node_execution_times.png'.")

    # Print the MCTS tree summary.
    print("\nMCTS Tree Structure:")
    print_mcts_tree(mcts_tree)

    # Save simulation plots for each node.
    save_plots(mcts_tree, folder="MCTS_Node_Images")

    # Save a graph image of the MCTS tree.
    plot_mcts_tree_graph(mcts_tree, filename="MCTS_Tree_Graph.png")

    # Write node-level stats to JSON for later analysis.
    write_mcts_stats_to_json(mcts_tree, filename="mcts_stats.json")

    # For a final decision, select the child of the root with the highest average reward.
    if mcts_tree.children:
        best_child = max(mcts_tree.children, key=lambda n: n.reward / n.visits if n.visits > 0 else -float('inf'))
        avg_reward = best_child.reward / best_child.visits if best_child.visits > 0 else 0
        print(f"\nBest initial action: {best_child.node_name} with average reward {avg_reward:.3f}")
        # Since reward = -fire_score, convert back to the fire score.
        best_fire_score = -avg_reward
        print(f"Estimated fire score (area burned): {best_fire_score:.3f}")
    else:
        print("No actions were explored in MCTS.")


if __name__ == '__main__':
    cProfile.run('main()', 'profile_experiment_edits_cupy.out')
    # stats = pstats.Stats('profile_newsplit_3_assets_600_depth4.out')
    # stats.strip_dirs().sort_stats('time').print_stats(20)
