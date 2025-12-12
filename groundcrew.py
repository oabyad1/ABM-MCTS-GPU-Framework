import math
import numpy as np
import mesa
import firelinepath
from collections import namedtuple

# Define a Point tuple for clarity.
Point = namedtuple("Point", ["x", "y"])



# ─────────────────────────────────────────────────────────────
# ANCHOR DEBUGGING: log + dump helpers
# ─────────────────────────────────────────────────────────────
def _anchor_log_event(model, *, action: str, boundary_idx: int, role: str,
                      cell: tuple[int, int] | None, by=None):
    """
    Append a structured log row any time anchors are set/completed/purged.
    action ∈ {"set","complete","purge"}
    by may be an Agent, an int, or a short string (e.g., "sync", "planner").
    """
    if not hasattr(model, "_anchor_log"):
        model._anchor_log = []
    if not hasattr(model, "_anchor_seq"):
        model._anchor_seq = 0
    seq = model._anchor_seq
    model._anchor_seq += 1
    if hasattr(by, "unique_id"):
        by = f"GC#{by.unique_id:02d}"
    model._anchor_log.append({
        "seq": seq,
        "t": float(getattr(model, "time", float("nan"))),
        "bdy": int(boundary_idx),
        "role": str(role),
        "action": str(action),
        "cell": tuple(cell) if cell is not None else None,
        "by": by,
    })


def dump_anchors(model, include_world=True):
    """
    Print the current state of ALL anchors (completed or not) for every boundary.
    """
    print("\n=== boundary_anchors (NOW) ===")
    n_bdys = len(getattr(model, "sector_boundaries", []))
    store = getattr(model, "boundary_anchors", {})
    if not store:
        print("<empty>")
        return
    for b in range(n_bdys):
        roles = store.get(b, {})
        for role in ("start", "finish"):
            rec = roles.get(role)
            if not rec:
                print(f"b{b:02d} {role:<6} — <none>")
                continue
            cell, ts, completed = rec
            msg = f"b{b:02d} {role:<6} rc={tuple(cell)}  t={ts:.1f}  completed={bool(completed)}"
            if include_world:
                try:
                    wx, wy = cell_to_world(model, *cell)
                    msg += f"  world=({wx:.1f}, {wy:.1f})"
                except Exception:
                    pass
            print(msg)
    print("=== end ===\n")


def dump_anchor_log(model, tail: int | None = 50):
    """
    Print the chronological anchor *history* (who set/completed/purged).
    tail=None prints full history; default shows last 50 rows.
    """
    log = getattr(model, "_anchor_log", [])
    if not log:
        print("\n<no anchor log yet>\n")
        return
    rows = log[-tail:] if tail else log
    print("\nseq   time   bdy  role    action    cell         by")
    for e in rows:
        print(f"{e['seq']:04d}  {e['t']:6.1f}  {e['bdy']:3d}  {e['role']:<7} "
              f"{e['action']:<8} {str(e['cell']):<12} {e['by']}")
    print()


# BASE_CLEAR_RATE = {
#     1: 0.019,  2: 0.025,  3: 0.107, 4: 0.186, 5: 0.107,
#     6: 0.107,  7: 0.107,  8: 0.027, 9: 0.009, 10: 0.075,
#     11: 0.075, 12: 0.075, 13: 0.186,
#     91: 0.0,   92: 0.25, 93: 0.0,  98: 0.0,  99: 0.0,
#     101: 0.6 / 20, 102: 0.6 / 20, 103: 0.6 / 20, 104: 0.7 / 20,
#     105: 0.5 / 20, 106: 0.6 / 20,
#     107: 2.5 / 20, 108: 2.5 / 20, 109: 2.5 / 20,
#     121: 2.3 / 20, 122: 2.4 / 20, 123: 2.3 / 20, 124: 2.4 / 20,
#     141: 2.3 / 20, 142: 4.0 / 20, 143: 3.9 / 20, 144: 2.2 / 20,
#     145: 4.1 / 20, 146: 2.2 / 20, 147: 4.2 / 20, 148: 2.4 / 20, 149: 4.1 / 20,
#     161: 1.7 / 20, 162: 1.7 / 20, 163: 1.7 / 20, 164: 2.3 / 20, 165: 2.0 / 20,
#     181: 1.4 / 20, 182: 0.2 / 20, 183: 0.8 / 20, 184: 1.5 / 20, 185: 0.9 / 20,
#     186: 0.3 / 20, 187: 3.7 / 20, 188: 0.7 / 20, 189: 0.5 / 20,
#     201: 1.5 / 20, 202: 1.5 / 20, 203: 3.7 / 20, 204: 3.9 / 20,
# }


BASE_CLEAR_RATE = {
    1: 0.157,
    2: 0.157,
    3: 0.157,
    4: 0.298,
    5: 0.304,
    6: 0.304,
    7: 0.373,
    8: 0.216,
    9: 0.216,
    10: 0.216,
    11: 0.099,
    12: 0.213,
    13: 0.298,
    91: 0.000,
    92: 0.250,
    93: 0.000,
    97: 0.000,
    98: 0.000,
    99: 0.000,
    101: 0.157,
    102: 0.157,
    103: 0.157,
    104: 0.157,
    105: 0.157,
    106: 0.157,
    107: 0.157,
    108: 0.157,
    109: 0.157,
    121: 0.157,
    122: 0.157,
    123: 0.157,
    124: 0.157,
    141: 0.304,
    142: 0.304,
    143: 0.304,
    144: 0.304,
    145: 0.304,
    146: 0.298,
    147: 0.298,
    148: 0.298,
    149: 0.298,
    161: 0.216,
    162: 0.216,
    163: 0.216,
    164: 0.216,
    165: 0.216,
    166: 0.216,
    181: 0.216,
    182: 0.216,
    183: 0.216,
    184: 0.216,
    185: 0.216,
    186: 0.216,
    187: 0.216,
    188: 0.216,
    189: 0.216,
    201: 0.099,
    202: 0.213,
    203: 0.298,
}

def pretty_print_fireline_result(result, precision=2, show_header=True):
    """
    Nicely format the list returned by fireline_between_two_points().
    Each element: ([row, col], [clear, burn_pen, dist_pen, cum_time, cum_cost])
    """
    if show_header:
        print(f"{'idx':>4} {'row':>5} {'col':>5} "
              f"{'clear':>10} {'burn':>10} {'dist':>10} "
              f"{'cum_t':>10} {'cum_cost':>10}")

    fmt = f"{{:4d}} {{:5d}} {{:5d}} " \
          f"{{:10.{precision}f}} {{:10.{precision}f}} {{:10.{precision}f}} " \
          f"{{:10.{precision}f}} {{:10.{precision}f}}"

    for i, (rc, metrics) in enumerate(result):
        r, c = int(rc[0]), int(rc[1])
        clear, burn, dist, cum_t, cum_cost = map(float, metrics)
        print(fmt.format(i, r, c, clear, burn, dist, cum_t, cum_cost))

def cell_to_world(model, r: int, c: int) -> tuple[float, float]:
    dx, dy = model.fire.transform[0], model.fire.transform[4]  # Note: dy is negative.
    x0, y0 = model.fire.bounds.left, model.fire.bounds.top
    return (x0 + (c + 0.5) * dx, y0 + (r + 0.5) * dy)

def world_to_cell(model, x: float, y: float) -> tuple[int, int]:
    dx, dy = model.fire.transform[0], model.fire.transform[4]
    x0, y0 = model.fire.bounds.left, model.fire.bounds.top
    row = int((y - y0) / dy)  # dy is negative → rows grow downward
    col = int((x - x0) / dx)
    return (row, col)

# Module-level knobs.
BATCH_SIZE = 200   # Number of cells to clear before updating the fuel model.
CLEARED_ID = 98    # Identifier for a cleared cell.


# ──────────────────────────────────────────────────────────────────
#  Helpers for probe-and-extend buffering
# ──────────────────────────────────────────────────────────────────
def _travel_minutes(model,
                    start_rc: tuple[int, int],
                    origin_rc: tuple[int, int] | None = None) -> float:
    """
    Minutes a ground-crew needs to walk from *origin_rc* to *start_rc*
    at the nominal groundcrew_speed (m / min).

    If origin_rc is None we conservatively fall back to the sector-centre.
    """
    if origin_rc is None:
        ox, oy = model.sector_center.x, model.sector_center.y
    else:
        ox, oy = cell_to_world(model, *origin_rc)

    sx, sy = cell_to_world(model, *start_rc)
    v = getattr(model, "groundcrew_speed", 60.0)      # default 60 m / min
    return math.hypot(sx - ox, sy - oy) / v


def _rough_line_minutes(model,
                        start_rc: tuple[int, int],
                        finish_rc: tuple[int, int]) -> float:
    """
    One-shot call to firelinepath that returns *only* the cumulative
    build time (minutes) for a straight-up path.  All cells are treated
    as feasible – we need a magnitude, not perfection.
    """
    fire      = model.fire
    raw_mtt   = fire.arrival_time_grid
    mtt       = np.where(np.isnan(raw_mtt), np.nan,
                         np.clip(raw_mtt - model.time, 0, None))
    fuel      = fire.fuel_model.astype(int)
    feasibility = np.ones_like(fuel, dtype=bool)

    value_map = BASE_CLEAR_RATE          # hours / cell

    fuel = fire.fuel_model.astype(int)
    fuel = np.where(fuel < 0, 102, fuel)  # or whatever valid ID you’ve defined

    res = firelinepath.fireline_between_two_points(
        start=np.array(start_rc, int),
        mtt=mtt,
        fuel=fuel,
        fuel_clear_cost=value_map,
        feasibility=feasibility,
        finish=np.array(finish_rc, int),
        clear_burning_penalty=1000000.0,
        distance_penalty=0.001,
        firefighter_fire_buffer_time=120
    )
    times = [metrics[3] for _, metrics in res]    # cumulative min
    return times[-1] if times else 0.0



# ─────────────────────────────────────────────────────────────────────────────
# ADD *once* near the top of groundcrew.py (after the imports)
# ─────────────────────────────────────────────────────────────────────────────
def _crews_in_sector(model, sector_idx) -> list["GroundCrewAgent"]:
    """Return *all* ground-crew currently assigned to <sector_idx>."""
    from groundcrew import GroundCrewAgent
    return [
        ag for ag in model.schedule.agents
        if isinstance(ag, GroundCrewAgent)
        and ag.sector_index == sector_idx
    ]

def _active_crews(model, sector_idx) -> int:
    """
    Number of crews that are actually *building line* right now in <sector_idx>.
    (“ToStart” counts too because they’ll soon be on the line.)
    """
    return sum(
        1 for ag in _crews_in_sector(model, sector_idx)
        if ag.state in ("BuildLine", "ToStart")
    )


def _record_anchor(model, boundary_idx: int, cell, role: str, by=None):
    """
    Store <cell> on <boundary_idx> with its role ("start" or "finish"),
    along with the current model time as a timestamp and a completion flag.
    The flag is False by default and set to True when the anchor is completed.
    """
    if not hasattr(model, "boundary_anchors"):
        model.boundary_anchors = {}  # {bdy: {"start": (cell, time, completed),
                                     #        "finish": (cell, time, completed)}}
    model.boundary_anchors.setdefault(boundary_idx, {})
    model.boundary_anchors[boundary_idx][role] = (cell, model.time, False)

    # optional audit
    try:
        _anchor_log_event(model, action="set", boundary_idx=boundary_idx, role=role, cell=cell, by=by)
    except Exception:
        pass

    print(f"\n— after _record_anchor(bdy={boundary_idx}, role={role}, cell={cell}, by={by}) —")
    try:
        dump_anchors(model)  # uses the helper I gave you
    except NameError:
        # minimal inline dump so this never fails
        store = getattr(model, "boundary_anchors", {})
        n_bdys = len(getattr(model, "sector_boundaries", [])) or (max(store.keys(), default=-1) + 1)
        for b in range(n_bdys):
            roles = store.get(b, {})
            for rle in ("start", "finish"):
                rec = roles.get(rle)
                if rec:
                    rc, ts, done = rec
                    print(f"b{b:02d} {rle:<6} rc={tuple(rc)} t={ts:.1f} completed={bool(done)}")
                else:
                    print(f"b{b:02d} {rle:<6} — <none>")
    print("— end dump —\n")


def _get_most_recent_anchor(model, boundary_idx: int):
    """
    For a given boundary index, return a tuple (cell, timestamp, role)
    for the anchor (regardless of its completion state) that was recorded most recently.
    """
    d = getattr(model, "boundary_anchors", {}).get(boundary_idx, {})
    if not d:
        return None, None, None
    most_recent = None
    most_recent_time = None
    most_recent_role = None
    for role, (cell, timestamp, completed) in d.items():
        if most_recent is None or timestamp > most_recent_time:
            most_recent = cell
            most_recent_time = timestamp
            most_recent_role = role
    return most_recent, most_recent_time, most_recent_role


def _get_most_recent_completed_anchor(model, boundary_idx: int):
    """
    Similar to _get_most_recent_anchor but filters to only completed anchors.
    Returns (cell, timestamp, role) if a completed anchor exists; otherwise, (None, None, None).
    """
    d = getattr(model, "boundary_anchors", {}).get(boundary_idx, {})
    most_recent = None
    most_recent_time = None
    most_recent_role = None
    for role, (cell, timestamp, completed) in d.items():
        if completed:
            if most_recent is None or timestamp > most_recent_time:
                most_recent = cell
                most_recent_time = timestamp
                most_recent_role = role
    return most_recent, most_recent_time, most_recent_role


def _mark_anchor_completed(model, boundary_idx: int, role: str,by=None):
    """
    Mark the anchor at <boundary_idx> with the given role as completed.
    """
    if hasattr(model, "boundary_anchors") and boundary_idx in model.boundary_anchors:
        if role in model.boundary_anchors[boundary_idx]:
            cell, timestamp, _ = model.boundary_anchors[boundary_idx][role]
            model.boundary_anchors[boundary_idx][role] = (cell, timestamp, True)
            try:
                _anchor_log_event(model, action="complete", boundary_idx=boundary_idx, role=role, cell=cell, by=by)
            except Exception:
                pass


            print(f"\n— after _mark_anchor_completed(bdy={boundary_idx}, role={role}, by={by}) —")
            try:
                dump_anchors(model)
            except NameError:
                store = getattr(model, "boundary_anchors", {})
                n_bdys = len(getattr(model, "sector_boundaries", [])) or (max(store.keys(), default=-1) + 1)
                for b in range(n_bdys):
                    roles = store.get(b, {})
                    for rle in ("start", "finish"):
                        rec = roles.get(rle)
                        if rec:
                            rc, ts, done = rec
                            print(f"b{b:02d} {rle:<6} rc={tuple(rc)} t={ts:.1f} completed={bool(done)}")
                        else:
                            print(f"b{b:02d} {rle:<6} — <none>")
            print("— end dump —\n")


def _mark_anchor_uncompleted(model, boundary_idx: int, role: str, by=None):
    """
    Flip the anchor’s 'completed' flag back to False (e.g., when a line is truncated).
    """
    if hasattr(model, "boundary_anchors") and boundary_idx in model.boundary_anchors:
        if role in model.boundary_anchors[boundary_idx]:
            cell, timestamp, _ = model.boundary_anchors[boundary_idx][role]
            model.boundary_anchors[boundary_idx][role] = (cell, timestamp, False)
            try:
                _anchor_log_event(model, action="reopen", boundary_idx=boundary_idx, role=role, cell=cell, by=by)
            except Exception:
                pass

            print(f"\n— after _mark_anchor_uncompleted(bdy={boundary_idx}, role={role}, by={by}) —")
            try:
                dump_anchors(model)
            except NameError:
                store = getattr(model, "boundary_anchors", {})
                n_bdys = len(getattr(model, "sector_boundaries", [])) or (max(store.keys(), default=-1) + 1)
                for b in range(n_bdys):
                    roles = store.get(b, {})
                    for rle in ("start", "finish"):
                        rec = roles.get(rle)
                        if rec:
                            rc, ts, done = rec
                            print(f"b{b:02d} {rle:<6} rc={tuple(rc)} t={ts:.1f} completed={bool(done)}")
                        else:
                            print(f"b{b:02d} {rle:<6} — <none>")
            print("— end dump —\n")



# -------------------------------
# Updated plan_start_finish function
# -------------------------------

# def plan_start_finish(model, sector_idx: int, origin_rc: tuple[int, int] | None = None):
#     """
#     Decide a crew’s start/finish cells for <sector_idx> using updated logic.
#
#     Boundaries
#     ----------
#     LEFT  = (sector_idx) mod N
#     RIGHT = (sector_idx+1) mod N
#
#     Rules
#     -----
#     1. The start point is chosen as the most recent completed anchor among the two sector boundaries.
#        • If completed anchors exist on either boundary, the one with the later timestamp is chosen.
#        • If no completed anchor exists, fall back to the most recent (incomplete) anchor.
#        • If no anchor exists at all, buffer a fresh point on the LEFT boundary.
#     2. The finish point is set from the opposite boundary:
#        • If an anchor exists on the opposite boundary, use it.
#        • Otherwise, compute a buffered finish.
#     3. Once chosen, record the start (with role "start") and finish (with role "finish").
#     Returns:
#        (start_cell, finish_cell, start_boundary, finish_boundary)
#     """
#     n_bdys = len(model.sector_boundaries)
#     left_bdy = sector_idx % n_bdys
#     right_bdy = (sector_idx + 1) % n_bdys
#
#     # Attempt to use completed anchors first.
#     left_comp_cell, left_comp_time, _ = _get_most_recent_completed_anchor(model, left_bdy)
#     right_comp_cell, right_comp_time, _ = _get_most_recent_completed_anchor(model, right_bdy)
#
#     if left_comp_cell is not None or right_comp_cell is not None:
#         if left_comp_cell is not None and right_comp_cell is not None:
#             if left_comp_time >= right_comp_time:
#                 start_cell = left_comp_cell
#                 start_bdy  = left_bdy
#             else:
#                 start_cell = right_comp_cell
#                 start_bdy  = right_bdy
#         elif left_comp_cell is not None:
#             start_cell = left_comp_cell
#             start_bdy  = left_bdy
#         else:
#             start_cell = right_comp_cell
#             start_bdy  = right_bdy
#     else:
#         # ✅ No completed anchors → synthesize a fresh start
#         print("no completed anchors found → making NEW start")
#         start_cell = _buffered_cell_by_mtt(model, sector_idx,
#                                            buffer_minutes=model.GC_START_BUFFER_MIN)
#         start_bdy  = left_bdy
#
#
#     # # Attempt to use completed anchors first.
#     # left_comp_cell, left_comp_time, _ = _get_most_recent_completed_anchor(model, left_bdy)
#     # right_comp_cell, right_comp_time, _ = _get_most_recent_completed_anchor(model, right_bdy)
#     #
#     # if left_comp_cell is not None or right_comp_cell is not None:
#     #     if left_comp_cell is not None and right_comp_cell is not None:
#     #         if left_comp_time >= right_comp_time:
#     #             start_cell = left_comp_cell
#     #             start_bdy = left_bdy
#     #         else:
#     #             start_cell = right_comp_cell
#     #             start_bdy = right_bdy
#     #     elif left_comp_cell is not None:
#     #         start_cell = left_comp_cell
#     #         start_bdy = left_bdy
#     #     else:
#     #         start_cell = right_comp_cell
#     #         start_bdy = right_bdy
#     # else:
#     #     # Fallback to using any most recent anchor (even if not completed).
#     #     left_cell, left_time, _ = _get_most_recent_anchor(model, left_bdy)
#     #     right_cell, right_time, _ = _get_most_recent_anchor(model, right_bdy)
#     #     if left_cell is not None and right_cell is not None:
#     #         if left_time >= right_time:
#     #             start_cell = left_cell
#     #             start_bdy = left_bdy
#     #         else:
#     #             start_cell = right_cell
#     #             start_bdy = right_bdy
#     #     elif left_cell is not None:
#     #         start_cell = left_cell
#     #         start_bdy = left_bdy
#     #     elif right_cell is not None:
#     #         start_cell = right_cell
#     #         start_bdy = right_bdy
#     #     else:
#     #         print("called for new START")
#     #         # start_cell = _buffered_start_cell(model, sector_idx, buffer_minutes=1000)
#     #         start_cell = _buffered_cell_by_mtt(model, sector_idx, buffer_minutes=model.GC_START_BUFFER_MIN)
#     #         start_bdy = left_bdy
#
#     # # Choose finish from the opposite boundary.
#     # other_bdy = right_bdy if start_bdy == left_bdy else left_bdy
#     # other_cell, _, _ = _get_most_recent_anchor(model, other_bdy)
#     # if other_cell is not None:
#     #     finish_cell = other_cell
#     #     finish_bdy = other_bdy
#     # else:
#     #     if other_bdy == left_bdy:
#     #         finish_cell = _buffered_start_cell(model, sector_idx)
#     #     else:
#     #         finish_cell = _buffered_finish_cell(model, right_bdy)
#     #     finish_bdy = other_bdy
#
#     # ─────────────────────────────────────────────────────────────
#     #  FINISH  — existing anchor?  else probe (500 min) → extend
#     # ─────────────────────────────────────────────────────────────
#     other_bdy   = right_bdy if start_bdy == left_bdy else left_bdy
#     # other_cell, _, _ = _get_most_recent_anchor(model, other_bdy)
#     other_cell, _, _ = _get_most_recent_completed_anchor(model, other_bdy)
#
#
#     if other_cell is not None:
#         # ❶ There *is* an anchor on the opposite boundary – use it as-is.
#         finish_cell = other_cell
#         finish_bdy  = other_bdy
#
#     else:
#         # ❷ No anchor – fall back to probe-and-extend buffering
#         probe_buf    = model.GC_FINISH_PROBE_MIN
#         # probe_finish = _buffered_finish_cell(model, other_bdy,
#         #                                      buffer_minutes=probe_buf)
#
#         probe_finish = _buffered_cell_by_mtt(model, other_bdy,
#                                              buffer_minutes=probe_buf)
#         import numpy as np
#         fuel_ids = sorted(np.unique(model.fire.fuel_model.astype(int)))
#         missing = [fid for fid in fuel_ids if fid not in BASE_CLEAR_RATE]
#         print("Fuel IDs in raster:", fuel_ids)
#         print("Missing clear-rate IDs:", missing)
#         print(probe_finish)
#
#         R, C = model.fire.arrival_time_grid.shape
#         pr, pc = probe_finish
#         assert 0 <= pr < R and 0 <= pc < C, f"probe_finish out of grid: {probe_finish} not in [0,{R})×[0,{C})"
#
#         dx, dy = model.fire.transform[0], model.fire.transform[4]
#         left, right = model.fire.bounds.left, model.fire.bounds.right
#         top, bottom = model.fire.bounds.top, model.fire.bounds.bottom
#
#         x_min = left + 0.5 * dx
#         x_max = right - 0.5 * dx
#         y_min = bottom + 0.5 * abs(dy)
#         y_max = top - 0.5 * abs(dy)
#
#         wx, wy = cell_to_world(model, pr, pc)
#
#         print(f"probe_finish rc={probe_finish}  world=({wx:.2f}, {wy:.2f})")
#         print(f"valid world ranges: x∈[{x_min:.2f}, {x_max:.2f}], y∈[{y_min:.2f}, {y_max:.2f}]")
#         R, C = model.fire.arrival_time_grid.shape
#         print(f"Valid cell indices → rows: 0 … {R - 1},   cols: 0 … {C - 1}")
#
#         eta_clear  = _rough_line_minutes(model, start_cell, probe_finish)
#         eta_travel = _travel_minutes(model, start_cell, origin_rc)
#
#         if eta_clear > (probe_buf - 100):                      # your threshold
#
#             # final_buf   = probe_buf + eta_clear + eta_travel +  model.GC_FINISH_SAFETY_MIN
#             final_buf   = eta_clear + eta_travel +  model.GC_FINISH_SAFETY_MIN
#
#             # finish_cell = _buffered_finish_cell(model, other_bdy,
#             #                                     buffer_minutes=final_buf)
#             finish_cell = _buffered_cell_by_mtt(model, other_bdy,
#                                                 buffer_minutes=final_buf)
#         else:
#             finish_cell = probe_finish
#
#         finish_bdy = other_bdy
#
#     print("recording from plan start finish ")
#     # Record the chosen anchors (initially as incomplete).
#     _record_anchor(model, start_bdy, start_cell, "start")
#     _record_anchor(model, finish_bdy, finish_cell, "finish")
#
#     print(f"[DEBUG] sector {sector_idx}: start@bdy{start_bdy} {start_cell}   finish@bdy{finish_bdy} {finish_cell}")
#     return start_cell, finish_cell, start_bdy, finish_bdy


def plan_start_finish(model, sector_idx: int, origin_rc: tuple[int, int] | None = None):
    """
    Decide a crew’s start/finish cells for <sector_idx>.

    START:
      • If a completed anchor exists on either boundary, use the most recent.
      • Else: probe the sector's LEFT boundary, compute travel ETA to that probe,
              then final start buffer = ETA_travel + model.GC_START_SAFETY_MIN
              and pick that boundary cell via _buffered_cell_by_mtt().

    FINISH:
      • If a completed anchor exists on the opposite boundary, use it.
      • Else: probe far boundary, compute
              final_buf = eta_clear(start→probe) + eta_travel(start→probe_origin) + model.GC_FINISH_SAFETY_MIN
              and return _buffered_cell_by_mtt(far_boundary, final_buf).
    """
    n_bdys   = len(model.sector_boundaries)
    left_bdy = sector_idx % n_bdys
    right_bdy= (sector_idx + 1) % n_bdys

    # -------------------
    # START (prefer anchors)
    # -------------------
    l_comp_cell, l_comp_t, _ = _get_most_recent_completed_anchor(model, left_bdy)
    r_comp_cell, r_comp_t, _ = _get_most_recent_completed_anchor(model, right_bdy)

    if l_comp_cell is not None or r_comp_cell is not None:
        if l_comp_cell is not None and r_comp_cell is not None:
            if l_comp_t >= r_comp_t:
                start_cell, start_bdy = l_comp_cell, left_bdy
            else:
                start_cell, start_bdy = r_comp_cell, right_bdy
        elif l_comp_cell is not None:
            start_cell, start_bdy = l_comp_cell, left_bdy
        else:
            start_cell, start_bdy = r_comp_cell, right_bdy
    else:
        # No completed anchors → dynamic START from LEFT boundary
        print("no completed anchors found → making NEW dynamic START")

        # 1) initial guess (probe) just to size travel
        start_probe = _buffered_cell_by_mtt(model, sector_idx,
                                            buffer_minutes=model.GC_START_BUFFER_MIN)

        # 2) travel ETA from origin to the probe
        eta_travel = _travel_minutes(model, start_probe, origin_rc)

        # 3) final buffer = travel + safety (120 by default)
        final_buf_start = eta_travel + getattr(model, "GC_START_SAFETY_MIN", 120)

        # 4) pick the boundary cell at that buffer
        start_cell = _buffered_cell_by_mtt(model, sector_idx,
                                           buffer_minutes=final_buf_start)
        start_bdy  = left_bdy

    # -------------------
    # FINISH
    # -------------------
    other_bdy = right_bdy if start_bdy == left_bdy else left_bdy

    # Prefer a completed anchor on the far boundary
    other_comp_cell, _, _ = _get_most_recent_completed_anchor(model, other_bdy)
    if other_comp_cell is not None:
        finish_cell, finish_bdy = other_comp_cell, other_bdy
    else:
        # Probe only for ETA sizing
        probe_buf    = model.GC_FINISH_PROBE_MIN
        probe_finish = _buffered_cell_by_mtt(model, other_bdy, buffer_minutes=probe_buf)

        # ETAs to build/arrive (from the chosen start)
        eta_clear  = _rough_line_minutes(model, start_cell, probe_finish)
        eta_travel = _travel_minutes(model, start_cell, origin_rc)

        # Always dynamic now (probe is only a guess)
        final_buf = eta_clear + eta_travel + model.GC_FINISH_SAFETY_MIN
        finish_cell = _buffered_cell_by_mtt(model, other_bdy, buffer_minutes=final_buf)
        finish_bdy  = other_bdy

    # Record anchors (initially incomplete)
    _record_anchor(model, start_bdy,  start_cell,  "start")
    _record_anchor(model, finish_bdy, finish_cell, "finish")

    print(f"[DEBUG] sector {sector_idx}: start@bdy{start_bdy} {start_cell}   "
          f"finish@bdy{finish_bdy} {finish_cell}")
    return start_cell, finish_cell, start_bdy, finish_bdy





# ─────────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------ planning –
# def _buffered_start_cell(model, sector_idx, buffer_minutes=200):
#     """Exact copy of the old compute_start_cell() but local to this file."""
#     center   = model.sector_center
#     b0 = Point(*model.sector_boundaries[sector_idx - 1])
#     b1 = Point(*model.sector_boundaries[sector_idx])
#     # choose the boundary with the smaller angular gap
#     def _ang(p): return (math.degrees(math.atan2(p.y - center.y,
#                                                  p.x - center.x)) + 360) % 360
#     θ0, θ1 = _ang(b0), _ang(b1)
#     chosen = b0 if (θ1 - θ0) % 360 < (θ0 - θ1) % 360 else b1
#     ang    = math.atan2(chosen.y - center.y, chosen.x - center.x)
#     base_d = math.hypot(chosen.x - center.x, chosen.y - center.y)
#     spread = base_d / model.time if model.time > 0 else model.fire.transform[0]
#     sx     = chosen.x + math.cos(ang) * spread * buffer_minutes
#     sy     = chosen.y + math.sin(ang) * spread * buffer_minutes
#     return world_to_cell(model, sx, sy)

#GOOD
# def _buffered_finish_cell(model, sector_idx, buffer_minutes=1500):
#     """
#     Trimmed-down version of the old compute_finish_cell().
#     The heavy fire-line path pre-lookahead has been dropped – we only need
#     to push the raw perimeter point outward by a time buffer.
#     """
#     center = model.sector_center
#     b = Point(*model.sector_boundaries[sector_idx])          # finish boundary
#     ang  = math.atan2(b.y - center.y, b.x - center.x)
#     base_d = math.hypot(b.x - center.x, b.y - center.y)
#     spread = base_d / model.time if model.time > 0 else model.fire.transform[0]
#     fx = b.x + math.cos(ang) * spread * buffer_minutes
#     fy = b.y + math.sin(ang) * spread * buffer_minutes
#     return world_to_cell(model, fx, fy)

# def _buffered_cell_by_mtt(model,
#                           boundary_idx: int,
#                           buffer_minutes: float
#                           ) -> tuple[int, int]:
#     """
#     March from the current fire centre, along the sector-boundary ray,
#     one raster cell at a time, until we find the first cell with
#     (arrival_time - model.time) >= buffer_minutes.  Skip NaNs.
#     If we exit the grid without hitting the buffer, return the last
#     in‐bounds cell we saw.
#     """
#     # 1) build the shifted-MTT grid
#     raw_mtt = model.fire.arrival_time_grid            # full arrival times
#     mtt = raw_mtt - model.time                       # minutes until fire
#     # 2) origin and boundary in RC indices
#     cx, cy = model.sector_center.x, model.sector_center.y
#     center_rc = world_to_cell(model, cx, cy)
#     bx, by = model.sector_boundaries[boundary_idx].x, \
#              model.sector_boundaries[boundary_idx].y
#     boundary_rc = world_to_cell(model, bx, by)
#     # 3) step direction: sign of delta-R, delta-C
#     dr = boundary_rc[0] - center_rc[0]
#     dc = boundary_rc[1] - center_rc[1]
#     step_r =   0 if dr == 0 else (dr  // abs(dr))
#     step_c =   0 if dc == 0 else (dc  // abs(dc))
#     # 4) march outward
#     R, C = raw_mtt.shape
#     r, c = boundary_rc
#     last_in_bounds = (r, c)
#     while 0 <= r < R and 0 <= c < C:
#         val = mtt[r, c]
#         if not np.isnan(val) and val >= buffer_minutes:
#             return (r, c)
#         last_in_bounds = (r, c)
#         r += step_r
#         c += step_c
#     # 5) nothing met the buffer → return the last edge cell
#     return last_in_bounds

#good july 18/ old perfect

# def _buffered_cell_by_mtt(model,
#                           boundary_idx: int,
#                           buffer_minutes: float
#                          ) -> tuple[int, int]:
#     """
#     March from the fire’s center along the given boundary until we find
#     a cell whose (arrival_time - model.time) >= buffer_minutes.  Skip
#     NaNs, infinite values, and any negative/sentinel values.
#     If no such cell is found before exiting the grid, return the last
#     in‑bounds cell.
#     """
#     import numpy as np
#
#     # 1) pull in the full arrival‐time grid
#     raw = model.fire.arrival_time_grid  # shape (R, C)
#
#     # 2) mask out ANY invalid cells (NaN, ±Inf, negative/sentinel)
#     raw = np.where(
#         np.logical_or.reduce((
#             np.isnan(raw),         # already NaN
#             ~np.isfinite(raw),     # +Inf or -Inf
#             raw < 0,               # negative or sentinel
#         )),
#         np.nan,
#         raw
#     )
#
#     # 3) compute the minutes‑to‑burn grid
#     mtt = raw - model.time
#
#     # 4) find start/end in RC coordinates
#     cx, cy = model.sector_center.x, model.sector_center.y
#     center_rc   = world_to_cell(model, cx, cy)
#
#     bx, by      = (model.sector_boundaries[boundary_idx].x,
#                    model.sector_boundaries[boundary_idx].y)
#     boundary_rc = world_to_cell(model, bx, by)
#
#     # 5) marching step direction
#     dr = boundary_rc[0] - center_rc[0]
#     dc = boundary_rc[1] - center_rc[1]
#     step_r =  0 if dr == 0 else (dr  // abs(dr))
#     step_c =  0 if dc == 0 else (dc  // abs(dc))
#
#     # 6) march outward until you hit the grid edge
#     R, C = raw.shape
#     r, c = boundary_rc
#     last_in_bounds = (r, c)
#
#     while 0 <= r < R and 0 <= c < C:
#         val = mtt[r, c]
#         # only stop on a real, finite value that meets the buffer
#         if np.isfinite(val) and val >= buffer_minutes:
#             return (r, c)
#         last_in_bounds = (r, c)
#         r += step_r
#         c += step_c
#
#     # 7) if nothing qualified, fall back to the last edge cell
#     return last_in_bounds

#new
# def _buffered_cell_by_mtt(model,
#                           boundary_idx: int,
#                           buffer_minutes: float
#                          ) -> tuple[int, int]:
#     """
#     March from the fire’s center along the given boundary until:
#       • we find a cell with (arrival_time - model.time) >= buffer_minutes → return it
#       • we hit a sentinel (NaN/∞/negative) whose preceding valid cell was
#         within 100 minutes of the fire_spread_sim_time → return that sentinel cell
#       • we run off the grid or exhaust without satisfying either → return last valid cell
#     """
#     import numpy as np
#
#     raw       = model.fire.arrival_time_grid   # full arrival times R×C
#     sim_end   = model.fire_spread_sim_time     # final arrival time cutoff
#     tol       = 100.0                          # ±100 min tolerance around sim_end
#
#     # convert center & boundary into grid indices
#     cx, cy        = model.sector_center.x, model.sector_center.y
#     center_rc     = world_to_cell(model, cx, cy)
#     bx, by        = (model.sector_boundaries[boundary_idx].x,
#                      model.sector_boundaries[boundary_idx].y)
#     boundary_rc   = world_to_cell(model, bx, by)
#
#     # march direction
#     dr = boundary_rc[0] - center_rc[0]
#     dc = boundary_rc[1] - center_rc[1]
#     step_r = 0 if dr == 0 else (dr // abs(dr))
#     step_c = 0 if dc == 0 else (dc // abs(dc))
#
#     R, C      = raw.shape
#     r, c      = boundary_rc
#     last_valid = boundary_rc
#
#     while 0 <= r < R and 0 <= c < C:
#         rv = raw[r, c]
#
#         # 1) VALID CELL: finite & non-negative
#         if np.isfinite(rv) and rv >= 0:
#             last_valid = (r, c)
#             mtt = rv - model.time
#             if mtt >= buffer_minutes:
#                 # meets buffer requirement
#                 return (r, c)
#
#         else:
#             # 2) SENTINEL: check if the cell just before (last_valid)
#             #    was within tol of sim_end → if so, this is the fire edge
#             prev_r, prev_c = last_valid
#             prev_rt = raw[prev_r, prev_c]
#             if np.isfinite(prev_rt) and abs(prev_rt - sim_end) <= tol:
#                 return (r, c)
#             # otherwise it's just a stray NaN inside the burn area → skip it
#
#         # step outward
#         r += step_r
#         c += step_c
#
#     # 3) exhausted grid without hitting a buffer or true edge
#     return last_valid



# # new fixed
# def _buffered_cell_by_mtt(model,
#                           boundary_idx: int,
#                           buffer_minutes: float
#                          ) -> tuple[int, int]:
#     """
#     If model.wind_schedule is set, march outward until:
#       • you hit mtt >= buffer_minutes → return that cell
#       • you hit a sentinel (NaN/Inf/negative) whose last valid cell
#         was within 100 min of fire_spread_sim_time → return that sentinel
#       • else run off-grid → return last valid cell
#
#     If no wind_schedule, fall back to the original:
#       March outward, skip any invalid cells, stop at first mtt>=buffer.
#       If none found, return last in-bounds cell.
#     """
#     import numpy as np
#
#     raw = model.fire.arrival_time_grid  # full arrival times (R×C)
#     # translate world → grid
#     cx, cy      = model.sector_center.x, model.sector_center.y
#     center_rc   = world_to_cell(model, cx, cy)
#
#     bx, by = (model.sector_boundaries[boundary_idx].x,
#               model.sector_boundaries[boundary_idx].y)
#     # 1) raw world → grid
#     raw_rc = world_to_cell(model, bx, by)
#
#     # 2) clamp into [0..R-1]×[0..C-1]
#     R, C = raw.shape
#     r0 = max(0, min(R - 1, raw_rc[0]))
#     c0 = max(0, min(C - 1, raw_rc[1]))
#     boundary_rc = (r0, c0)
#     last_valid = boundary_rc
#
#     # 3) now compute your march‐direction from a valid start
#     dr = boundary_rc[0] - center_rc[0]
#     dc = boundary_rc[1] - center_rc[1]
#
#
#     step_r = 0 if dr == 0 else (dr // abs(dr))
#     step_c = 0 if dc == 0 else (dc // abs(dc))
#
#     R, C = raw.shape
#     r, c = boundary_rc
#     last_valid = boundary_rc
#
#     if model.wind_schedule:
#         # new sentinel‐aware logic
#         sim_end = model.fire_spread_sim_time
#         tol     = 100.0
#
#         while 0 <= r < R and 0 <= c < C:
#             rv = raw[r, c]
#             if np.isfinite(rv) and rv >= 0:
#                 last_valid = (r, c)
#                 if (rv - model.time) >= buffer_minutes:
#                     print(f"[FINAL] buffered cell {(r, c)} → "
#                           f"raw_arrival={rv:.1f}, mtt={(rv - model.time):.1f}")
#                     return (r, c)
#             else:
#                 # invalid cell → check if last_valid was right at fire edge
#                 pr, pc = last_valid
#                 prt = raw[pr, pc]
#                 if np.isfinite(prt) and abs(prt - sim_end) <= tol:
#                     mtt_val = prt - model.time
#                     print(f"[FINAL] sentinel-edge at {(r, c)} → "
#                           f"last_valid raw_arrival={prt:.1f}, mtt={mtt_val:.1f}")
#                     return (r, c)
#                 # else skip this NaN/Inf and keep marching
#             r += step_r; c += step_c
#         rv = raw[last_valid]
#         mtt_val = rv - model.time
#         print(f"[FINAL] off-grid cell {last_valid} → "
#               f"raw_arrival={rv:.1f}, mtt={mtt_val:.1f}")
#         return last_valid
#
#     else:
#         # original “perfect” logic
#         # mask out ANY invalid cells
#         clean = np.where(
#             np.logical_or.reduce((
#                 np.isnan(raw),
#                 ~np.isfinite(raw),
#                 raw < 0,
#             )),
#             np.nan,
#             raw
#         )
#         mtt = clean - model.time
#
#         while 0 <= r < R and 0 <= c < C:
#             v = mtt[r, c]
#             if np.isfinite(v) and v >= buffer_minutes:
#                 return (r, c)
#             last_valid = (r, c)
#             r += step_r; c += step_c
#
#         return last_valid


def _buffered_cell_by_mtt(model,
                          boundary_idx: int,
                          buffer_minutes: float
                         ) -> tuple[int, int]:
    """
    March along the *true sector ray* (center → boundary) using a fast
    grid-accurate ray traversal (Amanatides & Woo). This avoids drift at
    non-axial angles (e.g., 60° with 6 sectors).

    If model.wind_schedule is set (truth/forecast run):
      • stop at the first cell with (arrival_time - now) >= buffer_minutes
      • OR if we hit an invalid/sentinel (NaN/Inf/negative) and the last
        valid cell was within 100 min of the sim end → return that sentinel
      • else if we run off-grid → return the last valid cell

    If no wind_schedule:
      • same traversal but with the simpler “first mtt >= buffer” stop.
    """
    raw = model.fire.arrival_time_grid  # full arrival times (R×C)

    # --- raster geometry ---
    dx      = model.fire.transform[0]          # > 0
    dy_abs  = abs(model.fire.transform[4])     # > 0  (rows grow downward)
    x0      = model.fire.bounds.left
    y0      = model.fire.bounds.top

    # --- world coords for center & boundary point of this sector ---
    cx, cy  = model.sector_center.x, model.sector_center.y
    bx, by  = (model.sector_boundaries[boundary_idx].x,
               model.sector_boundaries[boundary_idx].y)

    # direction in world space
    vx, vy  = (bx - cx), (by - cy)

    # start in "grid space" (cell = 1 unit)
    xg = (bx - x0) / dx
    yg = (y0 - by) / dy_abs  # note the sign flip because rows increase downward

    R, C = raw.shape

    # starting cell (clamped in-bounds, consistent with xg/yg floor)
    c = int(np.floor(xg)); r = int(np.floor(yg))
    c = 0 if c < 0 else (C - 1 if c >= C else c)
    r = 0 if r < 0 else (R - 1 if r >= R else r)
    last_valid = (r, c)

    # direction in grid units
    vxg = vx / dx
    vyg = -vy / dy_abs  # minus because rows grow downward

    # handle degenerate case (ray length ~ 0) by returning the start cell
    if vxg == 0 and vyg == 0:
        rv = raw[r, c]
        if np.isfinite(rv) and (rv - model.time) >= buffer_minutes:
            return (r, c)
        return last_valid

    # step signs and parametric increments per crossed cell
    step_c  = 0 if vxg == 0 else (1 if vxg > 0 else -1)
    step_r  = 0 if vyg == 0 else (1 if vyg > 0 else -1)
    inf     = float("inf")
    tDeltaX = inf if vxg == 0 else abs(1.0 / vxg)
    tDeltaY = inf if vyg == 0 else abs(1.0 / vyg)

    # distance (in param t) from (xg,yg) to the first grid boundary
    next_c_edge = (c + 1) if step_c > 0 else c
    next_r_edge = (r + 1) if step_r > 0 else r
    tMaxX = inf if vxg == 0 else abs((next_c_edge - xg) / vxg)
    tMaxY = inf if vyg == 0 else abs((next_r_edge - yg) / vyg)

    sim_end = model.fire_spread_sim_time
    tol     = 100.0  # tolerance for “right at the sim edge” check

    if model.wind_schedule:
        # -------- sentinel-aware logic (truth/forecast runs) ----------
        while 0 <= r < R and 0 <= c < C:
            rv = raw[r, c]
            if np.isfinite(rv) and rv >= 0:
                last_valid = (r, c)
                if (rv - model.time) >= buffer_minutes:
                    return (r, c)
            else:
                # invalid cell → if the last valid cell was near the sim edge, accept
                pr, pc = last_valid
                prt = raw[pr, pc]
                if np.isfinite(prt) and abs(prt - sim_end) <= tol:
                    return (r, c)

            # advance to the next cell the ray actually enters
            if tMaxX <= tMaxY:
                c += step_c
                tMaxX += tDeltaX
            else:
                r += step_r
                tMaxY += tDeltaY

        # off-grid → return last in-bounds
        return last_valid

    else:
        # -------- simpler mtt check without sentinel heuristic ----------
        while 0 <= r < R and 0 <= c < C:
            rv = raw[r, c]
            if np.isfinite(rv):
                last_valid = (r, c)
                if (rv - model.time) >= buffer_minutes:
                    return (r, c)

            if tMaxX <= tMaxY:
                c += step_c
                tMaxX += tDeltaX
            else:
                r += step_r
                tMaxY += tDeltaY

        return last_valid


def _buffered_finish_cell(model,
                          sector_idx: int,
                          buffer_minutes: float) -> tuple[int, int]:
    """
    Push the boundary point outward by `buffer_minutes` worth of spread-time.
    """
    center = model.sector_center
    b      = Point(*model.sector_boundaries[sector_idx])
    ang    = math.atan2(b.y - center.y, b.x - center.x)
    base_d = math.hypot(b.x - center.x, b.y - center.y)
    spread = base_d / model.time if model.time > 0 else model.fire.transform[0]

    fx = b.x + math.cos(ang) * spread * buffer_minutes
    fy = b.y + math.sin(ang) * spread * buffer_minutes
    return world_to_cell(model, fx, fy)




# ──────────────────────────────────────────────────────────────────
# 1)  Always buffer from the *left* boundary of the sector
# ──────────────────────────────────────────────────────────────────
def _buffered_start_cell(model, sector_idx: int, buffer_minutes: int = 500):
    """
    Pick the LEFT-hand sector boundary ((idx-1) mod N) **every time** so that
    subsequent crews see a consistent “start-on-left / finish-on-right”
    convention.  The rest is exactly the old spread-time offset.
    """

    center = model.sector_center
    left_bdy = Point(*model.sector_boundaries[sector_idx % len(model.sector_boundaries)])
    ang = math.atan2(left_bdy.y - center.y, left_bdy.x - center.x)
    base_d = math.hypot(left_bdy.x - center.x, left_bdy.y - center.y)
    spread = base_d / model.time if model.time > 0 else model.fire.transform[0]

    sx = left_bdy.x + math.cos(ang) * spread * buffer_minutes
    sy = left_bdy.y + math.sin(ang) * spread * buffer_minutes
    return world_to_cell(model, sx, sy)






class GroundCrewAgent(mesa.Agent):
    def __init__(self,
                 uid: int,
                 model: "WildfireModel",
                 speed_m_per_min: float,
                 sector_index: int,
                 buffer_dist: float = 1300,
                 line_width: int = 1,
                 buffer_time: float = 1500,
                 finish_extra_buffer: float = 500,
                 planned_start: tuple[int, int] = None,
                 planned_finish: tuple[int, int] = None):
        """
        Constructs a ground-crew agent that always relies on externally provided planned values.
        The agent never computes its own start or finish; it always uses the values provided from the higher-level plan.
        """
        super().__init__(uid, model)
        self.sector_index = sector_index
        # ── keep the original speed so we can rebuild the agent later ──
        self._speed_m_per_min = speed_m_per_min  # <<< NEW line
        self.buffer_dist = buffer_dist
        self.line_width = max(1, line_width)
        self.cell_size = model.fire.transform[0]
        self.buffer_time = buffer_time  # Safety margin (in minutes).
        self.finish_extra_buffer = finish_extra_buffer
        self.walk_ticks = max(1, round(self.cell_size / speed_m_per_min / model.time_step))
        # if planned_start is None or planned_finish is None:
        #     planned_start, planned_finish = plan_start_finish(model, sector_index)

        # if planned_start is None or planned_finish is None:
        #     planned_start, planned_finish, start_bdy, finish_bdy = plan_start_finish(model, sector_index)
        # else:
        #     # If pre-planned targets are provided, assume default boundaries.
        #     start_bdy = sector_index % len(model.sector_boundaries)
        #     finish_bdy = (sector_index + 1) % len(model.sector_boundaries)


        # Clearing-rate map.
        # self.value_map = {
        #     1: 0.019, 2: 0.025, 3: 0.107, 4: 0.186, 5: 0.107,
        #     6: 0.107, 7: 0.107, 8: 0.027, 9: 0.009, 10: 0.075,
        #     11: 0.075, 12: 0.075, 13: 0.186,
        #     91: 0.0, 92: 0.25, 98: 0.0, 99: 0.0,
        #     101: 0.6 / 20, 102: 0.6 / 20, 103: 0.6 / 20, 104: 0.7 / 20,
        #     105: 0.5 / 20, 106: 0.6 / 20,
        #     107: 2.5 / 20, 108: 2.5 / 20, 109: 2.5 / 20,
        #     121: 2.3 / 20, 122: 2.4 / 20, 123: 2.3 / 20, 124: 2.4 / 20,
        #     141: 2.3 / 20, 142: 4.0 / 20, 143: 3.9 / 20, 144: 2.2 / 20,
        #     145: 4.1 / 20, 146: 2.2 / 20, 147: 4.2 / 20, 148: 2.4 / 20, 149: 4.1 / 20,
        #     161: 1.7 / 20, 162: 1.7 / 20, 163: 1.7 / 20, 164: 2.3 / 20, 165: 2.0 / 20,
        #     181: 1.4 / 20, 182: 0.2 / 20, 183: 0.8 / 20, 184: 1.5 / 20, 185: 0.9 / 20,
        #     186: 0.3 / 20, 187: 3.7 / 20, 188: 0.7 / 20, 189: 0.5 / 20,
        #     201: 1.5 / 20, 202: 1.5 / 20, 203: 3.7 / 20, 204: 3.9 / 20,
        # }
        # self.value_map= {
        #     1: 0.157,
        #     2: 0.157,
        #     3: 0.157,
        #     4: 0.298,
        #     5: 0.304,
        #     6: 0.304,
        #     7: 0.373,
        #     8: 0.216,
        #     9: 0.216,
        #     10: 0.216,
        #     11: 0.099,
        #     12: 0.213,
        #     13: 0.298,
        #     91: 0.000,
        #     92: 0.250,
        #     93: 0.000,
        #     98: 0.000,
        #     99: 0.000,
        #     101: 0.157,
        #     102: 0.157,
        #     103: 0.157,
        #     104: 0.157,
        #     105: 0.157,
        #     106: 0.157,
        #     107: 0.157,
        #     108: 0.157,
        #     109: 0.157,
        #     121: 0.157,
        #     122: 0.157,
        #     123: 0.157,
        #     124: 0.157,
        #     141: 0.304,
        #     142: 0.304,
        #     143: 0.304,
        #     144: 0.304,
        #     145: 0.304,
        #     146: 0.298,
        #     147: 0.298,
        #     148: 0.298,
        #     149: 0.298,
        #     161: 0.216,
        #     162: 0.216,
        #     163: 0.216,
        #     164: 0.216,
        #     165: 0.216,
        #     166: 0.216,
        #     181: 0.216,
        #     182: 0.216,
        #     183: 0.216,
        #     184: 0.216,
        #     185: 0.216,
        #     186: 0.216,
        #     187: 0.216,
        #     188: 0.216,
        #     189: 0.216,
        #     201: 0.099,
        #     202: 0.213,
        #     203: 0.298,
        # }
        # self.value_map = BASE_CLEAR_RATE
        # apply the model’s clear-rate multiplier (dividing time per cell)
        mult = getattr(model, "GC_CLEAR_RATE_MULTIPLIER", 1.0)
        self.value_map = {
            fid: rate / mult
            for fid, rate in BASE_CLEAR_RATE.items()
        }

        self._walk_tick = 0   # Ticks accumulated while walking.
        self._ticks_left = 0  # Ticks remaining for clearing the current cell.
        self.cleared_batch: list[tuple[int, int]] = []  # Batch of cells to be cleared.

        # ALWAYS use the externally provided planned values.
        # These must be supplied by the higher-level plan.
        # self.planned_start = planned_start
        # self.planned_finish = planned_finish
        self.planned_start = planned_start
        self.planned_finish = planned_finish
        # self.start_boundary = start_bdy
        # self.finish_boundary = finish_bdy

        self.start_boundary = None
        self.finish_boundary = None



        # For the agent’s progress we store the current “start cell”
        self.start_rc = self.planned_start
        self.path: list[tuple[int, int]] = []  # Fireline path.
        self._eta_remaining = 0    # (minutes) updated every re‑plan
        self._path_cell_times = []
        self.state = "ToStart"  # Initial state: head to start.
        self.start_time_abs = None  # wall-clock minute when we began BuildLine
        self.last_cell_idx = -1  # index of last cell already walked/cleared
        self.path_times = []  # cumulative minutes for every cell in self.path

        self.aborted_fire_intercept = False
        # Spawn the agent roughly at the center of the fire grid (for visibility).
        cx = (model.fire.bounds.left + model.fire.bounds.right) / 2
        cy = (model.fire.bounds.top + model.fire.bounds.bottom) / 2
        self.grid_rc = world_to_cell(model, cx, cy)
        self.position = cell_to_world(model, *self.grid_rc)
        model.space.place_agent(self, self.position)
        self.previous_positions = []

        self.replan_disabled = False  # ← new: allow re‑plans by default
        print(f"[GC#{uid:02d}] spawned at cell {self.grid_rc} with walk_ticks={self.walk_ticks}")

        # ─────────────────────────────────────────────────────────────────────
        # DEEP CLONE_
        # ─────────────────────────────────────────────────────────────────────

    def deep_clone(self, new_model: "WildfireModel") -> "GroundCrewAgent":
        """
        Re-create the agent on <new_model> **without** touching any GPU data.
        Only plain-Python objects are copied.
        """
        clone = GroundCrewAgent(
            uid=self.unique_id,
            model=new_model,
            speed_m_per_min=self._speed_m_per_min,
            sector_index=self.sector_index,
            buffer_dist=self.buffer_dist,
            line_width=self.line_width,
            buffer_time=self.buffer_time,
            finish_extra_buffer=self.finish_extra_buffer,
            planned_start=self.planned_start,
            planned_finish=self.planned_finish,
        )

        # ---------- runtime state ---------------------------------------
        clone.state = self.state
        clone.start_rc = self.start_rc
        clone.grid_rc = tuple(self.grid_rc)
        clone.position = tuple(self.position)
        clone.path = list(self.path)
        clone._walk_tick = self._walk_tick
        clone._ticks_left = self._ticks_left
        clone._eta_remaining = self._eta_remaining
        clone.cleared_batch = list(self.cleared_batch)
        clone.previous_positions = list(self.previous_positions)
        clone.replan_disabled = self.replan_disabled
        clone.start_time_abs = self.start_time_abs
        clone.last_cell_idx  = self.last_cell_idx
        clone.path_times     = list(self.path_times)
        clone.start_boundary = self.start_boundary  # NEW
        clone.finish_boundary = self.finish_boundary  # NEW
        clone.aborted_fire_intercept = self.aborted_fire_intercept
        # make sure the agent is on the new space grid
        new_model.space.move_agent(clone, clone.position)

        return clone

        # ─────────────────────────────────────────────────────────────────────
        #  NEW 1-liner helper so the model can re-task the crew later
        # ─────────────────────────────────────────────────────────────────────

    def assign_to_sector(self, sector_idx: int):
        """(Re)assign this crew to <sector_idx> and perform fresh planning."""
        self.sector_index = sector_idx
        # self.planned_start, self.planned_finish = plan_start_finish(
        #     self.model, sector_idx)
        # self.planned_start, self.planned_finish, start_bdy, finish_bdy = plan_start_finish(self.model, sector_idx)
        self.planned_start, self.planned_finish, start_bdy, finish_bdy = \
            plan_start_finish(self.model, sector_idx, origin_rc=self.grid_rc)

        self.start_boundary = start_bdy
        self.finish_boundary = finish_bdy

        self.start_rc = self.planned_start
        self.state = "ToStart"  # reset mini-FSM
        self.replan_disabled = False
        # NEW: clear any legacy “stopped-on-fire” state from the prior mission.
        self.aborted_fire_intercept = False

        if hasattr(self.model, "_register_first_pair"):
            self.model._register_first_pair(self)


        print(f"[GC#{self.unique_id:02d}] → sector {sector_idx}  "
              f"start {self.planned_start}  finish {self.planned_finish}")

    def _print_start_origin(self):
        """
        Print whether the current start_rc is a NEW buffered start or reusing
        an existing anchor. Uses the audit log if present; otherwise falls back
        to a completed-anchor check on the two sector boundaries.
        """
        M = self.model
        start = tuple(self.start_rc)

        # If this build was triggered by a replan, label it explicitly.
        if getattr(self, "_replan_label", None) == "replan":
            print(f"[START] GC#{self.unique_id:02d} using AD-HOC (replan) start {start}")
            return

        # Prefer using the anchor log (if you added _anchor_log_event before)
        kind = None
        if hasattr(M, "_anchor_log"):
            for e in reversed(M._anchor_log):
                if (e.get("action") == "set" and e.get("role") == "start"
                        and e.get("bdy") in (
                                self.sector_index % len(M.sector_boundaries),
                                (self.sector_index + 1) % len(M.sector_boundaries)
                        )
                        and tuple(e.get("cell") or ()) == start):
                    by = e.get("by", "")
                    if isinstance(by, str) and "new" in by:
                        kind = "NEW (buffered by planner)"
                    elif isinstance(by, str) and "reuse" in by:
                        kind = "REUSED (existing anchor)"
                    else:
                        kind = "ANCHOR (unspecified source)"
                    break

        # Fallback if no log match: was this start one of the completed anchors?
        if kind is None:
            N = len(M.sector_boundaries)
            left = self.sector_index % N
            right = (self.sector_index + 1) % N
            lcell, _, _ = _get_most_recent_completed_anchor(M, left)
            rcell, _, _ = _get_most_recent_completed_anchor(M, right)
            if (lcell is not None and tuple(lcell) == start) or (rcell is not None and tuple(rcell) == start):
                kind = "REUSED (completed anchor)"
            else:
                kind = "NEW (buffered by planner)"

        print(f"[START] GC#{self.unique_id:02d} using {kind} → start {start} on boundary {self.start_boundary}")

    def flush_clear_buffer(self) -> None:
        """
        Ensure any cells waiting in the write-behind buffer
        (cleared_batch) are committed to the fire model before
        the simulation is cloned.
        """
        self._flush_batch()  # ← uses your cleared_batch list

    def update_planned_targets(self, planned_start: tuple[int, int], planned_finish: tuple[int, int]) -> None:
        """
        Update the planned start and finish targets.
        This method is invoked by the model when the higher-level plan is refreshed.
        """
        self.planned_start = planned_start
        self.planned_finish = planned_finish
        # Also update the start target if not already departed.
        # if self.state == "ToStart":
        #     self.start_rc = planned_start
        self.start_rc = planned_start

    def _debug_start_cell_status(self):
        M = self.model
        r, c = self.grid_rc
        raw = M.fire.arrival_time_grid
        raw_rc = float(raw[r, c])
        mtt_rc = float(raw_rc - M.time) if np.isfinite(raw_rc) else np.nan
        buf = 120  # must match firefighter_fire_buffer_time passed to pathfinder
        print(
            f"[DEBUG] GC#{self.unique_id:02d} start@{(r, c)}  "
            f"raw_arr={raw_rc:.1f}  now={M.time:.1f}  mtt={mtt_rc:.1f}  "
            f"within_buffer={(np.isfinite(mtt_rc) and mtt_rc <= buf)}"
        )

    # ─────────────────────────────────────────────────────────────────────
    #  Full, self-contained replacement for GroundCrewAgent.recalc_path_from_here
    # ─────────────────────────────────────────────────────────────────────
    def recalc_path_from_here(self) -> None:
        """
        Attempt to rebuild the fire-line starting from the crew’s *current* cell.
        The stopwatch (`start_time_abs`) is only reset if the new path is
        accepted; otherwise every runtime field is rolled back to its pre-
        attempt state and further re-plans are disabled.
        """
        # Only allow re-plans while actively building line

        if self.state in ("Standby", "ToStart"):
            print(f"[DEBUG] GC#{self.unique_id:02d}  re-plan skipped (state={self.state})")
            return

        if self.replan_disabled:
            print(f"[DEBUG] GC#{self.unique_id:02d}  re-plan skipped "
                  f"(disabled after previous failure)")
            return

        # ── 1) snapshot current state so we can undo cleanly ───────────
        old_path         = self.path.copy()
        old_path_times   = self.path_times.copy()
        old_last_idx     = self.last_cell_idx
        old_start_abs    = self.start_time_abs
        old_start_rc     = self.start_rc
        old_eta          = self._eta_remaining
        old_len          = len(old_path)

        print(f"[DEBUG] GC#{self.unique_id:02d}  attempting re-plan… "
              f"remaining path = {old_len} cells")

        # ── 2) set up a provisional new start (do NOT touch the clock yet)
        self.start_rc      = self.grid_rc
        self.planned_start = self.grid_rc
        self.last_cell_idx = -1

        # ── 3) try to build the new path; on any exception roll back ——─
        try:
            self._debug_start_cell_status()

            self._build_fireline_path()       # fills self.path & self.path_times
        except Exception as exc:
            print(f"[ERROR] GC#{self.unique_id:02d}  re-plan failed: {exc}")
            self.path            = old_path
            self.path_times      = old_path_times
            self.last_cell_idx   = old_last_idx
            self.start_time_abs  = old_start_abs
            self.start_rc        = old_start_rc
            self._eta_remaining  = old_eta
            self.replan_disabled = True
            return

        new_len = len(self.path)

        # --- EARLY GUARD: empty path → rollback, DO NOT disable re-plans ---
        # This covers the “truncated at idx 0” case that yields no cells after
        # the start cell is dropped, and avoids IndexError on self.path_times[-1].
        if new_len == 0 or len(self.path_times) == 0:
            print(f"[DEBUG] GC#{self.unique_id:02d}  re-plan produced empty path "
                  f"(likely truncated at start) → keeping old path")
            self.path = old_path
            self.path_times = old_path_times
            self.last_cell_idx = old_last_idx
            self.start_time_abs = old_start_abs
            self.start_rc = old_start_rc
            self._eta_remaining = old_eta
            # NOTE: replan_disabled stays False on purpose
            return

        # ── 4) sanity check on path length / trivial cases ─────────────
        if (new_len > old_len + 5
                or self.start_rc == self.planned_finish
                or old_len < 10):
            print(f"[DEBUG] GC#{self.unique_id:02d}  new path length "
                  f"{new_len} > old+5  →  rejecting and DISABLING further re-plans")
            self.path            = old_path
            self.path_times      = old_path_times
            self.last_cell_idx   = old_last_idx
            self.start_time_abs  = old_start_abs
            self.start_rc        = old_start_rc
            self._eta_remaining  = old_eta
            self.replan_disabled = True
            return

        # ── 5) ACCEPT: start the stopwatch for the new line ────────────
        self.start_time_abs  = self.model.time
        self._eta_remaining  = self.path_times[-1]
        self.replan_disabled = False
        print(f"[DEBUG] GC#{self.unique_id:02d}  new path accepted "
              f"(length {new_len}, ETA {self._eta_remaining:.1f} min)")

    def _emit_debug_plots(self, reason: str = "IndexError@ETA") -> None:
        """
        Dump MTT debug plots (and a full-path overlay) just like the commented
        blocks in _build_fireline_path(), then return. Any failures here should
        never mask the original exception.
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from pathlib import Path

            M, fire = self.model, self.model.fire

            # --- Grids identical to your commented code ---
            raw_mtt = fire.arrival_time_grid
            mtt = np.where(np.isnan(raw_mtt), np.nan,
                           np.clip(raw_mtt - M.time, 0, None))
            fuel = fire.fuel_model.astype(int)
            raw_mtt_masked = np.where(raw_mtt > 3000, np.nan, raw_mtt)
            mtt_masked = np.where(mtt > (3000 - M.time), np.nan, mtt)

            R, C = mtt_masked.shape
            sr, sc = (self.start_rc or (0, 0))
            fr, fc = (self.planned_finish or (0, 0))

            out = Path(getattr(M, "case_folder", ".")) / "debug"
            out.mkdir(parents=True, exist_ok=True)

            # Overlays: zero-time, start/finish
            overlay_zero = np.zeros((R, C, 4), dtype=float)
            zero_locs = (mtt_masked == 0)
            overlay_zero[zero_locs, 0] = 1.0  # red
            overlay_zero[zero_locs, 3] = 1.0  # opaque

            overlay_sf = np.zeros((R, C, 4), dtype=float)
            if 0 <= sr < R and 0 <= sc < C:
                overlay_sf[sr, sc] = [0.0, 0.0, 0.0, 1.0]  # start = black
            if 0 <= fr < R and 0 <= fc < C:
                overlay_sf[fr, fc] = [1.0, 0.0, 1.0, 1.0]  # finish = magenta

            # ── DEBUG PLOT #1: raw_mtt vs mtt_masked ──
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 14))
            im0 = ax0.imshow(raw_mtt_masked, origin='upper', vmin=0, vmax=3000)
            ax0.set_title(f'GC#{self.unique_id:02d}  raw_mtt (≤3000)  [{reason}]')
            plt.colorbar(im0, ax=ax0, label='arrival time (min)')

            im1 = ax1.imshow(mtt_masked, origin='upper')
            ax1.set_title(f'GC#{self.unique_id:02d}  mtt (t={M.time:.1f} min)  [{reason}]')
            plt.colorbar(im1, ax=ax1, label='minutes until fire')
            ax1.imshow(overlay_zero, origin='upper')
            ax1.imshow(overlay_sf, origin='upper')

            fname1 = out / f"gc{self.unique_id:02d}_mtt_debug_{int(M.time)}.png"
            fig.savefig(fname1, dpi=600, bbox_inches='tight')
            plt.close(fig)

            # ── DEBUG PLOT #2: FULL path overlay BEFORE truncation ──
            # Recreate the feasibility & run pathfinder exactly like _build_fireline_path
            feasibility = np.ones((R, C), dtype=bool)
            sectors = M.sector_angle_ranges  # list[(lo, hi)]
            mids = [((lo + hi) * 0.5) % 360 for (lo, hi) in sectors]
            cur_mid = mids[self.sector_index]
            opp_angle = (cur_mid + 180.0) % 360

            def _ang_dist(a, b):  # smallest angular distance
                return abs(((a - b + 180) % 360) - 180)

            opp_idx = int(np.argmin([_ang_dist(m, opp_angle) for m in mids]))
            dx, dy = fire.transform[0], fire.transform[4]  # dy < 0
            xs = fire.bounds.left + (np.arange(C) + 0.5) * dx
            ys = fire.bounds.top + (np.arange(R) + 0.5) * dy
            xs, ys = np.meshgrid(xs, ys)
            cx, cy = M.sector_center.x, M.sector_center.y
            ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360) % 360

            lo, hi = sectors[opp_idx]
            if lo < hi:
                in_sector = (ang >= lo) & (ang < hi)
            else:
                in_sector = (ang >= lo) | (ang < hi)
            feasibility[in_sector] = False

            crews_here = max(1, sum(
                1 for ag in M.schedule.agents
                if isinstance(ag, GroundCrewAgent)
                and ag.sector_index == self.sector_index
                and ag.state in ("BuildLine", "ToStart")
            ))
            adj_value_map = (
                {k: v / crews_here for k, v in self.value_map.items()}
                if crews_here > 1 else self.value_map
            )
            fuel2 = np.where(fuel < 0, 102, fuel)

            # If start/finish are missing, skip the full-path overlay quietly.
            if self.start_rc is not None and self.planned_finish is not None:
                result = firelinepath.fireline_between_two_points(
                    start=np.array(self.start_rc, int),
                    mtt=mtt,
                    fuel=fuel2,
                    fuel_clear_cost=adj_value_map,
                    feasibility=feasibility,
                    finish=np.array(self.planned_finish, int),
                    clear_burning_penalty=1000000.0,
                    distance_penalty=0.001,
                    firefighter_fire_buffer_time=120,
                )

                full_path = [tuple(map(int, rc)) for rc, _ in result]
                overlay_full = np.zeros((R, C, 4), dtype=float)
                for r, c in full_path:
                    if 0 <= r < R and 0 <= c < C:
                        overlay_full[r, c] = [0.0, 1.0, 1.0, 1.0]  # cyan

                fig_full, axf = plt.subplots(figsize=(12, 10))
                imf = axf.imshow(mtt_masked, origin='upper')
                axf.set_title(f'GC#{self.unique_id:02d}  FULL path @ t={M.time:.1f}  [{reason}]')
                plt.colorbar(imf, ax=axf, label='minutes until fire')
                axf.imshow(overlay_zero, origin='upper')
                axf.imshow(overlay_full, origin='upper')
                axf.imshow(overlay_sf, origin='upper')

                fname2 = out / f"gc{self.unique_id:02d}_path_full_debug_{int(M.time)}.png"
                fig_full.savefig(fname2, dpi=600, bbox_inches='tight')
                plt.close(fig_full)

            print(f"[DEBUG PLOTS] saved to: {out}")
        except Exception as e:
            print(f"[DEBUG PLOTS] failed: {e}")

    def step(self):
        # print("Time:", self.model.time)
        # ── 1) walk to start cell (old logic, tick-based) ────────────────
        if self.state == "ToStart":
            if self.start_rc is None:
                self.start_rc = self.planned_start
            if self._walk_one_step(self.start_rc):
                self._build_fireline_path()
                _mark_anchor_completed(self.model,
                                       self.start_boundary,
                                       "start")

                # Optimistically mark FINISH complete at the moment we start building…
                _mark_anchor_completed(self.model, self.finish_boundary, "finish", by=self)

                # …and ONLY if we detected truncation, flip it back to not completed.
                if getattr(self, "aborted_fire_intercept", False):
                    _mark_anchor_uncompleted(self.model, self.finish_boundary, "finish", by=self)
                self.state = "BuildLine"
            return

        # ── 2) follow the fire-line driven purely by elapsed time ────────
        if self.state == "BuildLine":
            if self.start_time_abs is None:          # first entry only
                self.start_time_abs = self.model.time

            elapsed = self.model.time - self.start_time_abs

            # advance along the path as far as today’s elapsed minutes allow
            while (self.last_cell_idx + 1 < len(self.path_times) and
                   self.path_times[self.last_cell_idx + 1] <= elapsed):
                self.last_cell_idx += 1
                rc = self.path[self.last_cell_idx]
                self._move_to(rc)
                self._schedule_clearing_for(rc)

            # update live ETA
            # self._eta_remaining = max(0.0, self.path_times[-1] - elapsed)
            # update live ETA (plot & re-raise if path_times is empty)
            try:
                self._eta_remaining = max(0.0, self.path_times[-1] - elapsed)
            except IndexError:
                # Emit the exact plots you wanted, then break the run as before.
                self._emit_debug_plots("IndexError on self.path_times[-1] in step()")
                raise

            # finished?
            # if self.last_cell_idx >= len(self.path) - 1:
            #     self._flush_batch()
            #     self.state = "Standby"
            #     self.replan_disabled = False
            #     self.model.contained_sectors.add(self.sector_index)
            #     print("Sector contained ", self.sector_index)
            #     # >>> reset per-line timers & progress pointers
            #     _mark_anchor_completed(self.model, self.start_boundary, "start")
            #     _mark_anchor_completed(self.model, self.finish_boundary, "finish")
            #     self.start_time_abs = None
            #     self.last_cell_idx = -1
            #     self.path.clear()
            #     self.path_times.clear()
            #     # <<<
            #
            #     print(f"[GC#{self.unique_id:02d}] Sector {self.sector_index} contained → Standby")
            # return
            # finished? (we've walked/cleared every cell in self.path)
            if self.last_cell_idx >= len(self.path) - 1:
                self._flush_batch()
                self.state = "Standby"
                self.replan_disabled = False

                if not self.aborted_fire_intercept:
                    # full line built all the way to planned_finish
                    self.model.contained_sectors.add(self.sector_index)
                    print("Sector contained ", self.sector_index)
                    _mark_anchor_completed(self.model, self.start_boundary, "start")
                    _mark_anchor_completed(self.model, self.finish_boundary, "finish")
                    print(f"[GC#{self.unique_id:02d}] Sector {self.sector_index} contained → Standby")
                else:
                    # stopped early because fire was encountered
                    print(f"[GC#{self.unique_id:02d}] stopped early (fire intercept) – "
                          f"sector {self.sector_index} NOT contained; awaiting reassignment.")

                # reset per-line timers & progress pointers
                self.start_time_abs = None
                self.last_cell_idx = -1
                self.path.clear()
                self.path_times.clear()
                return

    def _walk_one_step(self, target_rc: tuple[int, int]) -> bool:
        if self.grid_rc == target_rc:
            return True
        self._walk_tick += 1
        if self._walk_tick < self.walk_ticks:
            return False
        self._walk_tick = 0
        r, c = self.grid_rc
        tr, tc = target_rc
        new_r = r + (tr > r) - (tr < r)
        new_c = c + (tc > c) - (tc < c)
        self._move_to((new_r, new_c))
        return self.grid_rc == target_rc

    def _move_to(self, rc: tuple[int, int]) -> None:
        self.grid_rc = tuple(rc)
        self.position = cell_to_world(self.model, *self.grid_rc)
        self.previous_positions.append(self.position)
        self.model.space.move_agent(self, self.position)


    def _schedule_clearing_for(self, centre_rc: tuple[int, int]) -> None:
        # 1) queue the cells exactly as before
        for r, c in self._cells_within_width(centre_rc, self.line_width):
            if self.model.fire.fuel_model[r, c] != CLEARED_ID:
                self.cleared_batch.append((r, c))

        # 2) look up the base clearing time for the centre cell
        r, c = centre_rc
        fuel = int(self.model.fire.fuel_model[r, c])
        hours = self.value_map.get(fuel, 0.1)

        # 3) count *all* ground crews that are working in the same sector
        crews_here = sum(
            1 for ag in self.model.schedule.agents
            if isinstance(ag, GroundCrewAgent)
            and ag.sector_index == self.sector_index
            and ag.state in ("BuildLine", "ToStart")  # already walking / working
        ) or 1  # safety: never divide by 0

        # 4) share the work: divide the time by the crew count
        hours_per_crew = hours / crews_here

        # 5) convert to ticks
        self._ticks_left = max(
            1,
            math.floor(hours_per_crew * 60 / self.model.time_step)
        )

        # 6) flush if the batch reached the threshold
        if len(self.cleared_batch) >= BATCH_SIZE:
            self._flush_batch()

    def _cells_within_width(self, rc: tuple[int, int], w: int) -> list[tuple[int, int]]:
        r0, c0 = rc
        R, C = self.model.fire.fuel_model.shape
        out = []
        for dr in range(-w, w + 1):
            for dc in range(-w, w + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < R and 0 <= c < C:
                    out.append((r, c))
        return out

    def _flush_batch(self) -> None:
        if not self.cleared_batch:
            return
        fire = self.model.fire
        for r, c in self.cleared_batch:
            fire.fuel_model[r, c] = CLEARED_ID
        self.cleared_batch.clear()
        fire.re_run(fire.fuel_model)
        print(f"[GC#{self.unique_id:02d}] updated fire after clearing batch")

    # to combine the run with the drop
    def flush_pending_to_fuel_no_rerun(self) -> int:
        """
        Push queued cleared cells into fire.fuel_model WITHOUT calling re_run.
        Returns how many cells were committed, and clears the buffer (resets counter).
        """

        if not self.cleared_batch:
            return 0
        fire = self.model.fire
        for r, c in self.cleared_batch:
            fire.fuel_model[r, c] = CLEARED_ID
        n = len(self.cleared_batch)
        self.cleared_batch.clear()
        print('FLUSHED')
        return n

    # def _build_fireline_path(self, from_rc: tuple[int, int] | None = None) -> None:
    #     """Generates self.path **and** self.path_times (cumulative minutes)."""
    #     M, fire = self.model, self.model.fire
    #
    #     raw_mtt = fire.arrival_time_grid
    #     mtt = np.where(np.isnan(raw_mtt), np.nan, np.clip(raw_mtt - M.time, 0, None))
    #     fuel = fire.fuel_model.astype(int)
    #     # feasibility = np.loadtxt("feasibility_1_converted.csv", delimiter=",", dtype=bool)
    #
    #
    #     #############################################################################################################
    #     #############################################################################################################
    #     #############################################################################################################
    #     import matplotlib.pyplot as plt  # local to avoid circulars
    #
    #     # ── 2) build DYNAMIC feasibility map ────────────────────────────
    #     R, C = fuel.shape
    #     feasibility = np.ones((R, C), dtype=bool)  # start: all 1s
    #
    #     # a) angular mid-point of every sector
    #     sectors = M.sector_angle_ranges  # list[(low, high)]
    #     mids = [((lo + hi) * 0.5) % 360 for (lo, hi) in sectors]
    #     n_sec = len(mids)
    #
    #     # b) current sector’s mid-angle + 180°
    #     cur_mid = mids[self.sector_index]
    #     opp_angle = (cur_mid + 180.0) % 360
    #
    #     # c) find the sector whose mid-angle is closest to opp_angle
    #     def _ang_dist(a, b):
    #         """smallest signed distance in ° on a circle"""
    #         return abs(((a - b + 180) % 360) - 180)
    #
    #     opp_idx = int(np.argmin([_ang_dist(m, opp_angle) for m in mids]))
    #
    #     # d) compute angle of every cell (vectorised)
    #     dx, dy = fire.transform[0], fire.transform[4]  # dy < 0
    #     xs = fire.bounds.left + (np.arange(C) + 0.5) * dx
    #     ys = fire.bounds.top + (np.arange(R) + 0.5) * dy
    #     xs, ys = np.meshgrid(xs, ys)  # shape (R, C)
    #     cx, cy = M.sector_center.x, M.sector_center.y
    #     ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360) % 360
    #
    #     # e) mask cells whose angle lives in the OPPOSITE sector’s span
    #     lo, hi = sectors[opp_idx]
    #     if lo < hi:
    #         in_sector = (ang >= lo) & (ang < hi)
    #     else:  # wrap-around case (e.g. 300°–360°–20°)
    #         in_sector = (ang >= lo) | (ang < hi)
    #     feasibility[in_sector] = False  # 0s in opposite wedge
    #
    #     # f) quick visual sanity-check
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(feasibility, cmap="gray_r", origin="upper")
    #     plt.title(f"Feasibility – GC#{self.unique_id:02d}  "
    #               f"sector {self.sector_index}  (opposite = {opp_idx})")
    #     plt.axis("off")
    #     plt.show()
    #
    #     #############################################################################################################
    #     #############################################################################################################
    #     #############################################################################################################
    #
    #
    #     finish_rc = self.planned_finish
    #     crews_here = max(1, _active_crews(self.model, self.sector_index))
    #     adj_value_map = (
    #         {k: v / crews_here for k, v in self.value_map.items()}
    #         if crews_here > 1 else self.value_map
    #     )
    #     print("Start ", self.planned_start)
    #     print("Finish ", self.planned_finish)
    #
    #     result= firelinepath.fireline_between_two_points(
    #         start=np.array(self.start_rc, int),
    #         mtt=mtt,
    #         fuel=fuel,
    #         fuel_clear_cost=adj_value_map,
    #         feasibility=feasibility,
    #         finish=np.array(self.planned_finish, int),
    #         clear_burning_penalty=1000000.0,
    #         distance_penalty=0.001,
    #         firefighter_fire_buffer_time= 60
    #     )
    #     pretty_print_fireline_result(result)  # ← quick console table
    #     # ▸ split the Rust result into geometry & cumulative minutes
    #     self.path, self.path_times = [], []
    #     for cell_rc, metrics in result:
    #         self.path.append(tuple(cell_rc))          # row/col
    #         self.path_times.append(metrics[3])        # cumulative minutes
    #
    #     if self.path and self.path[0] == self.start_rc:
    #         self.path.pop(0); self.path_times.pop(0)
    #
    #     self.last_cell_idx = -1                      # reset progress pointer
    #     self._eta_remaining = self.path_times[-1]    # total minutes remaining
    #     print(f"[GC#{self.unique_id:02d}] fire-line length {len(self.path)} cells "
    #           f"({self._eta_remaining:.1f} min)")

    def _build_fireline_path(self, from_rc: tuple[int, int] | None = None) -> None:
        """
        Build a fireline path from self.start_rc to self.planned_finish.

        If any returned cell from firelinepath has burn_penalty > 0, truncate
        the path *before* that cell and mark self.aborted_fire_intercept = True.
        Sector will NOT be marked contained when we finish walking this path.
        """
        M, fire = self.model, self.model.fire

        # --- grids ----------------------------------------------------
        raw_mtt = fire.arrival_time_grid
        mtt = np.where(np.isnan(raw_mtt), np.nan,
                       np.clip(raw_mtt - M.time, 0, None))
        fuel = fire.fuel_model.astype(int)
        raw_mtt_masked = np.where(raw_mtt > 3000, np.nan, raw_mtt)
        mtt_masked = np.where(mtt > (3000 - M.time), np.nan, mtt)

        # ### DEBUG PLOTS: raw_mtt_masked vs mtt with start/finish ###
        # ######################################################################################################################
        # ######################################################################################################################
        # ######################################################################################################################
        # R, C = mtt_masked.shape
        # sr, sc = self.start_rc
        # fr, fc = self.planned_finish
        # import matplotlib.pyplot as plt
        # from pathlib import Path
        # # Pre‐build overlays for zero‐time, start, finish
        # overlay_zero = np.zeros((R, C, 4), dtype=float)
        # zero_locs = (mtt_masked == 0)
        # overlay_zero[zero_locs, 0] = 1.0  # red
        # overlay_zero[zero_locs, 3] = 1.0  # opaque
        #
        # overlay_sf = np.zeros((R, C, 4), dtype=float)
        # # START → black
        # overlay_sf[sr, sc] = [0.0, 0.0, 0.0, 1.0]
        # # FINISH → pink
        # overlay_sf[fr, fc] = [1.0, 0.0, 1.0, 1.0]
        #
        # # ── DEBUG PLOT #1: raw_mtt vs mtt_masked ────────────────────────
        # fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 14))
        #
        # # raw arrival-time
        # im0 = ax0.imshow(raw_mtt_masked, origin='upper', vmin=0, vmax=3000)
        # ax0.set_title(f'GC#{self.unique_id:02d}  raw_mtt (≤3000)')
        # plt.colorbar(im0, ax=ax0, label='arrival time (min)')
        #
        # # masked MTT
        # im1 = ax1.imshow(mtt_masked, origin='upper')
        # ax1.set_title(f'GC#{self.unique_id:02d}  mtt (t={M.time:.1f} min)')
        # plt.colorbar(im1, ax=ax1, label='minutes until fire')
        #
        # # overlay zero-time and start/finish
        # ax1.imshow(overlay_zero, origin='upper')
        # ax1.imshow(overlay_sf, origin='upper')
        #
        # out =  Path(self.model.case_folder) / "debug"
        # out.mkdir(exist_ok=True)
        # fname1 = out / f"gc{self.unique_id:02d}_mtt_debug_{int(M.time)}.png"
        # fig.savefig(fname1, dpi=600, bbox_inches='tight')
        # plt.close(fig)
        # ## end DEBUG MTT ###
        # ######################################################################################################################
        # ######################################################################################################################
        # ######################################################################################################################





        # --- dynamic feasibility mask (same logic as before) ----------
        R, C = fuel.shape
        feasibility = np.ones((R, C), dtype=bool)

        sectors = M.sector_angle_ranges  # list[(lo, hi)]
        mids = [((lo + hi) * 0.5) % 360 for (lo, hi) in sectors]
        cur_mid = mids[self.sector_index]
        opp_angle = (cur_mid + 180.0) % 360

        def _ang_dist(a, b):
            return abs(((a - b + 180) % 360) - 180)

        opp_idx = int(np.argmin([_ang_dist(m, opp_angle) for m in mids]))

        dx, dy = fire.transform[0], fire.transform[4]  # dy < 0
        xs = fire.bounds.left + (np.arange(C) + 0.5) * dx
        ys = fire.bounds.top + (np.arange(R) + 0.5) * dy
        xs, ys = np.meshgrid(xs, ys)
        cx, cy = M.sector_center.x, M.sector_center.y
        ang = (np.degrees(np.arctan2(ys - cy, xs - cx)) + 360) % 360

        lo, hi = sectors[opp_idx]
        if lo < hi:
            in_sector = (ang >= lo) & (ang < hi)
        else:
            in_sector = (ang >= lo) | (ang < hi)
        feasibility[in_sector] = False

        # --- adjust clear rates for crew sharing ----------------------
        crews_here = max(1, _active_crews(self.model, self.sector_index))
        adj_value_map = (
            {k: v / crews_here for k, v in self.value_map.items()}
            if crews_here > 1 else self.value_map
        )
        fuel = fire.fuel_model.astype(int)
        fuel = np.where(fuel < 0, 102, fuel)  # or whatever valid ID you’ve defined
        # --- run Rust pathfinder --------------------------------------
        self._print_start_origin()

        result = firelinepath.fireline_between_two_points(
            start=np.array(self.start_rc, int),
            mtt=mtt,
            fuel=fuel,
            fuel_clear_cost=adj_value_map,
            feasibility=feasibility,
            finish=np.array(self.planned_finish, int),
            clear_burning_penalty=1000000.0,
            distance_penalty=0.001,
            firefighter_fire_buffer_time=120,
        )

        #print results in a pretty way
        # pretty_print_fireline_result(result, precision=2, show_header=True)
        print("Start: ", self.start_rc)
        print("Finish: ", self.planned_finish)
        ### DEBUG PLOTS: overlay the path on mtt ###


        ######################################################################################################################
        ######################################################################################################################
        ######################################################################################################################

        # # ── DEBUG PLOT: full path BEFORE truncation ─────────────────────
        # # build a list of every cell in the un-truncated result
        # full_path = [tuple(map(int, rc)) for rc, _ in result]
        #
        # # make the same overlay RGBA for the full path
        # overlay_full = np.zeros((R, C, 4), dtype=float)
        # for r, c in full_path:
        #     overlay_full[r, c] = [0.0, 1.0, 1.0, 1.0]  # cyan
        #
        # # now plot it exactly like Plot #2 but saving under a different name:
        # fig_full, axf = plt.subplots(figsize=(12, 10))
        # imf = axf.imshow(mtt_masked, origin='upper')
        # axf.set_title(f'GC#{self.unique_id:02d}  FULL path @ t={M.time:.1f}')
        # plt.colorbar(imf, ax=axf, label='minutes until fire')
        #
        # # layering: mtt → zero‐time → full‐path → start/finish
        # axf.imshow(overlay_zero, origin='upper')
        # axf.imshow(overlay_full, origin='upper')
        # axf.imshow(overlay_sf, origin='upper')
        #
        # pth_full = out / f"gc{self.unique_id:02d}_path_full_debug_{int(M.time)}.png"
        # fig_full.savefig(pth_full, dpi=600, bbox_inches='tight')
        # plt.close(fig_full)
        # # ── end full‐path debug plot ────────────────────────────────────
        # ### DEBUG PLOTS: overlay the path on mtt ###
        ######################################################################################################################
        ######################################################################################################################
        ######################################################################################################################

        # --- scan for fire intercept ----------------------------------
        first_fire_idx = None
        for i, (_rc, metrics) in enumerate(result):
            if float(metrics[1]) > 0.0:  # burn_penalty column
                first_fire_idx = i
                break

        if first_fire_idx is not None:
            safe_upto = max(0, first_fire_idx - 1)
            trimmed = result[:safe_upto + 1]
            fire_rc = tuple(map(int, result[first_fire_idx][0]))
            print(f"[GC#{self.unique_id:02d}] WARNING: fire intercept at {fire_rc} "
                  f"(burn_pen>0) – truncating path at idx {safe_upto}.")
            self.aborted_fire_intercept = True
        else:
            trimmed = result
            self.aborted_fire_intercept = False

        # --- console print (trimmed only) -----------------------------
        # pretty_print_fireline_result(trimmed, precision=2, show_header=True)


        # --- unpack into path/time ------------------------------------
        self.path, self.path_times = [], []
        for cell_rc, metrics in trimmed:
            self.path.append(tuple(map(int, cell_rc)))
            self.path_times.append(float(metrics[3]))  # cumulative minutes

        # # drop start cell (instant arrival)
        # if self.path and self.path[0] == self.start_rc:
        #     self.path.pop(0)
        #     self.path_times.pop(0)


        self.last_cell_idx = -1
        self._eta_remaining = self.path_times[-1] if self.path_times else 0.0

        print(f"[GC#{self.unique_id:02d}] fire-line length {len(self.path)} cells "
              f"({self._eta_remaining:.1f} min; "
              f"{'ABORTED EARLY' if self.aborted_fire_intercept else 'full plan'})")
