# fireline_abm.py
import numpy as np
import matplotlib.pyplot as plt
import firelinepath
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])

# def compute_fireline_for_sector(
#     mtt: np.ndarray,
#     fuel: np.ndarray,
#     feasibility: np.ndarray,
#     sector_center: Point,
#     sector_boundary: Point,
#     sector_angle: tuple[float,float],
#     transform: tuple[float,float],
#     bounds: tuple[float,float],
#     buffer_dist: float,
#     clear_burning_penalty: float = 100.0,
#     distance_penalty: float = 0.001,
#     finish: np.ndarray | None = None,
# ):
def compute_fireline_for_sector(
    mtt: np.ndarray,
    fuel: np.ndarray,
    feasibility: np.ndarray,
    sector_center: Point,
    sector_boundaries: tuple[Point,Point],
    sector_angle: tuple[float,float],
    transform: tuple[float,float],
    bounds: tuple[float,float],
    buffer_dist: float,
    clear_burning_penalty: float = 100.0,
    distance_penalty: float = 0.001,
    finish: np.ndarray | None = None,
):
    # Unpack some locals for both plotting and filtering
    dx_cell, dy_cell = transform
    x0, y0 = bounds
    θ_min, θ_max = sector_angle
    b1, b2 = sector_boundaries

    """
    Run firelinePath for exactly one sector.

    Args:
      mtt               : arrival‑time grid
      fuel              : fuel model (ints)
      feasibility       : boolean mask
      sector_center     : (x,y) world coords of fire centroid
      sector_boundary   : (x,y) world coords of sector boundary
      sector_angle      : (θ_min, θ_max) in degrees
      transform         : (dx_cell, dy_cell) from model.fire.transform
      bounds            : (x_left, y_top) of grid origin
      buffer_dist       : how far back along the ray to start (world units)
      clear_burning_penalty, distance_penalty, finish : passed to firelinepath

    Returns:
      filtered_result, filtered_barrier  (only points in that sector)
    """

    # dx_cell, dy_cell = transform
    # x0, y0 = bounds
    # θ_min, θ_max = sector_angle
    #
    # # 1) Compute buffered start in world‐coords
    # dx = sector_boundary.x - sector_center.x
    # dy = sector_boundary.y - sector_center.y
    # dist = np.hypot(dx, dy)
    # ux, uy = dx/dist, dy/dist
    # bx = sector_boundary.x - ux * buffer_dist
    # by = sector_boundary.y - uy * buffer_dist
    #
    # # 2) Map to grid‐indices (row, col)
    # col = int((bx - x0)/dx_cell)
    # row = int((y0 - by)/dy_cell)
    # start = np.array([col, row])
    # print('start', start)
    # Unpack the two boundary points
    b1, b2 = sector_boundaries

    # Compute the angle of a point around the center
    def angle_of(pt: Point):
        dx = pt.x - sector_center.x
        dy = pt.y - sector_center.y
        return (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    θ1 = angle_of(b1)
    θ2 = angle_of(b2)

    # Compute CCW arcs between them
    arc1 = (θ2 - θ1) % 360  # CCW from b1 → b2
    arc2 = (θ1 - θ2) % 360  # CCW from b2 → b1

    # Pick the boundary with the smaller CCW arc as the start
    if arc1 < arc2:
        start_boundary = b1
        start_angle = θ1
    else:
        start_boundary = b2
        start_angle = θ2

    print(f"🔍 start boundary chosen at angle {start_angle:.1f}° → {start_boundary}")

    # Buffer back along that boundary in world‐coords
    dx = start_boundary.x - sector_center.x
    dy = start_boundary.y - sector_center.y
    dist = np.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    bx = start_boundary.x + ux * buffer_dist
    by = start_boundary.y + uy * buffer_dist

    # Convert to grid indices
    dx_cell, dy_cell = transform
    x0, y0 = bounds
    col = int((bx - x0) / dx_cell)
    row = int((y0 - by) / dy_cell)
    start = np.array([row, col], dtype=int)
    print("🔍 grid start index =", start, "feasible?", feasibility[row, col])
    value_map = {
        1: 0.019,
        2: 0.025,
        3: 0.107,
        4: 0.186,
        5: 0.107,
        6: 0.107,
        7: 0.107,
        8: 0.027,
        9: 0.009,
        10: 0.075,
        11: 0.075,
        12: 0.075,
        13: 0.186,
        91: 0.0,  # Please add these; otherwise you'll get errors
        92: 0.25,  # Please add these; otherwise you'll get errors
        98: 0.0,  # Please add these; otherwise you'll get errors
        99: 0.0,  # Please add these; otherwise you'll get errors
        101: 0.6 / 20,
        102: 0.6 / 20,
        103: 0.6 / 20,
        104: 0.7 / 20,
        105: 0.5 / 20,
        106: 0.6 / 20,
        107: 2.5 / 20,
        108: 2.5 / 20,
        109: 2.5 / 20,
        121: 2.3 / 20,
        122: 2.4 / 20,
        123: 2.3 / 20,
        124: 2.4 / 20,
        141: 2.3 / 20,
        142: 4.0 / 20,
        143: 3.9 / 20,
        144: 2.2 / 20,
        145: 4.1 / 20,
        146: 2.2 / 20,
        147: 4.2 / 20,
        148: 2.4 / 20,
        149: 4.1 / 20,
        161: 1.7 / 20,
        162: 1.7 / 20,
        163: 1.7 / 20,
        164: 2.3 / 20,
        165: 2.0 / 20,
        181: 1.4 / 20,
        182: 0.2 / 20,
        183: 0.8 / 20,
        184: 1.5 / 20,
        185: 0.9 / 20,
        186: 0.3 / 20,
        187: 3.7 / 20,
        188: 0.7 / 20,
        189: 0.5 / 20,
        201: 1.5 / 20,
        202: 1.5 / 20,
        203: 3.7 / 20,
        204: 3.9 / 20,
    }

    # print("start (row,col) =", row, col,
    #       "feasible?", feasibility[row, col])
    # print("🔍 Casting mtt to float64 for consistency…")
    # mtt = mtt.astype(np.float64)
    #
    # print("🔍 DEBUG create_fireline_path inputs:")
    # print(f"  start:           {start!r}")
    # print(f"    • type:        {type(start)}")
    # print(f"    • dtype:       {getattr(start, 'dtype', None)}")
    # print(f"    • shape:       {getattr(start, 'shape', None)}")
    # print(f"  mtt:             type={type(mtt)}, dtype={mtt.dtype}, shape={mtt.shape}")
    # print(f"  fuel:            type={type(fuel)}, dtype={fuel.dtype}, shape={fuel.shape}")
    # print(f"  feasibility:     type={type(feasibility)}, dtype={feasibility.dtype}, shape={feasibility.shape}")
    # print(f"  clear_penalty:   {clear_burning_penalty!r} ({type(clear_burning_penalty)})")
    # print(f"  dist_penalty:    {distance_penalty!r} ({type(distance_penalty)})")
    # print("  fuel_clear_cost mapping types:")
    # for k, v in value_map.items():
    #     print(f"    key={k!r} → value type={type(v)}")
    # print("––– calling create_fireline_path() –––\n")

    # # 3) Call firelinepath
    # result, heuristic_barrier = firelinepath.create_fireline_path(
    #     start=start,
    #     mtt=mtt,
    #     fuel=fuel,
    #     fuel_clear_cost=value_map,            # if you have a cost map you can pass it here
    #     feasibility=feasibility,
    #     finish=None,
    #     clear_burning_penalty=clear_burning_penalty,
    #     distance_penalty=distance_penalty,
    # )
    #
    # # 4) Filter to only that sector by angle
    # def in_sector(pt):
    #     r,c = pt
    #     x = x0 + (c+0.5)*dx_cell
    #     y = y0 - (r+0.5)*dy_cell
    #     θ = (np.degrees(np.arctan2(y-sector_center.y, x-sector_center.x)) + 360) % 360
    #     if θ_min < θ_max:
    #         return θ_min <= θ < θ_max
    #     else:
    #         return θ >= θ_min or θ < θ_max
    #
    # filtered_result  = [(pt,st) for (pt,st) in result  if in_sector(pt)]
    # filtered_barrier = [pt       for pt      in heuristic_barrier if in_sector(pt)]
    #
    # # 5) Quick plot
    # fig, ax = plt.subplots()
    # ax.imshow(mtt, cmap="Greens", origin="upper")
    # # barrier in red
    # B = np.zeros_like(mtt, float)
    # for r,c in filtered_barrier: B[r,c]=1
    # ax.imshow(B, cmap="Reds", alpha=0.5, origin="upper")
    # # line in blue
    # L = np.zeros_like(mtt, float)
    # for (r,c),_ in filtered_result: L[r,c]=1
    # ax.imshow(L, cmap="Blues", alpha=0.5, origin="upper")
    # ax.set_title("Fireline – sector")
    # plt.tight_layout()
    #
    # return filtered_result, filtered_barrier
    adj_value_map = {k: v / 2 for k, v in value_map.items()}

    # … your existing code up through the call to create_fireline_path() …
    result = firelinepath.fireline_between_two_points(
        start=start,
        mtt=mtt,
        fuel=fuel,
        fuel_clear_cost=adj_value_map,
        feasibility=None,
        finish=np.array([550, 550]),
        # finish=None,
        clear_burning_penalty=clear_burning_penalty,
        distance_penalty=distance_penalty,
    )

    # ——— build the same overlay arrays as in fireline.py ———
    path_with_time = np.zeros_like(mtt, dtype=float)
    path_with_opt_cost = np.zeros_like(mtt, dtype=float)
    path_with_clears_through_fire = np.zeros_like(mtt, dtype=float)
    # heuristic_barrier_path = np.zeros_like(mtt, dtype=float)
    #
    # for r, c in heuristic_barrier:
    #     heuristic_barrier_path[r, c] = 1

    for (r, c), stats in result:
        _, burning_penalty, _, time_build_up, cost_build_up = stats
        path_with_clears_through_fire[r, c] = 1.0 if burning_penalty > 0 else 0.0
        path_with_time[r, c] = time_build_up
        path_with_opt_cost[r, c] = cost_build_up

    final_time = result[-1][1][3]
    final_cost = result[-1][1][4]
    path_time = path_with_time / final_time
    # (optional) path_cost = path_with_opt_cost / final_cost

    # ——— now replicate your fireline.py plotting ———
    fig, ax = plt.subplots()
    # ax.imshow(mtt, cmap="Greens", origin="upper", vmin=0)
    ## new – draw the mtt as before…
    ax.imshow(mtt, cmap="Greens", origin="upper", vmin=0)

    # …then highlight zeros in pure red
    zero_mask = (mtt == 0)
    ax.imshow(
        zero_mask,
        cmap=plt.cm.Reds,      # a red colormap
        origin="upper",
        alpha=0.8,             # adjust opacity (0.0–1.0) as you like
        vmin=0, vmax=1
    )
    # feasibility overlay (blue mask)
    ax.imshow(
        np.dstack([
            feasibility.astype(float),
            feasibility.astype(float),
            feasibility.astype(float),
            1.0 - feasibility.astype(float),
        ]),
        cmap="Blues",
        origin="upper",
    )

    # barrier overlay (rainbow)
    # ax.imshow(
    #     np.dstack([
    #         heuristic_barrier_path,
    #         heuristic_barrier_path,
    #         np.zeros_like(mtt),
    #         heuristic_barrier_path,
    #     ]),
    #     cmap="gist_rainbow",
    #     origin="upper",
    # )

    # path overlay (red→blue gradient)
    ax.imshow(
        np.dstack([
            path_time * 0.75,
            np.zeros_like(mtt),
            0.75 * (1 - path_time),
            np.ceil(path_time),
        ]),
        cmap="Reds",
        origin="upper",
    )

    # clears‐through‐fire overlay (red dots)
    ax.imshow(
        np.dstack([
            path_with_clears_through_fire,
            np.zeros_like(mtt),
            np.zeros_like(mtt),
            path_with_clears_through_fire,
        ]),
        cmap="gist_rainbow",
        origin="upper",
    )

    # mark the start
    # ax.plot(start[1], start[0], 'ro', markersize=6, label='Start')

    col1 = int((b1.x - x0) / dx_cell)
    row1 = int((y0 - b1.y) / dy_cell)
    col2 = int((b2.x - x0) / dx_cell)
    row2 = int((y0 - b2.y) / dy_cell)

    # plot start (in red), boundary1 (green), boundary2 (magenta)
    ax.plot(start[1], start[0], 'ro', markersize=6, label='Start')
    # ax.plot(col1, row1, 'g^', markersize=8, label='Boundary A')
    # ax.plot(col2, row2, 'm^', markersize=8, label='Boundary B')

    # draw fireline path as a thick connected line
    line_pts = np.array([(c, r) for (r, c), _ in result])
    if len(line_pts) > 1:
        ax.plot(line_pts[:, 0], line_pts[:, 1], color='blue', linewidth=2, label="Fireline Path")
    ax.legend(loc='upper right')

    ax.set_title("Fireline – full grid")
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig( "fireline.png", dpi=300, bbox_inches='tight')

    # plt.show()

    # ——— FILTER to only pts inside the sector ———
    def in_sector(pt):
        r, c = pt
        x = x0 + (c + 0.5) * dx_cell
        y = y0 - (r + 0.5) * dy_cell
        θ = (np.degrees(np.arctan2(y - sector_center.y, x - sector_center.x)) + 360) % 360
        return (θ_min <= θ < θ_max) if θ_min < θ_max else (θ >= θ_min or θ < θ_max)

    filtered_result = [(pt, st) for pt, st in result if in_sector(pt)]
    # filtered_barrier = [pt for pt in heuristic_barrier if in_sector(pt)]

    # build sector overlays
    barrier_sec = np.zeros_like(mtt, float)
    line_sec = np.zeros_like(mtt, float)
    # for r, c in filtered_barrier: barrier_sec[r, c] = 1
    for (r, c), _ in filtered_result: line_sec[r, c] = 1

    # ——— PLOT 2: sector‐limited ———
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.imshow(mtt, cmap="Greens", origin="upper", vmin=0)
    ax2.imshow(mtt, cmap="Greens", origin="upper", vmin=0)

    # …then highlight zeros in pure red
    zero_mask = (mtt == 0)
    ax2.imshow(
        zero_mask,
        cmap=plt.cm.Reds,  # a red colormap
        origin="upper",
        alpha=0.8,  # adjust opacity (0.0–1.0) as you like
        vmin=0, vmax=1
    )
    # just draw barrier+line in this sector
    ax2.imshow(barrier_sec, cmap="Reds", alpha=0.6, origin="upper")
    ax2.imshow(line_sec, cmap="Blues", alpha=0.6, origin="upper")
    ax2.plot(start[1], start[0], 'ro', markersize=6, label='Start')
    ax2.set_title(f"Fireline – sector [{θ_min:.0f}°, {θ_max:.0f}°]")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    return result
