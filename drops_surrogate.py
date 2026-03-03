#!/usr/bin/env python3
"""
drop_geometry.py

Adds an optional debug toggle to visualize:
  • the fire contour
  • the convex hull (and buffered hull)
  • the selected ray (your chosen angle)
  • 5° increment rays and where they intersect the (buffered) hull boundary

Usage (example):
  pt, drop_angle = get_point_angle(
      fire, angle=42, time=300, buffer_dist=0, origin=None,
      debug_plot=True, debug_step_deg=5, debug_label_every=15,
      debug_save_path="hull_debug.png",
  )
"""

import logging
import numpy as np
import shapely
from scipy.spatial import ConvexHull
from skimage.draw import polygon  # kept because you had it imported


from shapely.errors import EmptyPartError

log = logging.getLogger(__name__)


# ============================================================
# DEBUG PLOTTING HELPERS
# ============================================================
def _ray_linestring(origin_xy: np.ndarray, angle_deg: float, length: float):
    """Create a shapely LineString ray from origin at angle_deg out to `length`."""
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    ux, uy = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    end = (ox + length * ux, oy + length * uy)
    return shapely.geometry.LineString([(ox, oy), end])


def _closest_point_on_geom_to_origin(geom, origin_xy: np.ndarray):
    """
    From an intersection geometry (Point/MultiPoint/LineString/etc),
    return the point (x,y) closest to origin, or None if empty.
    """
    if geom is None or geom.is_empty:
        return None

    ox, oy = float(origin_xy[0]), float(origin_xy[1])

    def dist2_xy(x, y):
        return (x - ox) ** 2 + (y - oy) ** 2

    # Point
    if isinstance(geom, shapely.geometry.Point):
        return np.array([geom.x, geom.y], dtype=float)

    # MultiPoint
    if isinstance(geom, shapely.geometry.MultiPoint):
        pts = [np.array([p.x, p.y], dtype=float) for p in geom.geoms]
        if not pts:
            return None
        d2 = [dist2_xy(p[0], p[1]) for p in pts]
        return pts[int(np.argmin(d2))]

    # Use nearest_points if possible
    try:
        from shapely.ops import nearest_points
        o = shapely.geometry.Point(ox, oy)
        p_near = nearest_points(o, geom)[1]
        if isinstance(p_near, shapely.geometry.Point):
            return np.array([p_near.x, p_near.y], dtype=float)
    except Exception:
        pass

    # Fallback: try endpoints if it has coords
    try:
        coords = []
        if hasattr(geom, "geoms"):  # multi / collection
            for g in geom.geoms:
                if hasattr(g, "coords"):
                    coords.extend(list(g.coords))
        elif hasattr(geom, "coords"):
            coords = list(geom.coords)

        if not coords:
            return None
        pts = [np.array([c[0], c[1]], dtype=float) for c in coords]
        d2 = [dist2_xy(p[0], p[1]) for p in pts]
        return pts[int(np.argmin(d2))]
    except Exception:
        return None


def plot_hull_debug(
    contour,
    hull_polygon,
    buff_polygon,
    origin_xy: np.ndarray,
    selected_angle_deg: float,
    selected_intersection_xy: np.ndarray | None,
    step_deg: int = 5,
    label_every: int = 15,
    save_path: str | None = None,
    show: bool = True,
    # --- new knobs ---
    plot_contour: bool = False,      # <-- turn OFF squiggles by default
    label_offset: float = 250.0,     # <-- push angle labels outward (meters)
    text_scale: float = 1.6,         # <-- global-ish font bump
):
    """
    Debug plot: convex hull (+buffer) + origin + selected ray/intersection,
    plus step_deg rays and their hull boundary intersections.
    Angle labels are placed OUTSIDE the boundary along the ray direction.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))

    # ---------------------------
    # Font sizes (simple + robust)
    # ---------------------------
    base_fs = 10 * text_scale
    title_fs = 12 * text_scale
    legend_fs = 9 * text_scale
    angle_fs = 10.5 * text_scale

    # ---- Plot contour geometry (these are the “squiggly lines”) ----
    if plot_contour:
        try:
            if hasattr(contour, "geoms"):
                for g in contour.geoms:
                    if hasattr(g, "xy"):
                        ax.plot(*g.xy, linewidth=1.0, alpha=0.6)
                    elif hasattr(g, "coords"):
                        xy = np.asarray(g.coords)
                        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.0, alpha=0.6)
            elif hasattr(contour, "xy"):
                ax.plot(*contour.xy, linewidth=1.0, alpha=0.6)
            elif hasattr(contour, "coords"):
                xy = np.asarray(contour.coords)
                ax.plot(xy[:, 0], xy[:, 1], linewidth=1.0, alpha=0.6)
        except Exception:
            pass

    # ---- Plot hull polygon outline ----
    hx, hy = hull_polygon.exterior.xy
    ax.plot(hx, hy, linewidth=2.5, label="convex hull")

    # ---- Plot buffered hull outline ----
    boundary_poly = hull_polygon
    if buff_polygon is not None:
        bx, by = buff_polygon.exterior.xy
        ax.plot(bx, by, linewidth=2.5, linestyle="--", label="buffered hull")
        boundary_poly = buff_polygon

    boundary = boundary_poly.boundary

    # ---- Origin ----
    ax.scatter([origin_xy[0]], [origin_xy[1]], s=90, marker="x", label="origin")

    # ---- Determine ray length based on hull bounds ----
    minx, miny, maxx, maxy = hull_polygon.bounds
    diag = float(np.hypot(maxx - minx, maxy - miny))
    ray_len = 2.0 * diag + 1.0

    # ---- Plot step_deg rays + intersections ----
    angles = np.arange(0, 360, step_deg, dtype=float)
    ox, oy = float(origin_xy[0]), float(origin_xy[1])

    for a in angles:
        ray = _ray_linestring(origin_xy, a, ray_len)
        inter = ray.intersection(boundary)
        hit = _closest_point_on_geom_to_origin(inter, origin_xy)
        if hit is None:
            continue

        # faint ray + hit dot
        xs, ys = ray.xy
        ax.plot(xs, ys, linewidth=0.5, alpha=0.30)
        ax.scatter([hit[0]], [hit[1]], s=14)

        # label every N degrees (placed OUTSIDE the boundary)
        if label_every is not None and label_every > 0 and (int(a) % int(label_every) == 0):
            v = np.array([hit[0] - ox, hit[1] - oy], dtype=float)
            n = float(np.hypot(v[0], v[1]))
            if n > 1e-9:
                u = v / n
                label_xy = hit + label_offset * u  # <-- push outward

                ax.text(
                    label_xy[0],
                    label_xy[1],
                    f"{int(a)}°",
                    fontsize=angle_fs,
                    ha="center",
                    va="center",
                    clip_on=False,  # allow labels near/over edge
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                )

    # ---- Highlight selected angle ray + intersection ----
    sel_ray = _ray_linestring(origin_xy, selected_angle_deg, ray_len)
    xs, ys = sel_ray.xy
    ax.plot(xs, ys, linewidth=3.0, label=f"selected ray ({selected_angle_deg:.1f}°)")

    if selected_intersection_xy is not None:
        ax.scatter([selected_intersection_xy[0]], [selected_intersection_xy[1]],
                   s=140, label="selected hit", zorder=5)

    # ============================================================
    # Hard-limit axes to fixed window around origin (your request)
    # ============================================================
    ax.set_xlim(ox - 3000, ox + 3000)
    ax.set_ylim(oy - 3000, oy + 3000)

    ax.set_aspect("equal", "box")
    ax.set_title("Convex hull debug: 5° intersections", fontsize=title_fs)
    ax.legend(loc="best", fontsize=legend_fs)

    # Make tick labels larger too
    ax.tick_params(labelsize=base_fs)

    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

# ============================================================
# SAFE WRAPPER (WITH OPTIONAL DEBUG TOGGLE)
# ============================================================
def safe_get_point(
    fire,
    angle,
    time,
    buffer_dist,
    origin,
    debug_plot: bool = False,
    debug_step_deg: int = 5,
    debug_label_every: int = 15,
    debug_save_path: str | None = None,
):
    """
    Wraps get_point() so that *any* error in the geometric construction
    is turned into a graceful ‘None’ return instead of crashing the sim.

    debug_plot=True enables plotting of hull + 5° increments.
    """
    try:
        return get_point(
            fire,
            angle,
            time,
            buffer_dist,
            origin,
            debug_plot=debug_plot,
            debug_step_deg=debug_step_deg,
            debug_label_every=debug_label_every,
            debug_save_path=debug_save_path,
        )

    except AttributeError as e:  # fire has no get_contour()
        log.warning("⚠️  drop skipped – %s", e)
    except EmptyPartError as e:  # Shapely failed on empty lines
        log.warning("⚠️  drop skipped – %s", e)
    except ValueError as e:  # sometimes Shapely raises this
        log.warning("⚠️  drop skipped – %s", e)
    except Exception:
        log.exception("⚠️  unexpected error while picking drop point – skipping this drop")

    return None, None, None


# ============================================================
# MAIN DROP LOGIC
# ============================================================
class Drop:
    def __init__(self, fire, drop_mid: np.ndarray, angle: float, length: int = 500, width: int = 150):
        """
        Create a drop object.

        Parameters:
          fire: Surrogate fire model instance.
          drop_mid: (x,y) coordinate (in meters) for the midpoint of the drop.
          angle: Orientation (in degrees) for the drop.
          length, width: Dimensions of the drop in meters.
        """
        self.fire = fire
        self.drop_mid = drop_mid
        self.length = length
        self.width = width
        self.angle = angle
        self.rect = get_rect(self.fire, self.drop_mid[0], self.drop_mid[1], self.length, self.width, self.angle)
        self.drop_line = self.get_line()

    def run_drop(self, new_id=98):
        """
        Apply a drop (retardant) by updating the fuel model over the drop rectangle.
        Returns the new arrival time map (or similar path object) from the surrogate.
        """
        fuel_model, self.drop_array = self.fire.fuel_editor_numpy(self.rect, self.fire.fuel_model, new_id)
        path = self.fire.re_run(fuel_model)
        return path

    def run_one_drop(self, drop_num=0, new_id=98):
        """
        Resets the fuel model (if supported) and then performs a drop.
        """
        if hasattr(self.fire, "reset"):
            self.fire.reset(drop_num)
        return self.run_drop(new_id=new_id)

    def get_line(self):
        """
        Returns a shapely LineString representing the drop line.
        """
        point_mid = np.array(self.drop_mid, dtype=float)
        unit_vector = np.array([np.cos(np.deg2rad(self.angle)), np.sin(np.deg2rad(self.angle))], dtype=float)
        point_1 = point_mid + (self.length * unit_vector / 2)
        point_2 = point_mid - (self.length * unit_vector / 2)
        p1 = shapely.geometry.Point(point_1)
        p2 = shapely.geometry.Point(point_2)
        return shapely.geometry.LineString([p1, p2])


def get_rect(fire, x: float, y: float, length: int, width: int, angle: float) -> np.ndarray:
    """
    Compute the rectangle (as an array of corner points) for a drop.
    Uses the surrogate's bounds and transform to convert from meters to grid indices.
    """
    cell_size = fire.transform[0]
    x_arr = np.arange(fire.bounds.left, fire.bounds.right, cell_size)

    y_cell = abs(fire.transform[4])
    y_arr = np.arange(fire.bounds.top, fire.bounds.bottom, -y_cell)

    if x < x_arr[0] or x > x_arr[-1]:
        raise ValueError(f"x must be in the range {x_arr[0]}-{x_arr[-1]}")
    if y > y_arr[0] or y < y_arr[-1]:
        raise ValueError(f"y must be in the range {y_arr[-1]}-{y_arr[0]}")

    theta = np.deg2rad(angle)

    length_cells = int(length / cell_size)
    width_cells = int(width / cell_size)

    x_idx = int((x - fire.bounds.left) / cell_size)
    y_idx = int((fire.bounds.top - y) / y_cell)

    # Increase length to ensure proper overlap.
    length_cells = length_cells + width_cells

    rect = np.array(
        [
            (-length_cells / 2, -width_cells / 2),
            (length_cells / 2, -width_cells / 2),
            (length_cells / 2, width_cells / 2),
            (-length_cells / 2, width_cells / 2),
            (-length_cells / 2, -width_cells / 2),
        ],
        dtype=float,
    )

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]], dtype=float)

    offset = np.array([x_idx, y_idx], dtype=float)
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect


def set_drop(
    fire,
    angle: float,
    time: int,
    length=500,
    width=150,
    buffer_dist: int = 0,
    origin: np.ndarray = None,
    debug_plot: bool = False,
    debug_step_deg: int = 5,
    debug_label_every: int = 15,
    debug_save_path: str | None = None,
) -> Drop:
    """
    Create and return a Drop object.

    The drop point and drop angle are computed using get_point_angle.
    """
    drop_mid, drop_angle = get_point_angle(
        fire,
        angle,
        time,
        buffer_dist,
        origin,
        debug_plot=debug_plot,
        debug_step_deg=debug_step_deg,
        debug_label_every=debug_label_every,
        debug_save_path=debug_save_path,
    )
    return Drop(fire, drop_mid, drop_angle, length, width)


def get_point_angle(
    fire,
    angle,
    time,
    buffer_dist: int = 0,
    origin=None,
    debug_plot: bool = False,
    debug_step_deg: int = 5,
    debug_label_every: int = 15,
    debug_save_path: str | None = None,
):
    """
    Compute the drop point and drop angle.

    Returns (point, drop_angle) where drop_angle is perpendicular to the facet normal.

    debug_plot=True enables plotting of hull + 5° increments.
    """
    pt, angle_par, hull = safe_get_point(
        fire,
        angle,
        time,
        buffer_dist,
        origin,
        debug_plot=debug_plot,
        debug_step_deg=debug_step_deg,
        debug_label_every=debug_label_every,
        debug_save_path=debug_save_path,
    )
    if pt is None:
        return None, None
    drop_angle = np.mod(np.mod(angle_par, 360) + 90, 360)
    return pt, drop_angle


def get_point(
    fire,
    angle,
    time,
    buffer_dist: int = 0,
    origin=None,
    debug_plot: bool = False,
    debug_step_deg: int = 5,
    debug_label_every: int = 15,
    debug_save_path: str | None = None,
):
    """
    Compute the drop point as the intersection of a ray from the fire ignition point (or origin)
    with the convex hull of the fire contour at the given time (plus optional buffer).

    Returns (point, angle_par, hull) where:
      • point is the closest intersection point along the ray
      • angle_par is the orientation derived from the hull facet normal
      • hull is the scipy ConvexHull built from buffered hull boundary coords

    debug_plot=True enables plotting of hull + 5° increments.
    """
    # Attempt to use the surrogate's get_contour if available; otherwise, use a fallback.
    try:
        contour = fire.get_contour(time)
    except AttributeError:
        import matplotlib.pyplot as plt
        import shapely.geometry

        fig, ax = plt.subplots()
        cell_size = fire.transform[0]
        x_arr = np.arange(fire.bounds.left, fire.bounds.right, cell_size)
        y_arr = np.arange(fire.bounds.top, fire.bounds.bottom, fire.transform[4])

        contour_obj = ax.contour(x_arr, y_arr, fire.current_fire(time), levels=[time])

        line_list = []
        for level_paths in contour_obj.allsegs:
            for path in level_paths:
                line_list.append(shapely.geometry.LineString(path))

        contour = shapely.geometry.MultiLineString(line_list)
        plt.close(fig)

    hull_polygon = contour.convex_hull
    buff_polygon = hull_polygon.buffer(buffer_dist)

    # Extract coordinates from the buffered hull boundary.
    points = np.array(buff_polygon.exterior.coords)
    hull = ConvexHull(points)

    # Determine origin point.
    if origin is None:
        if hasattr(fire, "ignition_pt"):
            point = np.array(fire.ignition_pt, dtype=float)
        else:
            point = np.array(
                [(fire.bounds.left + fire.bounds.right) / 2,
                 (fire.bounds.top + fire.bounds.bottom) / 2],
                dtype=float,
            )
    else:
        point = np.array(origin, dtype=float)

    unit_ray = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))], dtype=float)

    closest_intersection_point = None
    closest_distance = np.inf
    angle_norm = None

    for plane in hull.equations:
        normal = plane[:-1]
        distance = -plane[-1]

        dot_product = np.dot(normal, unit_ray)
        if np.abs(dot_product) <= 1e-8:
            continue

        t = (distance - np.dot(normal, point)) / dot_product
        if t >= 0:
            intersection_point = point + t * unit_ray

            # Keep your original “inside-ish” test.
            if np.all(np.dot(normal, intersection_point) <= distance + 10):
                d = float(np.linalg.norm(intersection_point - point))
                if d < closest_distance:
                    closest_distance = d
                    closest_intersection_point = intersection_point
                    angle_norm = np.rad2deg(np.arctan2(normal[1], normal[0]))

    # Optional debug plot
    if debug_plot:
        try:
            plot_hull_debug(
                contour=contour,
                hull_polygon=hull_polygon,
                buff_polygon=buff_polygon,
                origin_xy=point,
                selected_angle_deg=float(angle),
                selected_intersection_xy=closest_intersection_point,
                step_deg=int(debug_step_deg),
                label_every=int(debug_label_every),
                save_path=debug_save_path,
                show=True,
            )
        except Exception:
            log.exception("⚠️  debug plotting failed (continuing without plot)")

    if closest_intersection_point is None:
        return None, None, None

    return closest_intersection_point, angle_norm, hull











# import numpy as np
# import shapely
# from scipy.spatial import ConvexHull
# from skimage.draw import polygon
#
# from shapely.errors import EmptyPartError
# import logging
#
# # ← new
# log = logging.getLogger(__name__)                  # use whatever logger you like
#
#
# # def safe_get_point(fire, angle, time, buffer_dist, origin):
# #     try:
# #         pt, angle_par, hull = get_point(fire, angle, time, buffer_dist, origin)
# #
# #         # 1) reject degenerate hulls (point or line)
# #         if len(hull.vertices) < 3 or hull.volume < 1.0:
# #             return None, None, None
# #
# #         # 2) ensure the cell will still be un-burned *at drop time*
# #         arrivals = fire.arrival_time_grid
# #         dx, dy  = fire.transform[0], -fire.transform[4]
# #         col = int((pt[0] - fire.bounds.left) / dx)
# #         row = int((fire.bounds.top - pt[1])  / dy)
# #
# #         burned_by_drop_time = (
# #             0 <= row < arrivals.shape[0] and
# #             0 <= col < arrivals.shape[1] and
# #             np.isfinite(arrivals[row, col]) and
# #             arrivals[row, col] <= time      # ← use the passed-in time
# #         )
# #         if burned_by_drop_time:
# #             return None, None, None
# #
# #         return pt, angle_par, hull
# #
# #     except (AttributeError, EmptyPartError, ValueError) as e:
# #         log.warning("⚠️  drop skipped – %s", e)
# #     except Exception:
# #         log.exception("⚠️  unexpected error while picking drop point – skipping this drop")
# #
# #     return None, None, None
#
#
# def safe_get_point(fire, angle, time, buffer_dist, origin):
#     """
#     Wraps the old get_point() so that *any* error in the geometric
#     construction is turned into a graceful ‘None’ return instead of
#     crashing the whole sim.
#     """
#     try:
#         # your original heavy-duty geometry
#         return get_point(fire, angle, time, buffer_dist, origin)
#
#     # ── handle the cases we have seen in real runs ──────────────────────
#     except AttributeError as e:            # fire has no get_contour()
#         log.warning("⚠️  drop skipped – %s", e)
#     except EmptyPartError as e:            # Shapely failed on empty lines
#         log.warning("⚠️  drop skipped – %s", e)
#     except ValueError as e:                # sometimes Shapely raises this
#         log.warning("⚠️  drop skipped – %s", e)
#
#     # *anything* else: keep the sim alive, report briefly
#     except Exception as e:
#         log.exception("⚠️  unexpected error while picking drop point – "
#                       "skipping this drop")
#     # -------------------------------------------------------------------
#     return None, None, None        # signal “no point available”
# class Drop:
#     def __init__(self, fire, drop_mid: np.ndarray, angle: float, length: int = 500, width: int = 150):
#         """
#         Create a drop object.
#
#         Parameters:
#           fire: Surrogate fire model instance.
#           drop_mid: (x,y) coordinate (in meters) for the midpoint of the drop.
#           angle: Orientation (in degrees) for the drop.
#           length, width: Dimensions of the drop in meters.
#         """
#         self.fire = fire
#         self.drop_mid = drop_mid
#         self.length = length
#         self.width = width
#         self.angle = angle
#         self.rect = get_rect(self.fire, self.drop_mid[0], self.drop_mid[1], self.length, self.width, self.angle)
#         self.drop_line = self.get_line()
#
#     def run_drop(self, new_id=98):
#         """
#         Apply a drop (retardant) by updating the fuel model over the drop rectangle.
#         Returns the new arrival time map (or similar path object) from the surrogate.
#         """
#         # Use the surrogate's fuel_editor_numpy with current fuel_model.
#         fuel_model, self.drop_array = self.fire.fuel_editor_numpy(self.rect, self.fire.fuel_model, new_id)
#         path = self.fire.re_run(fuel_model)
#         return path
#
#     def run_one_drop(self, drop_num=0, new_id=98):
#         """
#         Resets the fuel model (if supported) and then performs a drop.
#         """
#         if hasattr(self.fire, "reset"):
#             self.fire.reset(drop_num)
#         return self.run_drop(new_id=new_id)
#
#     def get_line(self):
#         """
#         Returns a shapely LineString representing the drop line.
#         """
#         point_mid = np.array(self.drop_mid)
#         unit_vector = np.array([np.cos(np.deg2rad(self.angle)), np.sin(np.deg2rad(self.angle))])
#         point_1 = point_mid + (self.length * unit_vector / 2)
#         point_2 = point_mid - (self.length * unit_vector / 2)
#         p1 = shapely.geometry.Point(point_1)
#         p2 = shapely.geometry.Point(point_2)
#         return shapely.geometry.LineString([p1, p2])
#
#
# def get_rect(fire, x: float, y: float, length: int, width: int, angle: float) -> np.ndarray:
#     """
#     Compute the rectangle (as an array of corner points) for a drop.
#     Uses the surrogate's bounds and transform to convert from meters to grid indices.
#     """
#     # Assume cell size is given by fire.transform[0] (x direction)
#     cell_size = fire.transform[0]
#     # Create x and y arrays from the surrogate's bounds.
#     x_arr = np.arange(fire.bounds.left, fire.bounds.right, cell_size)
#     # For y, we assume transform[4] is negative (top to bottom).
#     y_cell = abs(fire.transform[4])
#     y_arr = np.arange(fire.bounds.top, fire.bounds.bottom, -y_cell)
#
#     if x < x_arr[0] or x > x_arr[-1]:
#         raise ValueError(f"x must be in the range {x_arr[0]}-{x_arr[-1]}")
#     if y > y_arr[0] or y < y_arr[-1]:
#         raise ValueError(f"y must be in the range {y_arr[-1]}-{y_arr[0]}")
#
#     theta = np.deg2rad(angle)
#     # Convert dimensions from meters to grid cells.
#     length_cells = int(length / cell_size)
#     width_cells = int(width / cell_size)
#     # Convert drop location (in meters) to grid index.
#     x_idx = int((x - fire.bounds.left) / cell_size)
#     # Since y decreases downward, compute index as:
#     y_idx = int((fire.bounds.top - y) / y_cell)
#     # Increase length to ensure proper overlap.
#     length_cells = length_cells + width_cells
#     rect = np.array([
#         (-length_cells / 2, -width_cells / 2),
#         (length_cells / 2, -width_cells / 2),
#         (length_cells / 2, width_cells / 2),
#         (-length_cells / 2, width_cells / 2),
#         (-length_cells / 2, -width_cells / 2)
#     ])
#     # Rotation matrix.
#     R = np.array([[np.cos(theta), -np.sin(theta)],
#                   [np.sin(theta), np.cos(theta)]])
#     offset = np.array([x_idx, y_idx])
#     transformed_rect = np.dot(rect, R) + offset
#     return transformed_rect
#
#
# def set_drop(fire, angle: float, time: int, length=500, width=150, buffer_dist: int = 0,
#              origin: np.ndarray = None) -> Drop:
#     """
#     Create and return a Drop object.
#
#     The drop point and drop angle are computed using get_point_angle.
#     """
#     drop_mid, drop_angle = get_point_angle(fire, angle, time, buffer_dist, origin)
#     return Drop(fire, drop_mid, drop_angle, length, width)
#
#
# def get_point_angle(fire, angle, time, buffer_dist: int = 0, origin=None):
#     """
#     Compute the drop point and drop angle.
#
#     Returns a tuple (point, drop_angle) where point is the drop_mid coordinate and drop_angle is perpendicular
#     to the computed plane of fire growth.
#     """
#     # pt, angle_par, hull = get_point(fire, angle, time, buffer_dist, origin)
#     pt, angle_par, hull = safe_get_point(fire, angle, time, buffer_dist, origin)
#     if pt is None:  # wrapper signalled failure
#         return None, None  # propagate gracefully
#     drop_angle = np.mod(np.mod(angle_par, 360) + 90, 360)
#     return pt, drop_angle
#
#
# def get_point(fire, angle, time, buffer_dist: int = 0, origin=None):
#     """
#     Compute the drop point as the intersection of a ray from the fire's ignition point (or given origin)
#     with the convex hull of the fire contour at the given time (plus an optional buffer).
#
#     Returns a tuple (point, angle_par, hull) where point is the computed drop point, angle_par is the orientation
#     (in degrees) of the intersecting facet, and hull is the convex hull used.
#     """
#     # Attempt to use the surrogate's get_contour if available; otherwise, use a fallback.
#     try:
#         # print(time)
#         contour = fire.get_contour(time)
#     except AttributeError:
#         import matplotlib.pyplot as plt
#         import shapely.geometry
#         fig, ax = plt.subplots()
#         cell_size = fire.transform[0]
#         x_arr = np.arange(fire.bounds.left, fire.bounds.right, cell_size)
#         y_arr = np.arange(fire.bounds.top, fire.bounds.bottom, fire.transform[4])
#         contour_obj = ax.contour(x_arr, y_arr, fire.current_fire(time), levels=[time])
#         lineList = []
#         for level_paths in contour_obj.allsegs:
#             for path in level_paths:
#                 line = shapely.geometry.LineString(path)
#                 lineList.append(line)
#         contour = shapely.geometry.MultiLineString(lineList)
#         plt.close(fig)
#     hull_polygon = contour.convex_hull
#     buff = hull_polygon.buffer(buffer_dist)
#     # Extract coordinates from the buffered hull.
#     points = np.array(buff.exterior.coords)
#     hull = ConvexHull(points)
#     # If no origin is provided, use the surrogate's ignition_pt if available; else default to center.
#     if origin is None:
#         if hasattr(fire, "ignition_pt"):
#             point = np.array(fire.ignition_pt)
#         else:
#             point = np.array([(fire.bounds.left + fire.bounds.right) / 2, (fire.bounds.top + fire.bounds.bottom) / 2])
#     else:
#         point = np.array(origin)
#     unit_ray = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
#     closest_intersection_point = np.zeros((1, 2))
#     closest_distance = np.inf
#     angle_norm = None
#     for plane in hull.equations:
#         normal = plane[:-1]
#         distance = -plane[-1]
#         dot_product = np.dot(normal, unit_ray)
#         if np.abs(dot_product) <= 1e-8:
#             continue
#         t = (distance - np.dot(normal, point)) / dot_product
#         if t >= 0:
#             intersection_point = point + t * unit_ray
#             if np.all(np.dot(normal, intersection_point) <= distance + 10):
#                 distance_to_intersection = np.linalg.norm(intersection_point - point)
#                 if distance_to_intersection < closest_distance:
#                     closest_distance = distance_to_intersection
#                     closest_intersection_point = intersection_point
#                     angle_norm = np.rad2deg(np.arctan2(normal[1], normal[0]))
#     return closest_intersection_point, angle_norm, hull
