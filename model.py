import time as tm
import matplotlib.pyplot as plt
import mesa
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import rasterio
from pyproj import Transformer
import datetime
import random
from collections import namedtuple
import cupy as cp

# from groundcrew import _record_anchor

# Import agent classes.
from airtanker_agents.c130j_agent import C130JAgent
from airtanker_agents.fireherc_agent import FireHercAgent
# from airtanker_agents.CL415_agent import CL415Agent
from airtanker_agents.at_802f_agent import AT802FAgent
from airtanker_agents.Dash8_400MRE_agent import Dash8_400MREAgent
from groundcrew import GroundCrewAgent

# Import the surrogate fire model.
# from surrogate_fire_model import SurrogateFireModel
# from surrogate_fire_model_rothermal_mcts_ncw import SurrogateFireModelROS
from surrogate_fire_model_CK2_multi_phase import SurrogateFireModelROS_CK2Multi
from FIRE_MODEL_CUDA import SurrogateFireModelROS
# from surrogate_fire_model_CK2_multi_phase_optimized import SurrogateFireModelROS_CK2Multi
# Import our new plotting module.
import surrogate_plotting as plotting
import matplotlib.colors as mcolors
from higher_level_planner import GroundCrewPlanner
from airtankeragent import AirtankerAgent

from fireline_abm import compute_fireline_for_sector
from collections import namedtuple
Point = namedtuple("Point", ["x","y"])
import matplotlib.pyplot as plt
import plotly.io as pio
# these defaults will apply whenever you call write_image without width/height
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width  = 1100
pio.kaleido.scope.default_height = 800
# you can also set a default scale (DPI scaling)
pio.kaleido.scope.default_scale = 2

from groundcrew import (
    _record_anchor,
    _travel_minutes,
    _rough_line_minutes,
    # _buffered_finish_cell,   # we'll (re)use this for both start & finish pushes
    _buffered_cell_by_mtt,
    GroundCrewAgent,
)

# --- ground‐crew boundary buffering knobs (minutes of radial spread) ----------
GC_START_BUFFER_MIN        = 500.0    # modest outward push for shared START
GC_START_SAFETY_MIN        = 120       # start = travel_time + this

GC_FINISH_PROBE_MIN        = 400.0    # initial probe when sizing FINISH
GC_FINISH_CLEAR_THRESH_MIN = 600.0    # if rough clear time > thresh → enlarge
GC_FINISH_SAFETY_MIN       = 0      # extra margin when enlarging FINISH
GC_CLEAR_RATE_MULTIPLIER = 2.0
# ------------------------------------------------------------------------------


def _gc_make_finish_cell(model,
                         start_cell: tuple[int, int],
                         far_boundary_idx: int,
                         origin_rc: tuple[int, int] | None):
    """
    Finish point is always dynamic:
      probe → compute (eta_clear, eta_travel) → final_buf = eta_clear + eta_travel + GC_FINISH_SAFETY_MIN
      return boundary cell at that buffer via _buffered_cell_by_mtt()
    """
    # Probe only to size the task (do NOT use probe directly)
    probe_cell = _buffered_cell_by_mtt(model,
                                       far_boundary_idx,
                                       buffer_minutes=model.GC_FINISH_PROBE_MIN)

    eta_clear  = _rough_line_minutes(model, start_cell, probe_cell)
    eta_travel = _travel_minutes(model, start_cell, origin_rc)

    final_buf = eta_clear + eta_travel + model.GC_FINISH_SAFETY_MIN

    return _buffered_cell_by_mtt(model,
                                 far_boundary_idx,
                                 buffer_minutes=final_buf)



class WildfireModel(mesa.Model):
    def __init__(
        self,
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
        enable_plotting=True,
        elapsed_minutes=0,
        groundcrew_count=0,
        groundcrew_speed=None,
        groundcrew_sector_mapping=None,
        second_groundcrew_sector_mapping=None,
        wind_schedule=None,
        fuel_model_override=None,
        num_sectors=4,


    ):
        super().__init__()
        self.enable_plotting = enable_plotting
        self.operational_delay = operational_delay  # Operational delay in minutes
        self.operational_start_time = start_time + datetime.timedelta(minutes=operational_delay)

        self.GC_START_BUFFER_MIN = GC_START_BUFFER_MIN
        self.GC_FINISH_PROBE_MIN = GC_FINISH_PROBE_MIN
        self.GC_FINISH_CLEAR_THRESH_MIN = GC_FINISH_CLEAR_THRESH_MIN
        self.GC_FINISH_SAFETY_MIN = GC_FINISH_SAFETY_MIN
        self.GC_CLEAR_RATE_MULTIPLIER = GC_CLEAR_RATE_MULTIPLIER
        self.GC_START_SAFETY_MIN = GC_START_SAFETY_MIN

        self.groundcrew_count = groundcrew_count
        self.groundcrew_speed = groundcrew_speed
        self.groundcrew_sector_mapping = groundcrew_sector_mapping
        self.case_folder = case_folder  # Case-specific folder for outputs
        self.overall_time_limit = overall_time_limit  # Simulation time limit
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.wind_schedule = wind_schedule
        self.baseline_fire_score = None  # declare it here for clarity
        self.num_sectors = int(num_sectors)


        self.second_groundcrew_sector_mapping = second_groundcrew_sector_mapping
        self.base_positions = base_positions
        self.lake_positions = lake_positions
        self.fire_spread_sim_time = fire_spread_sim_time
        self.time_step = time_step
        self.debug = debug
        self.elapsed_minutes = elapsed_minutes


        # Initialize the fire model.

        self.fire = self.initialize_fire(fuel_model_override)

        self.containment = False
        self.drops = []
        self.time = elapsed_minutes
        self.recalculate_flag = False

        self.start_time = start_time
        self.current_time = start_time + datetime.timedelta(minutes=elapsed_minutes)

        self.agent_counts = airtanker_counts
        self.drop_number = 1

        # ── model-wide bookkeeping ───────────────────────────────────
        self.contained_sectors: set[int] = set()
        self.retarded_sectors: set[int] = set()  # aircraft drops

        self._pair_sync_done = False
        self._pair_sync_list = []

        # Get spatial bounds from the surrogate fire model.
        self.bounds = self.fire.bounds
        print(f"WildfireModel: Bounds: {self.bounds}")
        buffer = 10000  # 10,000 meters buffer
        self.space_width = (self.bounds.right - self.bounds.left) + 2 * buffer
        self.space_height = (self.bounds.top - self.bounds.bottom) + 2 * buffer

        # Update bounds to include the buffer.
        self.buffered_bounds = self.bounds._replace(
            left=self.bounds.left - buffer,
            right=self.bounds.right + buffer,
            bottom=self.bounds.bottom - buffer,
            top=self.bounds.top + buffer,
        )
        print(f"Checking Mesa Space Dimensions: Width = {self.space_width}, Height = {self.space_height}")

        print(f"Buffered Bounds: {self.buffered_bounds}")
        # Reproject the bounds to WGS84.
        with rasterio.open("default4003.tif") as src:
            original_crs = src.crs
            print(f"Original CRS: {original_crs}")
        if original_crs:
            transformer = Transformer.from_crs(original_crs, "EPSG:4326", always_xy=True)
            self.wgs84_bounds = {
                "left": transformer.transform(self.buffered_bounds.left, self.buffered_bounds.bottom)[0],
                "bottom": transformer.transform(self.buffered_bounds.left, self.buffered_bounds.bottom)[1],
                "right": transformer.transform(self.buffered_bounds.right, self.buffered_bounds.top)[0],
                "top": transformer.transform(self.buffered_bounds.right, self.buffered_bounds.top)[1],
            }
        else:
            raise ValueError("TIFF file has no defined CRS, manual input of CRS might be required.")
        self.update_sector_splits()
        # store the initial center and angle ranges forever
        self.fixed_center = self.sector_center
        self.fixed_angle_ranges = self.sector_angle_ranges
        # # Compute initial sector splits for agent targeting.
        # sector_angle_ranges, sector_boundaries, center, max_distance = self.compute_sector_splits(0, num_sectors=4)
        # self.sector_angle_ranges = sector_angle_ranges
        # self.sector_boundaries = sector_boundaries
        # self.sector_center = center

        # # Compute the center of the buffered bounds (i.e., the mesa world center)
        # mesa_center = (
        #     (self.buffered_bounds.left + self.buffered_bounds.right) / 2,
        #     (self.buffered_bounds.bottom + self.buffered_bounds.top) / 2
        # )
        # # Convert each relative base position (e.g., (10000, 10000)) into an absolute coordinate
        # self.base_positions = [
        #     (mesa_center[0] + bx, mesa_center[1] + by) for (bx, by) in self.base_positions
        # ]

        self.space = mesa.space.ContinuousSpace(
            self.space_width,
            self.space_height,
            False,
            self.buffered_bounds.left,
            self.buffered_bounds.bottom,
        )
        # self.space = mesa.space.ContinuousSpace(
        #     x_max=self.buffered_bounds.right,
        #     y_max=self.buffered_bounds.top,
        #     torus=False
        # )
        self.schedule = mesa.time.RandomActivation(self)

        # Instantiate the high‐level planning module.
        # self.groundcrew_planner = GroundCrewPlanner(
        #     model=self,
        #     buffer_dist_start=2000,
        #     buffer_dist_finish=2000
        # )
        # self.groundcrew_planner = GroundCrewPlanner(
        #     model=self,
        #     buffer_time_start=500,  # e.g., 10 minutes (adjust as needed)
        #     buffer_time_finish=500
        # )
        # # Precompute planned cells from the planner.
        # planned_groundcrew_cells = self.groundcrew_planner.get_planned_cells()

        # only instantiate and run the planner if you actually have crews
        # if self.groundcrew_count and self.groundcrew_count > 0:
        #     self.groundcrew_planner = GroundCrewPlanner(
        #         self,
        #         buffer_time_start=300,
        #         buffer_time_finish=300,
        #     )
        #     self.planned_groundcrew_cells = self.groundcrew_planner.get_planned_cells()
        # else:
        #     self.planned_groundcrew_cells = []


        available_bases = list(self.base_positions)
        print(f"Available bases: {available_bases}")
        self.agent_colors = {}
        self.agent_shapes = {}
        hex_colors = [
            "#3357FF", "#FF33A8", "#75FF33", "#FFBD33", "#33FFF3",
            "#8D33FF", "#FF3333", "#33FF8D", "#FF69B4", "#FFD700",
            "#00FA9A", "#FF4500", "#00CED1", "#DA70D6", "#4682B4",
        ]
        symbols = {
            "C130J": "circle",
            "FireHerc": "triangle-up",
            "Scooper": "cross",
            "AT802F": "diamond",
            "Dash8_400MRE": "square",
        }

        agent_id = 0
        for agent_type, count in self.agent_counts.items():
            for _ in range(count):
                base_position = random.choice(available_bases)
                if agent_type == "C130J":
                    agent = C130JAgent(agent_id, self, base_position, time_step)
                elif agent_type == "FireHerc":
                    agent = FireHercAgent(agent_id, self, base_position, time_step)
                # elif agent_type == "FireHerc":
                #     agent = FireHercAgent(agent_id, self, base_position, time_step)
                #     # Manually assign FireHerc to sector 1 (i.e., second sector)
                #     agent.assigned_sector = ("FireHerc", 0)
                #     print(f"Assigned FireHerc agent {agent_id} to sector 1")

                elif agent_type == "Scooper":
                    print('scooper')

                    # agent = CL415Agent(agent_id, self, base_position, time_step)
                elif agent_type == "AT802F":
                    agent = AT802FAgent(agent_id, self, base_position, time_step)
                elif agent_type == "Dash8_400MRE":
                    agent = Dash8_400MREAgent(agent_id, self, base_position, time_step)
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")

                self.agent_colors[agent_id] = hex_colors[agent_id % len(hex_colors)]
                self.agent_shapes[agent_id] = symbols.get(agent_type, "circle")
                self.schedule.add(agent)
                corrected_position = self.correct_position(base_position)
                print(f"Corrected position: {corrected_position}")
                self.space.place_agent(agent, base_position)

                agent_id += 1


        # now add ground-crew agents
        # for _ in range(self.groundcrew_count):
        #     gc = GroundCrewAgent(agent_id, self, speed=self.groundcrew_speed)
        #     # gc = GroundCrewAgent(agent_id, self, speed=self.groundcrew_speed,
        #     #                      sector_index=agent_id % len(self.sector_boundaries))
        #
        #     # give them a distinctive color & symbol
        #     self.agent_colors[agent_id] = "#00FF00"
        #     self.agent_shapes[agent_id] = "x"
        #     self.schedule.add(gc)
        #     agent_id += 1
        # for i in range(self.groundcrew_count):
        #     sector_index = i % len(self.sector_angle_ranges)
        #     gc = GroundCrewAgent(
        #         agent_id,
        #         self,
        #         speed_m_per_min=self.groundcrew_speed,
        #         sector_index=sector_index,  # start point computed internally
        #         buffer_dist= 1_000,
        #         line_width=1,
        #     )
        #     self.agent_colors[agent_id] = "#00FF00"
        #     self.agent_shapes[agent_id] = "x"
        #     self.schedule.add(gc)
        #     agent_id += 1
        # Create groundcrew agents. Their planned start and finish cells come from the planner.
        # for i in range(self.groundcrew_count):
        #     sector_index = i % len(self.sector_angle_ranges)
        #     planned = planned_groundcrew_cells[sector_index]
        #     gc = GroundCrewAgent(
        #         agent_id,
        #         self,
        #         speed_m_per_min=self.groundcrew_speed,
        #         sector_index=sector_index,
        #         buffer_dist=1000,
        #         line_width=1,
        #         buffer_time=1500,
        #         finish_extra_buffer=500,
        #         planned_start=planned["start_cell"],
        #         planned_finish=planned["finish_cell"]
        #     )
        #     self.agent_colors[agent_id] = "#00FF00"
        #     self.agent_shapes[agent_id] = "x"
        #     self.schedule.add(gc)
        #     agent_id += 1

        # #initial gc logic
        # if hasattr(self, "groundcrew_sector_mapping"):
        #     mapping = self.groundcrew_sector_mapping
        # else:
        #     # If no mapping is provided, default to sequential assignment
        #     mapping = list(range(self.groundcrew_count))
        #
        # for i in range(self.groundcrew_count):
        #     # Here, mapping[i] is the sector index that this groundcrew agent should be assigned to.
        #     sector_index = mapping[i]
        #     planned = planned_groundcrew_cells[sector_index]
        #     gc = GroundCrewAgent(
        #         agent_id,
        #         self,
        #         speed_m_per_min=self.groundcrew_speed,
        #         sector_index=sector_index,
        #         buffer_dist=1000,
        #         line_width=1,
        #         buffer_time=1500,
        #         finish_extra_buffer=500,
        #         planned_start=planned["start_cell"],
        #         planned_finish=planned["finish_cell"]
        #     )
        #     self.agent_colors[agent_id] = "#00FF00"
        #     self.agent_shapes[agent_id] = "x"
        #     self.schedule.add(gc)
        #     # Record initial start and finish to used_boundary_points
        #     start_boundary = (sector_index - 1) % len(self.sector_boundaries)
        #     finish_boundary = sector_index
        #
        #     if not hasattr(self, "used_boundary_points"):
        #         self.used_boundary_points = {}
        #
        #     for b_idx, cell in [(start_boundary, planned["start_cell"]), (finish_boundary, planned["finish_cell"])]:
        #         if b_idx not in self.used_boundary_points:
        #             self.used_boundary_points[b_idx] = []
        #         self.used_boundary_points[b_idx].append(cell)
        #         print(f"★ INIT recorded point {cell} on boundary {b_idx}")
        #     agent_id += 1
        # ────────────────────────────────────────────────────────────────
        # Ground-crew agents
        # ────────────────────────────────────────────────────────────────
        mapping = self.groundcrew_sector_mapping or []  # may be empty

        for i in range(self.groundcrew_count):
            # ── decide the crew’s initial sector ─────────────────────────
            if i < len(mapping):  # user supplied
                sector_index = mapping[i]
            else:  # no mapping → give each
                sector_index = i % len(self.sector_boundaries)  # a harmless default

            # planned = planned_groundcrew_cells[sector_index]
            # start_cell = planned["start_cell"]
            # finish_cell = planned["finish_cell"]

            # ── instantiate the agent ────────────────────────────────────
            # gc = GroundCrewAgent(
            #     agent_id,
            #     self,
            #     speed_m_per_min=self.groundcrew_speed,
            #     sector_index=sector_index,
            #     buffer_dist=1000,
            #     line_width=1,
            #     buffer_time=1500,
            #     finish_extra_buffer=500,
            #     planned_start=start_cell,
            #     planned_finish=finish_cell,
            # )
            # ── instantiate the agent ────────────────────────────────────
            gc = GroundCrewAgent(
                agent_id,
                self,
                speed_m_per_min=self.groundcrew_speed,
                sector_index=sector_index,

            )

            # start every crew in “Standby” so they won’t walk until the
            # planner (or MCTS) re-assigns them
            gc.state = "Standby"

            # ── visuals & bookkeeping ────────────────────────────────────
            self.agent_colors[agent_id] = "#00FF00"
            self.agent_shapes[agent_id] = "x"
            self.schedule.add(gc)

            # record the start / finish points so later crews can reuse them
            start_boundary = (sector_index - 1) % len(self.sector_boundaries)
            finish_boundary = sector_index

            if not hasattr(self, "used_boundary_points"):
                self.used_boundary_points = {}
            # for b_idx, cell in [(start_boundary, start_cell),
            #                     (finish_boundary, finish_cell)]:
            #     self.used_boundary_points.setdefault(b_idx, []).append(cell)
            #     print(f"★ INIT recorded point {cell} on boundary {b_idx}")
            # ( anchors are already stored by GroundCrewAgent.__init__ – nothing else to do )

            agent_id += 1

        self.plot_fig = go.Figure()

    def correct_position(self, position):
        epsilon = 1e-3  # Use a larger epsilon than 1e-6
        x, y = position
        x = min(max(x, self.buffered_bounds.left), self.buffered_bounds.right - epsilon)
        y = min(max(y, self.buffered_bounds.bottom), self.buffered_bounds.top - epsilon)
        return (x, y)



    # def _register_first_pair(self, gc):
    #     if self._pair_sync_done:
    #         return
    #
    #     self._pair_sync_list.append(gc)
    #     if len(self._pair_sync_list) < 2:
    #         return  # wait until we have two crews
    #
    #     gc1, gc2 = self._pair_sync_list
    #
    #     def _flip(g):
    #         g.planned_start, g.planned_finish = g.planned_finish, g.planned_start
    #         g.start_boundary, g.finish_boundary = g.finish_boundary, g.start_boundary
    #         g.start_rc = g.planned_start
    #         _record_anchor(g.model, g.start_boundary, g.planned_start, "start")
    #         _record_anchor(g.model, g.finish_boundary, g.planned_finish, "finish")
    #         print(f"[SYNC]  GC#{g.unique_id:02d}  start/finish flipped")
    #
    #     # If they already share the same START, do nothing.
    #     if gc1.planned_start == gc2.planned_start:
    #         pass
    #     # If one crew’s START equals the other’s FINISH, flip that crew.
    #     elif gc1.planned_start == gc2.planned_finish:
    #         _flip(gc2)
    #     elif gc2.planned_start == gc1.planned_finish:
    #         _flip(gc1)
    #
    #     self._pair_sync_done = True

    def _register_first_pair(self, gc: "GroundCrewAgent") -> None:
        """
        One-time sync of the *first two* ground crews.

        If the two first crews to report are adjacent sectors on the ring:
          • CLEAR *all* previously recorded anchors (pre-plan noise).
          • Create ONE shared START anchor on their shared boundary.
          • Create ONE FINISH anchor for each crew on its far boundary
            (probe/extend via _gc_make_finish_cell()).
          • Overwrite each crew's planning fields accordingly.

        If they are *not* adjacent, do nothing (keep existing anchors) and
        close the gate so we don't try again.
        """

        if self._pair_sync_done:
            return

        # collect until we have two crews
        self._pair_sync_list.append(gc)
        if len(self._pair_sync_list) < 2:
            return

        gc1, gc2 = self._pair_sync_list[:2]
        s1, s2 = gc1.sector_index, gc2.sector_index
        n_bdys = len(self.sector_boundaries)

        # --- adjacency test (sector i spans boundary i → i+1) -----------------
        # shared boundary is the *right* boundary of the lower-index sector
        if (s1 + 1) % n_bdys == s2:
            shared_bdy = (s1 + 1) % n_bdys
        elif (s2 + 1) % n_bdys == s1:
            shared_bdy = (s2 + 1) % n_bdys
        else:
            print(f"[SYNC] first two crews not adjacent ({s1},{s2}) – no sync.")
            self._pair_sync_done = True
            return

        # --- purge ALL old anchors (remove stale constructor/early-plan noise) --
        if hasattr(self, "boundary_anchors"):
            self.boundary_anchors.clear()
        else:
            self.boundary_anchors = {}

        # optional: reset a sequence counter if you add one for tie‑breaking
        self._anchor_seq = 0

        # --- compute shared START rc ------------------------------------------
        # shared_start_rc = _buffered_finish_cell(
        #     self, shared_bdy, buffer_minutes=GC_START_BUFFER_MIN
        # )
        shared_start_rc = _buffered_cell_by_mtt(
            self, shared_bdy, buffer_minutes=self.GC_START_BUFFER_MIN
        )
        print("recording from register first pair")
        _record_anchor(self, shared_bdy, shared_start_rc, "start")

        def _apply(gcrew: "GroundCrewAgent"):
            sec = gcrew.sector_index
            # sector i spans [i, i+1); far boundary is the one NOT shared
            left_bdy = sec % n_bdys
            right_bdy = (sec + 1) % n_bdys
            far_bdy = right_bdy if shared_bdy == left_bdy else left_bdy

            origin_rc = getattr(gcrew, "grid_rc", None)
            finish_rc = _gc_make_finish_cell(
                self,
                start_cell=shared_start_rc,
                far_boundary_idx=far_bdy,
                origin_rc=origin_rc,
            )
            print("recording from register first pair")
            _record_anchor(self, far_bdy, finish_rc, "finish")

            # overwrite crew fields
            gcrew.planned_start = shared_start_rc
            gcrew.planned_finish = finish_rc
            gcrew.start_boundary = shared_bdy
            gcrew.finish_boundary = far_bdy
            gcrew.start_rc = shared_start_rc
            gcrew.state = "ToStart"
            gcrew.path.clear()
            gcrew.path_times.clear()
            gcrew.last_cell_idx = -1
            gcrew._eta_remaining = 0.0
            gcrew.replan_disabled = False

            print(f"[SYNC] GC#{gcrew.unique_id:02d} sect{sec} "
                  f"START b{shared_bdy} {shared_start_rc}  "
                  f"FINISH b{far_bdy} {finish_rc}")

        _apply(gc1)
        _apply(gc2)

        self._pair_sync_done = True

    # def update_sector_splits(self):
    #     # This method updates the sector splits regardless of whether we plot.
    #     sector_angle_ranges, sector_boundaries, center, max_distance = self.compute_sector_splits(self.time,
    #                                                                                               num_sectors=4)
    #     self.sector_angle_ranges = sector_angle_ranges
    #     self.sector_boundaries = sector_boundaries
    #     self.sector_center = center

    # ------------------------------------------------------------------
    #  Frozen-centre sector geometry
    # ------------------------------------------------------------------
    def update_sector_splits(self):
        """
        • First ever call
              – run full compute_sector_splits()
              – store centre & angle ranges as immutable
        • Later calls
              – keep the same centre & angles
              – recompute only the boundary end-points so the rays
                extend to the current fire perimeter.
        """
        if not hasattr(self, "fixed_center"):
            # ── INITIALISE (once) ─────────────────────────────────────
            ang, bnd, ctr, _ = self.compute_sector_splits(
                self.time, num_sectors=self.num_sectors)

            self.fixed_center       = ctr        # freeze!
            self.fixed_angle_ranges = ang

            self.sector_center      = ctr
            self.sector_angle_ranges= ang
            self.sector_boundaries  = bnd
        else:
            # ── REFRESH BOUNDARIES ONLY ──────────────────────────────
            self.sector_center      = self.fixed_center
            self.sector_angle_ranges= self.fixed_angle_ranges

            # recompute boundary points at the current radius
            self.sector_boundaries, _ = \
                self._recompute_boundaries_from_fixed_center(self.time)


    # def update_sector_splits(self):
    #     # On the very first call, fixed_center doesn’t exist yet,
    #
    #     if not hasattr(self, "fixed_center"):
    #         sector_angle_ranges, sector_boundaries, center, max_distance = \
    #             self.compute_sector_splits(self.time, num_sectors=4)
    #         self.sector_angle_ranges = sector_angle_ranges
    #         self.sector_boundaries   = sector_boundaries
    #         self.sector_center       = center
    #         # store for all future calls
    #         self.fixed_center        = center
    #         self.fixed_angle_ranges  = sector_angle_ranges
    #     else:
    #         # every other time, reuse fixed_center & fixed_angle_ranges
    #         self.sector_angle_ranges = self.fixed_angle_ranges
    #         self.sector_center       = self.fixed_center
    #         self.sector_boundaries, _ = \
    #             self._recompute_boundaries_from_fixed_center(self.time)


    # def initialize_fire(self):
    #     # Use the surrogate fire model instead of the old wff-firesims modules.
    #     tif_path = Path().joinpath("TEST_TIF.tif").resolve()
    #     # return SurrogateFireModel(
    #     #     tif_path=tif_path,
    #     #     sim_time=self.fire_spread_sim_time,
    #     #     spread_rate=0.6,
    #     #     wind_influence=0.5,
    #     #     wind_speed=self.wind_speed,
    #     #     wind_direction=self.wind_direction,
    #     #     uphill_bonus=0.05,
    #     #     downhill_penalty=0.02,
    #     #     max_iter=450,
    #     #     tol=1e-4
    #     # )
    #
    #     # return SurrogateFireModelROS(
    #     #     tif_path=tif_path,
    #     #     sim_time=self.fire_spread_sim_time,
    #     #     wind_speed=self.wind_speed,
    #     #     wind_direction_deg=self.wind_direction,
    #     #     max_iter=450,
    #     #     tol=1e-4
    #     # )
    #
    #
    #     return SurrogateFireModelROS_CK2Multi(
    #         tif_path=tif_path,
    #         sim_time=self.fire_spread_sim_time,
    #         wind_speed=self.wind_speed,
    #         wind_direction_deg=self.wind_direction,
    #         max_iter=250,
    #         tol=1e-4,
    #         wind_schedule= self.wind_schedule
    #     )
    def initialize_fire(self, fuel_model_override=None):
        tif_path = Path().joinpath("cali_test_big_enhanced.tif").resolve()

        # return SurrogateFireModelROS_CK2Multi(
        #     tif_path=tif_path,
        #     sim_time=self.fire_spread_sim_time,
        #     wind_speed=self.wind_speed,
        #     wind_direction_deg=self.wind_direction,
        #     max_iter=250,
        #     tol=1e-4,
        #     wind_schedule=self.wind_schedule,
        #     fuel_model_override=fuel_model_override,
        # )

        return SurrogateFireModelROS(
            tif_path=tif_path,
            sim_time=self.fire_spread_sim_time,
            wind_speed=self.wind_speed,
            wind_direction_deg=self.wind_direction,
            max_iter=60,
            tol=1e-4,
            wind_schedule=self.wind_schedule,
            fuel_model_override=fuel_model_override,
        )




    def plot_fire(self):
        self.plot_fig = go.Figure()
        # self.plot_fig.update_layout(
        #     xaxis=dict(range=[self.buffered_bounds.left, self.buffered_bounds.right]),
        #     yaxis=dict(range=[self.buffered_bounds.bottom, self.buffered_bounds.top]),
        #     template="plotly_dark",
        #     uirevision="foo",
        #     title=f"Wildfire Simulation - Time: {self.time:.2f} minutes ",
        # )
        self.plot_fig.update_layout(
            title=dict(
                text=f"Wildfire State (t = {self.time:.0f} min)",
                font=dict(size=26, color="black"),
                x=0.5,
            ),

            # ← IMPORTANT: zoom to the TIFF, not the padded world
            xaxis=dict(
                range=[self.fire.bounds.left, self.fire.bounds.right],
                constrain="domain",
                scaleanchor="y",  # keep aspect ratio 1:1
                scaleratio=1,
                fixedrange=True,  # no autoscale on export
                title="X Coordinate (m)",
            ),
            yaxis=dict(
                range=[self.fire.bounds.bottom, self.fire.bounds.top],
                constrain="domain",
                fixedrange=True,
                title="Y Coordinate (m)",
            ),

            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
        )

        # # ——— Save a high-res PNG of this frame ———

        vmin = 0
        vmax = 204

        # Create a custom color scale for the fuel model using our surrogate_plotting module.
        cmaps = plotting.Cmaps()
        fuel_model_colors = cmaps.fuel_cmap.colors
        fuel_model_values = np.linspace(vmin, vmax, len(fuel_model_colors))
        fuel_model_color_scale = [
            [(value - vmin) / (vmax - vmin), mcolors.to_hex(color)]
            for value, color in zip(fuel_model_values, fuel_model_colors)
        ]

        # Use the surrogate's fuel_model and spatial attributes.
        fuel_model_data = self.fire.fuel_model
        x = np.arange(self.fire.bounds.left, self.fire.bounds.right, self.fire.transform[0])
        y = np.arange(self.fire.bounds.top, self.fire.bounds.bottom, self.fire.transform[4])
        self.plot_fig.add_trace(
            go.Heatmap(
                z=fuel_model_data,
                x=x,
                y=y,
                colorscale=fuel_model_color_scale,
                zmin=vmin,
                zmax=vmax,
                showscale=False,
                name="Fuel Model",
            )
        )

        # Build a colorscale for fire spread.
        f_cmap = cmaps.fire_cmap
        colorscale = []
        for i in range(f_cmap.N):
            rgba = f_cmap(i)
            colorscale.append([i / (f_cmap.N - 1), mcolors.to_hex(rgba)])

        # Get the current fire state from the surrogate.
        current_fire_data = self.fire.current_fire(self.time, max_time=self.fire_spread_sim_time)

        self.plot_fig.add_trace(
            go.Heatmap(
                z=current_fire_data,
                x=x,
                y=y,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title="Fire Arrival Time"),
                name="Fire Spread",
            )
        )

        # # Plot each agent's path and current position.
        # for agent in self.schedule.agents:
        #     color = self.agent_colors[agent.unique_id]
        #     shape = self.agent_shapes[agent.unique_id]
        #     if agent.previous_positions:
        #         prev_x, prev_y = zip(*agent.previous_positions)
        #         self.plot_fig.add_trace(
        #             go.Scatter(
        #                 x=prev_x,
        #                 y=prev_y,
        #                 mode="markers",
        #                 marker=dict(color=color, size=2, opacity=0.5),
        #                 name=f"Agent {agent.unique_id} Path (Flying)",
        #             )
        #         )
        #     self.plot_fig.add_trace(
        #         go.Scatter(
        #             x=[agent.position[0]],
        #             y=[agent.position[1]],
        #             mode="markers",
        #             marker=dict(color=color, size=10, symbol=shape),
        #             name=f"Agent {agent.unique_id} Position",
        #         )
        #     )
        #
        # Plot ground‐crew agents as lime “×” markers
        # for agent in self.schedule.agents:
        #     if isinstance(agent, GroundCrewAgent):
        #         self.plot_fig.add_trace(
        #             go.Scatter(
        #                 x=[agent.position[0]],
        #                 y=[agent.position[1]],
        #                 mode="markers",
        #                 marker=dict(color="lime", size=8, symbol="x"),
        #                 name=f"Ground Crew {agent.unique_id}",
        #             )
        #         )
        # ── Legend markers (no data, just symbols) ────────────────────────────
        self.plot_fig.add_trace(
            go.Scatter(
                x=[None], y=[None],  # no points → nothing on the canvas
                mode="markers",
                marker=dict(color="blue", size=12, symbol="triangle-up"),
                name="Aircraft",  # shown in legend only
                showlegend=True
            )
        )
        # self.plot_fig.add_trace(
        #     go.Scatter(
        #         x=[None], y=[None],
        #         mode="markers",
        #         marker=dict(color="lime", size=12, symbol="x"),
        #         name="Ground Crew",
        #         showlegend=True
        #     )
        # )
        # First, plot airtankers (as before)…
        for agent in self.schedule.agents:
            if not isinstance(agent, GroundCrewAgent):
                color = self.agent_colors[agent.unique_id]
                shape = self.agent_shapes[agent.unique_id]
                if agent.previous_positions:
                    prev_x, prev_y = zip(*agent.previous_positions)
                    self.plot_fig.add_trace(
                        go.Scatter(
                            x=prev_x, y=prev_y,
                            mode="markers",
                            marker=dict(color=color, size=2, opacity=0.5),
                            name=f"Agent {agent.unique_id} Path (Flying)",
                            showlegend=False,
                        )
                    )
                self.plot_fig.add_trace(
                    go.Scatter(
                        x=[agent.position[0]],
                        y=[agent.position[1]],
                        mode="markers",
                        marker=dict(color=color, size=10, symbol=shape),
                        name=f"Agent {agent.unique_id} Position",
                        showlegend=False,
                    )
                )

        # Then plot ground-crew history & current pos
        for agent in self.schedule.agents:
            if isinstance(agent, GroundCrewAgent):
                # history as small stars
                if agent.previous_positions:
                    hx, hy = zip(*agent.previous_positions)
                    self.plot_fig.add_trace(
                        go.Scatter(
                            x=hx, y=hy,
                            mode="markers",
                            marker=dict(color="yellow", size=1, opacity=0.5,symbol="star"),
                            name=f"Ground Crew {agent.unique_id} Trail",
                            showlegend=False,
                        )
                    )
                # current as “x”
                self.plot_fig.add_trace(
                    go.Scatter(
                        x=[agent.position[0]],
                        y=[agent.position[1]],
                        mode="markers",
                        marker=dict(color="lime", size=10, symbol="x"),
                        name=f"Ground Crew {agent.unique_id}",
                        showlegend=False,
                    )
                )
        # Compute sector splits; if the surrogate doesn’t provide get_contour(),
        # we use a fallback based on matplotlib contouring.
        # sector_angle_ranges, sector_boundaries, center, max_distance = self.compute_sector_splits(self.time, num_sectors=4)
        # self.sector_angle_ranges = sector_angle_ranges
        # self.sector_boundaries = sector_boundaries
        # self.sector_center = center


        # self.update_sector_splits()
        # for i, boundary_point in enumerate(self.sector_boundaries):
        #     self.plot_fig.add_trace(
        #         go.Scatter(
        #             x=[self.sector_center.x, boundary_point.x],
        #             y=[self.sector_center.y, boundary_point.y],
        #             mode='lines',
        #             line=dict(color='yellow', dash='dash'),
        #             name=f'Sector boundary {i + 1}',
        #             showlegend=False,
        #         )
        #     )

        self.update_sector_splits()
        for i, boundary_point in enumerate(self.sector_boundaries):
            self.plot_fig.add_trace(
                go.Scatter(
                    x=[self.sector_center.x, boundary_point.x],
                    y=[self.sector_center.y, boundary_point.y],
                    mode="lines",
                    line=dict(
                        color="rgba(0,0,0,0.5)",  # medium gray
                        width=3,
                        dash=None,  # solid
                    ),

            name=f"Sector boundary {i + 1}",
                    showlegend=False,
                )
            )

        # self.plot_fig.update_layout(
        #     title=f"Wildfire Simulation - Time: {self.current_time.strftime('%H:%M')} (elapsed {self.time} minutes) - Containment: ",
        #     xaxis_title="X Coordinate (meters)",
        #     yaxis_title="Y Coordinate (meters)",
        #     legend_title="Legend",
        # )
        # self.plot_fig.update_layout(
        #     title=(
        #         f"Wildfire Simulation – Time: "
        #         f"{self.current_time.strftime('%H:%M')} "
        #         f"(elapsed {self.time} min)"
        #     ),
        #     xaxis_title="X Coordinate (m)",
        #     yaxis_title="Y Coordinate (m)",
        #     legend=dict(
        #         orientation="h",  # horizontal row
        #         yanchor="bottom",
        #         y=1.02,  # just above the top of plotting area
        #         xanchor="left",
        #         x=0,
        #         bgcolor="rgba(0,0,0,0)"  # transparent background
        #     ),
        #     legend_title_text="",  # no extra “Legend” box
        # )
        self.plot_fig.update_layout(
            # nice big centered title for both dashboard + saved PNG
            title=dict(
                text=(
                    f"Wildfire Simulation – "
                    f"{self.current_time.strftime('%H:%M')} "
                    f"(elapsed {self.time:.0f} min)"
                ),
                font=dict(size=28, color="black"),  # bump size here
                x=0.5,
            ),

            # axis labels
            xaxis_title="X Coordinate (m)",
            yaxis_title="Y Coordinate (m)",

            # white theme for publication
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),

            # legend styling
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="rgba(0,0,0,0)",
            ),
            legend_title_text="",

            # preserve zoom, legend toggles, etc. between dashboard refreshes
            uirevision="foo",
        )

        # 1) zoom into the center half of the domain
        # xb = (self.buffered_bounds.left + self.buffered_bounds.right) / 4
        # yb = (self.buffered_bounds.top + self.buffered_bounds.bottom) / 4
        # xrad = (self.buffered_bounds.right - self.buffered_bounds.left) / 8
        # yrad = (self.buffered_bounds.top - self.buffered_bounds.bottom) / 8
        #
        # self.plot_fig.update_xaxes(range=[xb - xrad, xb + xrad])
        # self.plot_fig.update_yaxes(range=[yb - yrad, yb + yrad])

        # 2) save a super-high-res PNG
        # if int(self.time) % 5 == 0:
        #     img_dir = self.case_folder / "images"
        #     img_dir.mkdir(exist_ok=True)
        #     out_path = img_dir / f"{int(self.time):05d}nozoom_air.png"
        #     self.plot_fig.write_image(
        #         str(out_path),
        #         format="png",
        #         width=800,
        #         height=600,
        #         # scale=4
        #         scale=10  # ← text and all markers will render twice as large
        #     )

    def recalculate_cruise_speed(self, event):
        self.time_step = event.new
        self.recalculate_flag = True
        print(f"Cruise speeds recalculated for all agents with time_step: {self.time_step}")

    # def compute_sector_splits(self, time, num_sectors=4):
    #     try:
    #         contour = self.fire.get_contour(time)
    #     except AttributeError:
    #         import matplotlib.pyplot as plt
    #         import shapely.geometry
    #         fig, ax = plt.subplots()
    #         x = np.arange(self.fire.bounds.left, self.fire.bounds.right, self.fire.transform[0])
    #         y = np.arange(self.fire.bounds.top, self.fire.bounds.bottom, self.fire.transform[4])
    #         contour_obj = ax.contour(x, y, self.fire.current_fire(time), levels=[time])
    #         lineList = []
    #         for level_paths in contour_obj.allsegs:
    #             for path in level_paths:
    #                 line = shapely.geometry.LineString(path)
    #                 lineList.append(line)
    #         contour = shapely.geometry.MultiLineString(lineList)
    #         plt.close(fig)
    #     hull_polygon = contour.convex_hull
    #     if hull_polygon.geom_type == 'Point':
    #         center = hull_polygon
    #         max_distance = 100
    #     else:
    #         center = hull_polygon.centroid
    #         hull_coords = list(hull_polygon.exterior.coords)
    #         max_distance = max(np.sqrt((center.x - x_)**2 + (center.y - y_)**2) for x_, y_ in hull_coords)
    #     sector_width = 360 / num_sectors
    #     sector_angle_ranges = []
    #     sector_boundaries = []
    #     from shapely.geometry import LineString, Point
    #     for i in range(num_sectors):
    #         lower_bound = i * sector_width
    #         upper_bound = lower_bound + sector_width
    #         sector_angle_ranges.append((lower_bound, upper_bound))
    #         angle_rad = np.deg2rad(lower_bound)
    #         far_point = (center.x + max_distance * np.cos(angle_rad),
    #                      center.y + max_distance * np.sin(angle_rad))
    #         ray = LineString([(center.x, center.y), far_point])
    #         if hull_polygon.geom_type != 'Point':
    #             intersection = hull_polygon.intersection(ray)
    #             if intersection.geom_type == 'Point':
    #                 boundary_point = intersection
    #             elif intersection.geom_type == 'MultiPoint':
    #                 points = list(intersection)
    #                 distances = [np.sqrt((center.x - pt.x)**2 + (center.y - pt.y)**2) for pt in points]
    #                 boundary_point = points[np.argmax(distances)]
    #             else:
    #                 boundary_point = Point(far_point)
    #         else:
    #             boundary_point = Point(far_point)
    #         sector_boundaries.append(boundary_point)
    #     return sector_angle_ranges, sector_boundaries, center, max_distance

    from collections import namedtuple
    import numpy as np
    def _recompute_boundaries_from_fixed_center(self, time):
        """Return (sector_boundaries, max_distance) for current fire shape,
           but always from self.fixed_center & self.fixed_angle_ranges."""

        from collections import namedtuple
        Point = namedtuple("Point", ["x", "y"])

        # get the burning‐cell mask as before
        fire_data = self.fire.current_fire(time, max_time=self.fire_spread_sim_time)
        mask = fire_data <= time
        if not np.any(mask):
            # no fire yet: fallback radius
            max_distance = 100.0
        else:
            # build coordinate arrays
            x = np.arange(self.fire.bounds.left, self.fire.bounds.right, self.fire.transform[0])
            y = np.arange(self.fire.bounds.top, self.fire.bounds.bottom, self.fire.transform[4])
            rows, cols = np.nonzero(mask)
            burning_x = x[cols]
            burning_y = y[rows]
            # vectorized distances to the fixed center
            dx = burning_x - self.fixed_center.x
            dy = burning_y - self.fixed_center.y
            max_distance = np.sqrt(dx*dx + dy*dy).max()

        # now shoot rays at each fixed angle
        boundaries = []
        for lower, _ in self.fixed_angle_ranges:
            θ = np.deg2rad(lower)
            bx = self.fixed_center.x + max_distance * np.cos(θ)
            by = self.fixed_center.y + max_distance * np.sin(θ)
            boundaries.append(Point(bx, by))
        return boundaries, max_distance

    # def notify_drop(self, airtanker: "AirtankerAgent") -> None:
    #     sector_idx = airtanker.assigned_sector[1]
    #     print(f"[DEBUG] AT#{airtanker.unique_id:02d}  finished drop "
    #           f"in sector {sector_idx} – notifying ground crews")
    #
    #     for agent in self.schedule.agents:
    #         if (isinstance(agent, GroundCrewAgent)
    #                 and agent.sector_index == sector_idx
    #                 and agent.state == "BuildLine"):
    #             agent.recalc_path_from_here()

    def notify_drop(self, airtanker: "AirtankerAgent") -> None:
        sector_idx = airtanker.assigned_sector[1]
        print(f"[DEBUG] AT#{airtanker.unique_id:02d}  finished drop in sector "
              f"{sector_idx} – notifying ground crews")

        for agent in self.schedule.agents:
            if (isinstance(agent, GroundCrewAgent)
                    and agent.sector_index == sector_idx
                    and agent.state == "BuildLine"
                    and not getattr(agent, "replan_disabled", False)):   # ← new guard
                agent.recalc_path_from_here()


    def flush_all_groundcrews_no_rerun(self) -> int:
        from groundcrew import GroundCrewAgent
        total = 0
        for ag in self.schedule.agents:
            if isinstance(ag, GroundCrewAgent):
                total += ag.flush_pending_to_fuel_no_rerun()
        return total

    # def compute_sector_splits(self, time, num_sectors=4):
    #     """
    #     Fast alternative to compute sector splits assuming a circular approximation
    #     of the active fire region. This function:
    #       - Computes the fire state grid at the given time.
    #       - Uses a threshold to identify burning cells.
    #       - Computes the centroid of these cells as the fire's "center".
    #       - Determines the maximum Euclidean distance from the center among burning cells.
    #       - Uses the circle defined by that center and radius to establish sector boundaries
    #         as rays emanating from the center.
    #
    #     Parameters:
    #       time (float): Simulation time at which to extract the sectors.
    #       num_sectors (int): The number of sectors to split the fire area.
    #
    #     Returns:
    #       sector_angle_ranges (list of tuple): Angle ranges (degrees) for each sector.
    #       sector_boundaries (list of Point): Points where each sector's boundary hits the circle.
    #       center (Point): The computed center of the fire.
    #       max_distance (float): The computed maximum distance (radius) of the fire region.
    #     """
    #     Point = namedtuple("Point", ["x", "y"])
    #
    #     # Get the current fire grid; assumed to be a 2D NumPy array.
    #     fire_data = self.fire.current_fire(time, max_time=self.fire_spread_sim_time)
    #
    #     # Create coordinate arrays using the fire model's bounds and pixel/transform info.
    #     # (These should match the dimensions of fire_data.)
    #     x = np.arange(self.fire.bounds.left, self.fire.bounds.right, self.fire.transform[0])
    #     y = np.arange(self.fire.bounds.top, self.fire.bounds.bottom, self.fire.transform[4])
    #
    #     # Determine the burning area using a threshold.
    #     # Here we assume cells with fire arrival time <= current time are considered burning.
    #     mask = fire_data <= time
    #
    #     if not np.any(mask):
    #         # If there are no burning cells, default to the center of the domain.
    #         center_x = (self.fire.bounds.left + self.fire.bounds.right) / 2
    #         center_y = (self.fire.bounds.top + self.fire.bounds.bottom) / 2
    #         max_distance = 100.0  # Default radius if no fire cells detected.
    #     else:
    #         # Compute indices of burning cells.
    #         burning_indices = np.nonzero(mask)
    #         # Note: x corresponds to columns and y to rows.
    #         burning_x = x[burning_indices[1]]
    #         burning_y = y[burning_indices[0]]
    #
    #         # Compute the centroid (average coordinates) of the burning cells.
    #         center_x = burning_x.mean()
    #         center_y = burning_y.mean()
    #
    #         # Compute the distance from the center to each burning cell (vectorized).
    #         distances = np.sqrt((burning_x - center_x) ** 2 + (burning_y - center_y) ** 2)
    #         max_distance = distances.max()
    #
    #     center = Point(center_x, center_y)
    #
    #     # Pre-compute the width (in degrees) of each sector.
    #     sector_width = 360.0 / num_sectors
    #     sector_angle_ranges = []
    #     sector_boundaries = []
    #
    #     # For each sector, compute the corresponding ray from the center.
    #     for i in range(num_sectors):
    #         lower_bound = i * sector_width
    #         upper_bound = lower_bound + sector_width
    #         sector_angle_ranges.append((lower_bound, upper_bound))
    #
    #         # Convert the lower bound to radians.
    #         angle_rad = np.deg2rad(lower_bound)
    #         # Determine the corresponding boundary point on the circle.
    #         boundary_x = center_x + max_distance * np.cos(angle_rad)
    #         boundary_y = center_y + max_distance * np.sin(angle_rad)
    #         boundary_point = Point(boundary_x, boundary_y)
    #         sector_boundaries.append(boundary_point)
    #
    #     return sector_angle_ranges, sector_boundaries, center, max_distance


    # --------------------------------------------------------------
    # Fast GPU implementation – copies back only 3 scalars
    # --------------------------------------------------------------

    def compute_sector_splits(self, time, num_sectors: int = 4):
        """
        Identical API, but runs entirely on the GPU and transfers
        only (centre-x, centre-y, radius) to the host.
        """
        Point = namedtuple("Point", ["x", "y"])

        # 1. GPU arrival-time grid (NO host copy)
        T_cp = self.fire.current_fire_cp()        # CuPy array
        mask = T_cp <= time                       # CuPy bool mask

        if not mask.any():
            # nothing burning yet – use domain centre
            cx = (self.fire.bounds.left + self.fire.bounds.right) / 2
            cy = (self.fire.bounds.bottom + self.fire.bounds.top)  / 2
            radius = 100.0
        else:
            rows, cols = cp.nonzero(mask)
            cx_pix = cols.mean()                  # CuPy scalar
            cy_pix = rows.mean()

            # pixel → metres       (a = pixel-width, e = –pixel-height)
            cx = float(self.fire.bounds.left +
                       cx_pix * self.fire.transform[0])
            cy = float(self.fire.bounds.top  +
                       cy_pix * self.fire.transform[4])

            # max distance (pixels) → metres
            radius = float(
                cp.sqrt((cols - cx_pix)**2 + (rows - cy_pix)**2).max()
                * abs(self.fire.transform[0])
            )

        centre = Point(cx, cy)

        # 2. Build sector boundaries
        sector_width = 360.0 / num_sectors
        angle_ranges, boundaries = [], []
        for i in range(num_sectors):
            lo = i * sector_width
            hi = lo + sector_width
            angle_ranges.append((lo, hi))

            θ = np.deg2rad(lo)
            bx = cx + radius * np.cos(θ)
            by = cy + radius * np.sin(θ)
            boundaries.append(Point(bx, by))

        return angle_ranges, boundaries, centre, radius


    # def step(self):
    #     if self.time >= self.overall_time_limit or self.containment:
    #         print("Simulation complete. Fire contained or time limit reached")
    #         self.fire.calculate_fire_score(self.time)
    #         return False
    #     if self.recalculate_flag:
    #         for agent in self.schedule.agents:
    #             agent.cruise_speed = agent.convert_knots_to_model_units(agent.cruise_speed_knots, self.time_step)
    #         self.recalculate_flag = False
    #         print(f"Cruise speeds recalculated for all agents with time_step: {self.time_step}")
    #     self.schedule.step()
    #     self.time += self.time_step
    #     self.current_time = self.start_time + datetime.timedelta(minutes=self.time)
    #     if self.debug:
    #         print(f"Simulation Time: {self.current_time.strftime('%H:%M')} (elapsed {self.time} minutes)")
    #     # self.plot_fire()
    #
    #     # Instead of always calling plot_fire (which is expensive), check the flag.
    #     if self.enable_plotting:
    #         self.plot_fire()
    #     else:
    #         # Skip plotting but update the sector splits which are still needed.
    #         self.update_sector_splits()
    def groundcrew_etas(self) -> dict[int, float]:
        """Return {groundcrew_id: minutes‑to‑Standby} for every crew."""
        return {
            agent.unique_id: agent._eta_remaining
            for agent in self.schedule.agents
            if isinstance(agent, GroundCrewAgent)
        }

    # helper used by MCTS / simulate_branch
    def is_sector_contained(self, idx: int) -> bool:
        """True if a crew has already finished a fire-line in this sector."""
        return idx in self.contained_sectors

    def register_retarded_sector(self, idx: int) -> None:
        """
        Mark <idx> as fully retarded by airtankers.
        If there are *no* ground crews and every sector is now covered,
        flip the global `containment` flag so the next `step()` stops.
        """
        self.retarded_sectors.add(idx)
        print("Sector retarded")
        print(self.retarded_sectors)

        # Only the aircraft-only scenario should trigger an early stop
        if self.groundcrew_count == 0:
            if len(self.retarded_sectors) == len(self.sector_boundaries):
                print("✓ All sectors retarded by aircraft – simulation complete")
                self.containment = True

    def step(self):

        # 1) Termination check
        # ── aircraft-only termination guard ───────────────────────────
        if self.groundcrew_count == 0 and len(self.retarded_sectors) == len(self.sector_boundaries):
            print("Simulation complete (air-tanker containment achieved)")
            self.fire.calculate_fire_score(self.time)
            return False
        # ── Groundcrew and aircraft Termination check ───────────────────────────
        elif self.time >= self.overall_time_limit or self.containment:
            print("Simulation complete. Fire contained or time limit reached")
            self.fire.calculate_fire_score(self.time)
            return False

        # 2) Handle any cruise‐speed adjustments
        if self.recalculate_flag:
            for agent in self.schedule.agents:
                agent.cruise_speed = agent.convert_knots_to_model_units(
                    agent.cruise_speed_knots,
                    self.time_step
                )
            self.recalculate_flag = False
            print(f"Cruise speeds recalculated for time_step={self.time_step}")

        # 3) Advance all agents one tick
        self.schedule.step()
        # if self.debug:
        # for uid, eta in self.groundcrew_etas().items():
            # print(f"[ETA] GC#{uid:02d}  {eta:5.1f} min → Standby")
        # --------------------------------------------------------------
        #  Handle freshly‑finished drops → ground‑crew synchronisation
        # --------------------------------------------------------------
        # for at in self.schedule.agents:
        #     if isinstance(at, AirtankerAgent):
        #         # fire this exactly once per drop
        #         if at.drop_complete and not getattr(at, "_drop_processed", False):
        #             self.notify_drop(at)
        #             at._drop_processed = True
        #         if not at.drop_complete:
        #             at._drop_processed = False

        if self.second_groundcrew_sector_mapping is not None:
            for agent in self.schedule.agents:
                if (isinstance(agent, GroundCrewAgent)
                        and agent.state == "Standby"):
                    new_sector = self.second_groundcrew_sector_mapping[
                        agent.unique_id % len(self.second_groundcrew_sector_mapping)
                        ]
                    agent.assign_to_sector(new_sector)  # ← one-liner, done
                    print( f">>> GC#{agent.unique_id:02d} reassigned from standby to sector {new_sector} (using second mapping)")

        # If second_groundcrew_sector_mapping is provided, reassign Standby agents.
        # if self.second_groundcrew_sector_mapping is not None:
        #     for agent in self.schedule.agents:
        #         from groundcrew import GroundCrewAgent
        #         if isinstance(agent, GroundCrewAgent) and agent.state == "Standby":
        #             new_sector = self.second_groundcrew_sector_mapping[
        #                 agent.unique_id % len(self.second_groundcrew_sector_mapping)]
        #
        #             # Collect previously completed finish points from other agents
        #             existing_points = {
        #                 other.sector_index: other.planned_finish
        #                 for other in self.schedule.agents
        #                 if isinstance(other, GroundCrewAgent)
        #                    and other.state == "Standby"
        #                    and other.sector_index != new_sector
        #             }
        #
        #             # # Call planner with existing points for coordination
        #             # planned = self.groundcrew_planner.get_planned_cells(
        #             #     [new_sector],
        #             #     existing_fireline_points=existing_points
        #             # )[new_sector]
        #             #
        #             # # Update agent
        #             # agent.sector_index = new_sector
        #             # start_cell = planned["start_cell"]
        #             # finish_cell = planned["finish_cell"]
        #             # agent.update_planned_targets(
        #             #     planned_start=start_cell,
        #             #     planned_finish=finish_cell
        #             # )
        #
        #             agent.assign_to_sector(new_sector)  # 1-liner helper
        #             # Record start/finish to boundary points
        #             start_boundary = (agent.sector_index - 1) % len(self.sector_boundaries)
        #             finish_boundary = agent.sector_index
        #
        #             if not hasattr(self, "used_boundary_points"):
        #                 self.used_boundary_points = {}
        #
        #             # for b_idx, cell in [(start_boundary, start_cell), (finish_boundary, finish_cell)]:
        #             #     if b_idx not in self.used_boundary_points:
        #             #         self.used_boundary_points[b_idx] = []
        #             #     self.used_boundary_points[b_idx].append(cell)
        #             agent.state = "ToStart"
        #             print(
        #                 f">>> GC#{agent.unique_id:02d} reassigned from standby to sector {new_sector} (using second mapping)")
        # 4) Reassign any ground‐crew in Standby → next sector
        # for agent in self.schedule.agents:
        #     if isinstance(agent, GroundCrewAgent) and agent.state == "Standby":
        #         next_idx = (agent.sector_index + 1) % len(self.sector_boundaries)
        #         planned = self.groundcrew_planner.get_planned_cells([next_idx])[next_idx]
        #         agent.sector_index = next_idx
        #         agent.update_planned_targets(
        #             planned_start=planned["start_cell"],
        #             planned_finish=planned["finish_cell"]
        #         )
        #         agent.state = "ToStart"
        #         print(f">>> GC#{agent.unique_id:02d} reassigned → sector {next_idx}")

        # 5) Advance model clock
        self.time += self.time_step
        self.current_time = self.start_time + datetime.timedelta(minutes=self.time)

        # 6) Plot/update sectors as before
        if self.enable_plotting:
            self.plot_fire()
        else:
            self.update_sector_splits()

        return True

