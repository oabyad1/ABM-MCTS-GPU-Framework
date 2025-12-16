from enum import Enum
import matplotlib.pyplot as plt
import mesa
import math
import random
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

# Import the new drops module
from drops_surrogate import Drop, set_drop, get_point_angle
# Import the new plotting module for color maps.
import surrogate_plotting as plotting

class InputCondition(Enum):
    AIRCRAFT_AVAILABLE = "Aircraft Available"
    ARRIVED_AT_RUNWAY = "Arrived at runway"
    ARRIVED_AT_ALTITUDE = "Arrived at altitude"
    ARRIVED = "arrived"
    TARGET_RECEIVED = "target_received"
    SCRAP_MISSION = "scrap_mission"
    DROP_COMPLETE = "drop_complete"
    LANDED = "Landed"
    TAXI_COMPLETE_REFUEL_NEEDED = "Taxi complete - Refuel needed"
    TAXI_COMPLETE_REFUEL_NOT_NEEDED = "Taxi complete - No refuel needed"
    REFUEL_COMPLETE = "Refuel complete"
    NON_OPERATIONAL_TIME = "Non-operational time"
    OPERATIONAL_TIME = "Operational time"
    REASSIGN_AIRBORNE = "Reassign airborne"

# Minimum spacing rules
_MAX_RECENT_DROPS = 3          # keep at most this many of *my* drops
_MIN_SEPARATION_M = 250       # centre-to-centre clearance




class AirtankerAgent(mesa.Agent):
    def __init__(self, unique_id, model, base_position, time_step, config):
        super().__init__(unique_id, model)
        self.base_position = base_position
        self.position = base_position
        self.state = "Ready for deployment"
        self._orig_config = config.copy()

        # Config Attributes
        self.drop_capacity = config["drop_capacity"]
        self.current_retardant = config["drop_capacity"]
        self.cruise_speed_knots = config["cruise_speed_knots"]
        self.fuel_capacity = config["fuel_capacity"]
        self.taxi_burn_rate = config["burn_rates"]["Taxi"]
        self.takeoff_burn_rate = config["burn_rates"]["Takeoff&Climb"]
        self.cruise_burn_rate = config["burn_rates"]["Cruise"]
        self.drop_burn_rate = config["burn_rates"]["Cruise"]
        self.landing_burn_rate = config["burn_rates"]["Landing"]
        self.is_24_7 = config["is_24_7"]
        self.fuel_threshold = config["fuel_threshold"]


        self.deterministic_fire_target = config.get("deterministic_fire_target", False)
        self.fixed_fire_target_angle = config.get("fixed_fire_target_angle")  # None ⇒ use spacing rule

        self.STATE_TIME_taxi_to_runway_time = config["state_times"]["Taxi to Runway"]
        self.STATE_TIME_takeoff_and_climb_time = config["state_times"]["Takeoff&Climb"]
        self.STATE_TIME_dropping_time = config["state_times"]["Dropping"]
        self.STATE_TIME_landing_time = config["state_times"]["Landing"]
        self.STATE_TIME_taxi_to_station_time = config["state_times"]["Taxi to Station"]
        self.STATE_TIME_refill_time = config["state_times"]["Refill"]
        self.STATE_TIME_refuel_and_refill_time = config["state_times"]["Refuel and Refill"]

        self.buffer_time = config["buffer_time"]
        self.buffer_distance = config["buffer_distance"]
        # End Config Attributes

        self.cruise_speed = self.convert_knots_to_model_units(self.cruise_speed_knots, time_step)
        self.angle = 0
        self.previous_positions = []
        self.loiter_time = 0

        self.drop_target = None
        self.drop_angle = None
        self.drop_target_assigned = False
        self.drop_complete = False
        self.debug = self.model.debug

        # Update fire center: use surrogate's ignition_pt property.
        self.assign_fixed_fire_area_target()
        self.current_fuel = self.fuel_capacity
        self.time_step = self.model.time_step
        self.recent_drops: list[tuple[float, float]] = []
        # State time trackers
        self.taxi_to_runway_time = 0
        self.takeoff_and_climb_time = 0
        self.en_route_to_fire_time = 0
        self.moving_to_drop_target_time = 0
        self.dropping_time = 0
        self.returning_to_base_time = 0
        self.landing_time = 0
        self.taxi_to_station_time = 0
        self._next_drop_angle = 0
        self.state_transition_table = {
            ("Ready for deployment", InputCondition.AIRCRAFT_AVAILABLE): "Taxi to Runway",
            ("Ready for deployment", InputCondition.NON_OPERATIONAL_TIME): "Waiting at base",
            ("Ready for deployment", InputCondition.REASSIGN_AIRBORNE): "En route to fire",
            ("Waiting at base", InputCondition.OPERATIONAL_TIME): "Ready for deployment",
            ("Taxi to Runway", InputCondition.ARRIVED_AT_RUNWAY): "Takeoff&Climb",
            ("Takeoff&Climb", InputCondition.ARRIVED_AT_ALTITUDE): "En route to fire",
            ("En route to fire", InputCondition.ARRIVED): "Moving to drop target",
            ("Moving to drop target", InputCondition.ARRIVED): "Dropping",
            ("Dropping", InputCondition.DROP_COMPLETE): "Returning to base",
            ("Returning to base", InputCondition.ARRIVED): "Landing",
            ("Landing", InputCondition.LANDED): "Taxi to Station",
            ("Taxi to Station", InputCondition.TAXI_COMPLETE_REFUEL_NEEDED): "Refilling",
            ("Taxi to Station", InputCondition.TAXI_COMPLETE_REFUEL_NOT_NEEDED): "Refilling",
            ("Refilling", InputCondition.REFUEL_COMPLETE): "Ready for deployment"
        }
        self.assigned_sector = self.assign_sector_once()
        self.taxi_complete = False
        self.takeoff_complete = False
        self.landing_complete = False
        self.taxi_to_station_complete = False
        self.refuel_complete = False
        self.needs_refuel = False
        self.counter = 0

        self._skip_ground_ops = False  # will be set by deep_clone()

    def step(self):
        self.burn_fuel()
        condition = self.evaluate_condition()
        if self.debug:
            print(self.state)
        next_state = self.state_transition_table.get((self.state, condition), self.state)
        if next_state != self.state:
            if self.debug:
                print(f"\033[94mTime spent in {self.state} is {self.counter} minutes\033[0m")
            self.counter = 0
        self.state = next_state
        self.execute_state_action()
        self.counter += self.time_step


    # ─────────────────────────────────────────────────────────────────
    #  Deep-clone helper  (CPU-only fields – no GPU/CuPy touched)
    # ─────────────────────────────────────────────────────────────────
    # def deep_clone(self, new_model: "WildfireModel") -> "AirtankerAgent":
    #     """
    #     Re-create this airtanker on <new_model> with all runtime state.
    #     """
    #     # 1) build a fresh instance of the SAME subclass (FireHercAgent, …)
    #     cls = self.__class__  # keeps the exact type
    #     clone = cls(
    #         unique_id=self.unique_id,
    #         model=new_model,
    #         base_position=tuple(self.base_position),
    #         time_step=new_model.time_step,
    #     )
    #
    #
    #
    #     # copy *all* private runtime fields exactly as before …
    #     clone.position = tuple(self.position)
    #     clone.angle = self.angle
    #     clone.previous_positions = list(self.previous_positions)
    #     clone.current_fuel = self.current_fuel
    #     clone.current_retardant = self.current_retardant
    #     clone.recent_drops = list(self.recent_drops)
    #     clone._next_drop_angle = self._next_drop_angle
    #     # … etc. (leave the rest untouched)
    #
    #
    #
    #     clone.state = "Ready for deployment"
    #     clone.drop_target = None
    #     clone.drop_angle = None
    #     clone.drop_target_assigned = False
    #     clone.drop_complete = False
    #     clone.taxi_complete = False
    #     clone.takeoff_complete = False
    #     clone.landing_complete = False
    #     clone.taxi_to_station_complete = False
    #     clone.refuel_complete = False
    #     clone.counter = 0
    #
    #
    #     # ──────────────────────────────────────────────────────────────
    #     # place into the new space grid
    #     new_model.space.move_agent(clone, clone.position)
    #     return clone

    def deep_clone(self, new_model: "WildfireModel") -> "AirtankerAgent":
        """
                Re-create this airtanker on <new_model> with all runtime state.
                """
        # 1) build a fresh instance of the SAME subclass (FireHercAgent, …)
        cls = self.__class__  # keeps the exact type
        clone = cls(
            unique_id=self.unique_id,
            model=new_model,
            base_position=tuple(self.base_position),
            time_step=new_model.time_step,
        )


        clone.position = tuple(self.position)
        clone.angle = self.angle
        clone.previous_positions = list(self.previous_positions)
        clone.current_fuel = self.current_fuel
        clone.current_retardant = self.current_retardant
        # clone.recent_drops = list(self.recent_drops)
        # clone._next_drop_angle = self._next_drop_angle
        clone.recent_drops = list(self.recent_drops)[- _MAX_RECENT_DROPS:]
        # clone._next_drop_angle = self._next_drop_angle
        # ── preserve *air* status or reset, depending on parent -----------
        AIRBORNE = {"Takeoff&Climb", "En route to fire",
                    "Moving to drop target", "Dropping",
                    "Returning to base"}
        if self.state in AIRBORNE or self.takeoff_complete:
            # keep all flight flags so the FSM sees us as airborne
            clone.state = "Ready for deployment"  # placeholder
            clone.taxi_complete = True  # skip taxi
            clone.takeoff_complete = True  # skip climb
            clone._skip_ground_ops = True  # triggers fast path
        else:
            clone.state = "Ready for deployment"
            clone.taxi_complete = False
            clone.takeoff_complete = False
            clone._skip_ground_ops = False

        # reset *only* mission-specific bits

        clone.drop_target = None
        clone.drop_angle = None
        clone.drop_target_assigned = False
        clone.drop_complete = False
        clone.counter = 0
        clone.landing_complete = False
        clone.taxi_to_station_complete = False
        clone.refuel_complete = False
        clone.counter = 0

        # ── place into grid ───────────────────────────────────────────────
        new_model.space.move_agent(clone, clone.position)
        return clone

    def get_fire_center_with_buffer(self, buffer=100):
        # Use the surrogate's ignition_pt property.
        fire_center_x, fire_center_y = self.model.fire.ignition_pt
        return buffer, (fire_center_x, fire_center_y)

    def is_operational_time(self):
        current_hour = self.model.current_time.hour
        return 7 <= current_hour < 19

    # helper – now an *instance* method
    def _too_close_to_past(self, cand_mid: tuple[float, float]) -> bool:
        import numpy as np
        c = np.array(cand_mid)
        for mid in self.recent_drops:
            if np.linalg.norm(c - np.array(mid)) < _MIN_SEPARATION_M:
                return True
        return False

    def _is_cell_burned(self, xy: tuple[float, float]) -> bool:
        """
        Return True if the fire arrival-time grid shows this (x,y) cell
        has ignited on or before the current model time.
        """
        fire = self.model.fire
        arrivals = fire.arrival_time_grid

        dx = fire.transform[0]  # cell size X
        dy = -fire.transform[4]  # cell size Y (transform[4] is negative)

        col = int((xy[0] - fire.bounds.left) / dx)
        row = int((fire.bounds.top - xy[1]) / dy)

        if not (0 <= row < arrivals.shape[0] and 0 <= col < arrivals.shape[1]):
            return True  # outside grid – treat as invalid/burned

        val = arrivals[row, col]
        return np.isfinite(val) and val <= self.model.time

    def has_enough_time_for_mission(self):
        total_mission_time = self.estimate_mission_time()
        current_time = self.model.current_time
        end_of_day = current_time.replace(hour=19, minute=0, second=0, microsecond=0)
        time_remaining = (end_of_day - current_time).total_seconds() / 60
        buffer_time = 15
        return (time_remaining - buffer_time) >= total_mission_time

    def estimate_mission_time(self):
        cruise_speed_meters_per_minute = self.cruise_speed_knots * 0.51444 * 60
        base_x, base_y = self.base_position
        fire_x, fire_y = self.fire_area_target
        distance_to_fire = math.sqrt((fire_x - base_x) ** 2 + (fire_y - base_y) ** 2)
        en_route_to_fire_time = distance_to_fire / cruise_speed_meters_per_minute
        return_to_base_time = en_route_to_fire_time
        taxi_to_runway_time = self.STATE_TIME_taxi_to_runway_time
        takeoff_and_climb_time = self.STATE_TIME_takeoff_and_climb_time
        dropping_time = self.STATE_TIME_dropping_time
        landing_time = self.STATE_TIME_landing_time
        taxi_to_station_time = self.STATE_TIME_taxi_to_station_time
        total_mission_time = (taxi_to_runway_time + takeoff_and_climb_time +
                              en_route_to_fire_time + dropping_time +
                              return_to_base_time + landing_time + taxi_to_station_time)
        return total_mission_time

    # def assign_fixed_fire_area_target(self):
    #     radius, (fire_center_x, fire_center_y) = self.get_fire_center_with_buffer(buffer=2000)
    #     angle = random.uniform(0, 360)
    #     target_x = fire_center_x + radius * math.cos(math.radians(angle))
    #     target_y = fire_center_y + radius * math.sin(math.radians(angle))
    #     self.fire_area_target = (target_x, target_y)
    #     if self.debug:
    #         print(f"\033[96mAgent {self.unique_id} assigned fixed fire area target: {self.fire_area_target}\033[0m")
    # ------------------------------------------------------------------
    #  Deterministic “loiter” point around the fire, used before dropping
    # ------------------------------------------------------------------
    def assign_fixed_fire_area_target(self):
        """
        Pick a point on a ring (radius = 2 km) around the fire centre.
        The choice is fully repeatable:

        1. If self.fixed_fire_target_angle is set → use it verbatim.
        2. elif self.deterministic_fire_target is True
           → spread aircraft evenly: angle = k * 360 / N, where
              k = self.unique_id  (or index inside schedule),
              N = total number of airtankers.
        3. else → fall back to the original random.uniform(0, 360) behaviour.
        """
        radius, (cx, cy) = self.get_fire_center_with_buffer(buffer=2000)

        # ------- rule #1: an explicit fixed angle beats everything -------
        if self.fixed_fire_target_angle is not None:
            angle_deg = float(self.fixed_fire_target_angle)

        # ------- rule #2: even spacing by unique-id ----------------------
        elif self.deterministic_fire_target:
            # How many airtankers are in the simulation?
            try:
                total_aircraft = self.model.num_aircraft
            except AttributeError:
                total_aircraft = len(self.model.schedule.agents)  # <- Mesa fallback

            angle_deg = (360.0 / total_aircraft) * (self.unique_id % total_aircraft)

        # ------- rule #3: original stochastic pick -----------------------
        else:
            angle_deg = random.uniform(0, 360)

        # Convert polar → Cartesian
        theta = math.radians(angle_deg)
        target_x = cx + radius * math.cos(theta)
        target_y = cy + radius * math.sin(theta)

        self.fire_area_target = (target_x, target_y)

        if self.debug:
            print(
                f"\033[96mAgent {self.unique_id} fixed fire-area target: "
                f"{self.fire_area_target}  (θ = {angle_deg:.1f}°)\033[0m"
            )

    def burn_fuel(self):
        fuel_burn = 0
        if self.state in ["Taxi to Runway", "Taxi to Station"]:
            fuel_burn = self.taxi_burn_rate * self.time_step
        elif self.state == "Takeoff&Climb":
            fuel_burn = self.takeoff_burn_rate * self.time_step
        elif self.state in ["En route to fire", "Returning to base"]:
            fuel_burn = self.cruise_burn_rate * self.time_step
        elif self.state == "Moving to drop target":
            fuel_burn = self.cruise_burn_rate * self.time_step * 0.8
        elif self.state == "Dropping":
            fuel_burn = self.drop_burn_rate * self.time_step
        elif self.state == "Landing":
            fuel_burn = self.landing_burn_rate * self.time_step
        self.current_fuel = max(0, self.current_fuel - fuel_burn)
        if self.debug:
            print(f"\033[95mAgent {self.unique_id} fuel level: {self.current_fuel}/{self.fuel_capacity}\033[0m")

    def evaluate_condition(self):
        if self.state == "Ready for deployment":
            # --- FAST REDEPLOYMENT -------------------------------------------
            if self._skip_ground_ops:  # ← set by deep_clone()
                self._skip_ground_ops = False  # consume once
                return InputCondition.REASSIGN_AIRBORNE
            if self.model.current_time < self.model.operational_start_time:
                return InputCondition.NON_OPERATIONAL_TIME
            if not self.is_24_7:
                if not self.is_operational_time() or not self.has_enough_time_for_mission():
                    return InputCondition.NON_OPERATIONAL_TIME
            return InputCondition.AIRCRAFT_AVAILABLE
        elif self.state == "Waiting at base":
            if self.is_24_7:
                if self.model.current_time >= self.model.operational_start_time:
                    return InputCondition.OPERATIONAL_TIME
                return InputCondition.NON_OPERATIONAL_TIME
            if self.is_operational_time() and self.has_enough_time_for_mission():
                return InputCondition.OPERATIONAL_TIME
            return InputCondition.NON_OPERATIONAL_TIME
        elif self.state == "Taxi to Runway":
            if self.taxi_complete:
                return InputCondition.ARRIVED_AT_RUNWAY
        elif self.state == "Takeoff&Climb":
            if self.takeoff_complete:
                return InputCondition.ARRIVED_AT_ALTITUDE
        elif self.state == "En route to fire":
            if self.has_arrived(self.fire_area_target):
                return InputCondition.ARRIVED
            else:
                return None
        elif self.state == "Moving to drop target":
            if self.has_arrived(self.drop_target):
                return InputCondition.ARRIVED
            else:
                return None
        elif self.state == "Dropping":
            if self.drop_complete:
                return InputCondition.DROP_COMPLETE
            else:
                return None
        elif self.state == "Returning to base":
            if self.has_arrived(self.base_position):
                return InputCondition.ARRIVED
            else:
                return None
        elif self.state == "Landing":
            if self.landing_complete:
                return InputCondition.LANDED
        elif self.state == "Taxi to Station":
            if self.taxi_to_station_complete:
                if self.current_fuel < self.fuel_threshold * self.fuel_capacity:
                    self.needs_refuel = True
                    return InputCondition.TAXI_COMPLETE_REFUEL_NEEDED
                else:
                    self.needs_refuel = False
                    return InputCondition.TAXI_COMPLETE_REFUEL_NOT_NEEDED
        elif self.state == "Refilling":
            if self.refuel_complete:
                return InputCondition.REFUEL_COMPLETE
        return None

    def has_arrived(self, fire_area_target):
        current_x, current_y = self.position
        target_x, target_y = fire_area_target
        distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        return distance <= self.cruise_speed

    def execute_state_action(self):
        if self.state == "Taxi to Runway":
            return self.taxi_to_runway()
        elif self.state == "Waiting at base":
            pass
        elif self.state == "Takeoff&Climb":
            return self.takeoff_and_climb()
        elif self.state == "En route to fire":
            return self.move_to_target()
        #fix
        elif self.state == "Moving to drop target":
            if not self.drop_target_assigned:
                self.assign_drop_target()  # ← do NOT set the flag here
            if self.drop_target is None:  # still no point? abort mission
                self._skip_current_drop("no drop target generated")
                return
            return self.move_to_drop_target()
        # elif self.state == "Moving to drop target":
        #     if not self.drop_target_assigned:
        #         self.assign_drop_target()
        #         self.drop_target_assigned = True
        #     return self.move_to_drop_target()
        elif self.state == "Dropping":
            return self.drop()
        elif self.state == "Returning to base":
            return self.return_to_base()
        elif self.state == "Landing":
            return self.landing()
        elif self.state == "Taxi to Station":
            return self.taxi_to_station()
        elif self.state == "Refilling":
            return self.refill()
        return 0

    def refill(self):
        if not hasattr(self, '_refill_time'):
            self._refill_time = 0
        time_to_act = self.time_step
        self._refill_time += time_to_act
        required_time = self.STATE_TIME_refuel_and_refill_time if self.needs_refuel else self.STATE_TIME_refill_time
        if self._refill_time >= required_time:
            self.current_retardant = self.drop_capacity
            if self.needs_refuel:
                self.current_fuel = self.fuel_capacity
            self.refuel_complete = True
            self._refill_time = 0
        else:
            self.refuel_complete = False

    def taxi_to_runway(self):
        if not hasattr(self, '_taxi_time'):
            self._taxi_time = 0
        time_to_act = self.time_step
        self._taxi_time += time_to_act
        self.taxi_to_runway_time += time_to_act
        if self._taxi_time >= self.STATE_TIME_taxi_to_runway_time:
            self.taxi_complete = True
            self._taxi_time = 0
            self.taxi_to_runway_time = 0
        else:
            self.taxi_complete = False

    def takeoff_and_climb(self):
        if not hasattr(self, '_takeoff_time'):
            self._takeoff_time = 0
        time_to_act = self.time_step
        self._takeoff_time += time_to_act
        self.takeoff_and_climb_time += time_to_act
        if self.debug:
            print(f"\033[95mTakeoff & Climb - Time spent: {self.takeoff_and_climb_time} minutes\033[0m")
        if self._takeoff_time >= self.STATE_TIME_takeoff_and_climb_time:
            self.takeoff_complete = True
            self._takeoff_time = 0
            self.takeoff_and_climb_time = 0
        else:
            self.takeoff_complete = False

    def landing(self):
        if not hasattr(self, '_landing_time'):
            self._landing_time = 0
        time_to_act = self.time_step
        self._landing_time += time_to_act
        self.landing_time += time_to_act
        if self.debug:
            print(f"\033[95mLanding - Time spent: {self.landing_time} minutes\033[0m")
        if self._landing_time >= self.STATE_TIME_landing_time:
            self.landing_complete = True
            self._landing_time = 0
            self.landing_time = 0
        else:
            self.landing_complete = False

    def taxi_to_station(self):
        if not hasattr(self, '_taxi_to_station_time'):
            self._taxi_to_station_time = 0
        time_to_act = self.time_step
        self._taxi_to_station_time += time_to_act
        self.taxi_to_station_time += time_to_act
        if self.debug:
            print(f"\033[95mTaxi to Station - Time spent: {self.taxi_to_station_time} minutes\033[0m")
        if self._taxi_to_station_time >= self.STATE_TIME_taxi_to_station_time:
            self.taxi_to_station_complete = True
            self._taxi_to_station_time = 0
            self.taxi_to_station_time = 0
        else:
            self.taxi_to_station_complete = False

    def convert_knots_to_model_units(self, knots, time_step):
        time_step_in_seconds = time_step * 60
        meters_per_second = knots * 0.51444
        meters_per_time_step = meters_per_second * time_step_in_seconds
        return meters_per_time_step

    def move_to_target(self):
        target_x, target_y = self.fire_area_target
        current_x, current_y = self.position
        distance_to_target = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        travel_distance = self.cruise_speed
        if travel_distance >= distance_to_target:
            self.position = self.fire_area_target
            time_taken = (distance_to_target / self.cruise_speed) * self.time_step
            self.en_route_to_fire_time = 0
        else:
            angle = math.atan2(target_y - current_y, target_x - current_x)
            new_x = current_x + travel_distance * math.cos(angle)
            new_y = current_y + travel_distance * math.sin(angle)
            self.position = (new_x, new_y)
            time_taken = self.time_step
        self.previous_positions.append(self.position)
        self.model.space.move_agent(self, self.position)
        self.en_route_to_fire_time += time_taken
        if self.debug:
            print(f"\033[95mEn Route to Fire - Time spent: {self.en_route_to_fire_time} minutes\033[0m")

    def drop_opt_time(self, fire, angle, time, length, width=150, buffer_dist=0):
        d = set_drop(fire, angle, time, length, width, buffer_dist)
        d.run_one_drop(self.model.drop_number)
        return fire.calculate_fire_score(self.model.time)

    def assign_sector_once(self):
        sectors = self.model.sector_angle_ranges
        num_sectors = len(sectors)
        sector_index = self.unique_id % num_sectors
        sector_name = f"Sector {sector_index + 1}"
        if self.model.debug:
            print(f"Agent {self.unique_id} permanently assigned to {sector_name} (sector index: {sector_index})")
        return (sector_name, sector_index)

    # def assign_drop_target(self):
    #     sector_name, sector_index = self.assigned_sector
    #     current_sector_range = self.model.sector_angle_ranges[sector_index]
    #     lower_bound, upper_bound = current_sector_range
    #     unique_drop_angle = random.uniform(lower_bound, upper_bound)
    #     if self.debug:
    #         print(f"Agent {self.unique_id} using {sector_name} with updated angle range {current_sector_range} and drop angle: {unique_drop_angle}")
    #     time = self.model.time + self.buffer_time
    #     drop_point, drop_angle = get_point_angle(self.model.fire, unique_drop_angle, time, self.buffer_distance)
    #     self.drop_target = drop_point
    #     self.drop_angle = drop_angle
    #     d = Drop(self.model.fire, self.drop_target, self.drop_angle, self.drop_capacity)
    #     d.run_drop()
    #     if self.debug:
    #         print(f"Agent {self.unique_id} set drop target at {self.drop_target} with drop angle: {unique_drop_angle}")


    # def assign_drop_target(self):
    #     sector_name, sector_index = self.assigned_sector
    #     # we’re no longer sampling within the sector; we force 0° or 90°
    #     unique_drop_angle = self._next_drop_angle
    #
    #     # flip for next time
    #     self._next_drop_angle = 90 if self._next_drop_angle == 0 else 0
    #
    #     if self.debug:
    #         print(f"Agent {self.unique_id} using {sector_name} "
    #               f"– forced drop angle: {unique_drop_angle}°")
    #
    #     # compute where & how to drop
    #     time = self.model.time + self.buffer_time
    #     drop_point, drop_angle = get_point_angle(
    #         self.model.fire, unique_drop_angle, time, self.buffer_distance
    #     )
    #     self.drop_target = drop_point
    #     self.drop_angle = drop_angle
    #
    #     # actually do the drop
    #     d = Drop(self.model.fire, self.drop_target, self.drop_angle, self.drop_capacity)
    #     d.run_drop()
    #
    #     if self.debug:
    #         print(f"Agent {self.unique_id} set drop at {self.drop_target} "
    #               f"with drop angle: {unique_drop_angle}°")
    #GOOD
    # def assign_drop_target(self):
    #     # Get the assigned sector (e.g., ("Sector 1", index)) and its angular bounds.
    #     sector_name, sector_index = self.assigned_sector
    #     # print(self.model.sector_angle_ranges[sector_index])
    #     current_sector_range = self.model.sector_angle_ranges[sector_index]
    #     lower_bound, upper_bound = current_sector_range
    #
    #     # Create 5° increments within the agent’s assigned sector.
    #     increments = []
    #     angle = lower_bound
    #     while angle < upper_bound:
    #         # Ensure that the last increment doesn’t overshoot the upper_bound.
    #         increments.append((angle, min(angle + 5, upper_bound)))
    #         angle += 5
    #
    #     # Use a drop_time for the analysis.
    #     # Here we use the current model time plus the agent’s buffer time.
    #     drop_time = self.model.time + self.buffer_time
    #     # print(self.buffer_time)
    #     # Query the ROS values on the fire perimeter over these increments.
    #     # Note: This calls the aggregate_perimeter_ros_by_increments() function defined in
    #     # the SurrogateFireModelROS_CK2Multi class (attached to self.model.fire).
    #     try:
    #         aggregated_ros = self.model.fire.aggregate_perimeter_ros_by_increments(
    #             angle_ranges=increments,
    #             statistic='median',  # or 'median'
    #             time_threshold=drop_time,  # using drop_time as the threshold
    #
    #             tolerance_past=5,
    #             tolerance_future=30,
    #             cell_size_ft=98.4  # the spatial resolution used in the fire model
    #         )
    #     except Exception as e:
    #         if self.debug:
    #             print("Error while aggregating ROS by increments:", e)
    #         # Fallback: choose a random angle in the sector
    #         aggregated_ros = {}
    #
    #     # Find the angular increment with the highest aggregated ROS.
    #     best_range = None
    #     best_ros = -float("inf")
    #     for angle_range, ros_value in aggregated_ros.items():
    #         # Skip ranges with no valid ROS (NaN) values.
    #         if np.isnan(ros_value):
    #             continue
    #         if ros_value > best_ros:
    #             best_ros = ros_value
    #             best_range = angle_range
    #
    #     # If no valid value was found, fall back to choosing a random angle.
    #     if best_range is None:
    #         unique_drop_angle = random.uniform(lower_bound, upper_bound)
    #     else:
    #         # Use the midpoint of the best (5°) increment as the drop angle.
    #         unique_drop_angle = (best_range[0] + best_range[1]) / 2
    #
    #     if self.debug:
    #         print(f"Agent {self.unique_id} (assigned {sector_name}) "
    #               f"searched range {current_sector_range} and selected drop angle: {unique_drop_angle} "
    #               f"(aggregated ROS {best_ros:.2f} ft/min in range {best_range})")
    #
    #     # Get the drop target point using the chosen angle.
    #     drop_point, drop_angle = get_point_angle(
    #         self.model.fire,
    #         unique_drop_angle,
    #         drop_time,
    #         self.buffer_distance
    #     )
    #     self.drop_target = drop_point
    #     self.drop_angle = drop_angle
    #
    #     # Execute the drop.
    #     d = Drop(self.model.fire, self.drop_target, self.drop_angle, self.drop_capacity)
    #     d.run_drop()
    #     self.drop_complete = False          # ← reset for next mission
    #
    #     if self.debug:
    #         print(f"Agent {self.unique_id} set drop target at {self.drop_target} using drop angle: {unique_drop_angle}")

    # ------------------------------------------------------------------
    # emergency fallback – behave as if a drop has just happened
    # ------------------------------------------------------------------
    def _skip_current_drop(self, reason: str = "") -> None:
        """
        Pretend a drop was carried out successfully and put the aircraft
        into the “Returning to base” phase straight away.
        """
        self.drop_target = self.position  # no travel needed
        self.drop_angle = 0  # dummy
        self.drop_target_assigned = True
        self.drop_complete = True  # makes the FSM jump
        self.current_retardant = 0  # spent the load
        self.state = "Returning to base"  # immediate jump
        if self.debug:
            print(f"[AT#{self.unique_id:02d}] 🔸 skipped drop – {reason}")

    def assign_drop_target(self):
        sector_name, idx = self.assigned_sector
        lo, hi = self.model.sector_angle_ranges[idx]

        increments = [(a, min(a + 5, hi)) for a in np.arange(lo, hi, 5)]
        t_drop = self.model.time + self.buffer_time

        stats = self.model.fire.aggregate_perimeter_ros_by_increments(
            increments, "median",
            time_threshold=t_drop,
            tolerance_past=5,
            tolerance_future=120,
            cell_size_ft=98.4,
        )

        ranked = sorted(
            ((rng, val) for rng, val in stats.items() if np.isfinite(val)),
            key=lambda kv: -kv[1]
        )

        chosen_mid = chosen_ang = None
        for rng, _ in ranked:
            trial_ang = (rng[0] + rng[1]) / 2
            trial_mid, trial_drop_ang = get_point_angle(
                self.model.fire, trial_ang, t_drop, self.buffer_distance
            )

            if trial_mid is None:
                self._skip_current_drop("no valid contour / drop-point")
                print(f"  → no contour at angle {trial_ang:.1f}°")

                self.model.register_retarded_sector(self.assigned_sector[1])
                return  # ← leave assign_drop_target() early
            # ------------------------------------------------------------------
            # 2) point already inside the burned area?
            if self._is_cell_burned(trial_mid):

                print(f"  → {trial_mid} already burned – skipping")

                self.model.register_retarded_sector(self.assigned_sector[1])
                return
            if not self._too_close_to_past(trial_mid):
                chosen_mid, chosen_ang = trial_mid, trial_drop_ang
                # print(f"  → point {trial_mid} too close to recent drops")
                print(f"  → selected {chosen_mid} as drop point")
                break

        # # fallback
        # if chosen_mid is None:
        #     print("RANDOM")
        #     rand_ang = random.uniform(lo, hi)
        #     chosen_mid, chosen_ang = get_point_angle(
        #         self.model.fire, rand_ang, t_drop, self.buffer_distance
        #     )
        #     if chosen_mid is None:  # <- still no luck
        #         self._skip_current_drop("RANDOM fallback angle also failed")
        #         return

        # ───────────────────────────────
        # final fallback – abort mission
        # ───────────────────────────────
        if chosen_mid is None:
            # No viable contour / angle found in this sector:
            #   → pretend the drop happened, dump the load,
            #     and send the aircraft back to base.

            self.model.register_retarded_sector(self.assigned_sector[1])

            self._skip_current_drop("no viable drop point in sector")
            return

        gc_flushed = self.model.flush_all_groundcrews_no_rerun()
        drop_succeeded = False


        # ────────── after you’ve chosen chosen_mid, chosen_ang ──────────
        try:
            d = Drop(self.model.fire, chosen_mid, chosen_ang, self.drop_capacity)
            d.run_drop()
            drop_succeeded = True
            print("DROP RAN")
            self.model.notify_drop(self)

        except ValueError as e:
            # rect was out of bounds
            if self.debug:
                print(f"[AT#{self.unique_id:02d}] ⚠️  drop rectangle error: {e}")

            # Drop failed BEFORE re_run; if we merged any GC cells, re_run once now:
            if gc_flushed > 0 and not drop_succeeded:
                self.model.fire.re_run(self.model.fire.fuel_model)
            self._skip_current_drop("invalid drop rectangle")
            return

        # # success → remember the drop
        # self.drop_target = chosen_mid
        # self.drop_angle = chosen_ang
        # self.drop_target_assigned = True
        # self.drop_complete = False

        # apply & remember

        # Drop(self.model.fire, chosen_mid, chosen_ang, self.drop_capacity).run_drop()


        self.drop_target, self.drop_angle = chosen_mid, chosen_ang
        self.drop_target_assigned = True
        self.drop_complete = False

        # record and clip history
        self.recent_drops.append(chosen_mid)
        if len(self.recent_drops) > _MAX_RECENT_DROPS:
            self.recent_drops = self.recent_drops[-_MAX_RECENT_DROPS:]

        if self.debug:
            print(f"[AT#{self.unique_id:02d}] picked {chosen_mid} in {sector_name} "
                  f"(history {len(self.recent_drops)}/{_MAX_RECENT_DROPS})")

    def get_unique_drop_angle(self):
        return random.uniform(0, 360)

    def move_to_drop_target(self):
        target_x, target_y = self.drop_target
        current_x, current_y = self.position
        distance_to_target = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        travel_distance = self.cruise_speed * 0.8
        if travel_distance >= distance_to_target:
            self.position = self.drop_target
            time_taken = (distance_to_target / (self.cruise_speed * 0.8)) * self.time_step
            self.moving_to_drop_target_time = 0
        else:
            angle = math.atan2(target_y - current_y, target_x - current_x)
            new_x = current_x + travel_distance * math.cos(angle)
            new_y = current_y + travel_distance * math.sin(angle)
            self.position = (new_x, new_y)
            time_taken = self.time_step
        self.previous_positions.append(self.position)
        self.model.space.move_agent(self, self.position)
        self.moving_to_drop_target_time += time_taken
        if self.debug:
            print(f"\033[95mMoving to Drop Target - Time spent: {self.moving_to_drop_target_time} minutes\033[0m")

    def drop(self):
        if not hasattr(self, '_drop_time'):
            self._drop_time = 0
        time_to_act = self.time_step
        self._drop_time += time_to_act
        self.dropping_time += time_to_act
        if self.debug:
            print(f"\033[95mDropping - Time spent: {self.dropping_time} minutes\033[0m")
        if self._drop_time >= self.STATE_TIME_dropping_time:
            if self.debug:
                print(f"Agent {self.unique_id} performed a drop at {self.drop_target}")
            self.drop_complete = True
            self.drop_target_assigned = False
            self._drop_time = 0
            self.dropping_time = 0
            self.current_retardant = 0
        else:
            self.drop_complete = False

    def return_to_base(self):
        base_x, base_y = self.base_position
        current_x, current_y = self.position
        distance_to_base = math.sqrt((base_x - current_x)**2 + (base_y - current_y)**2)
        travel_distance = self.cruise_speed
        if travel_distance >= distance_to_base:
            self.position = self.base_position
            time_taken = (distance_to_base / self.cruise_speed) * self.time_step
            self.returning_to_base_time = 0
        else:
            angle = math.atan2(base_y - current_y, base_x - current_x)
            new_x = current_x + travel_distance * math.cos(angle)
            new_y = current_y + travel_distance * math.sin(angle)
            self.position = (new_x, new_y)
            time_taken = self.time_step
        self.previous_positions.append(self.position)
        self.model.space.move_agent(self, self.position)
        self.returning_to_base_time += time_taken
        if self.debug:
            print(f"\033[95mReturning to Base - Time spent: {self.returning_to_base_time} minutes\033[0m")
