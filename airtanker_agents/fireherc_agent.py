# fireherc_agent.py
from airtankeragent import AirtankerAgent

class FireHercAgent(AirtankerAgent):
    def __init__(self, unique_id, model, base_position, time_step):
        config = {
            "drop_capacity": 380,
            "cruise_speed_knots": 300,
            "buffer_time": 0,
            "buffer_distance": 200,
            "fuel_capacity": 6716,
            "burn_rates": {
                "Taxi": 2,
                "Takeoff&Climb": 20,
                "Cruise": 11,
                "Landing": 5,
            },
            # ── deterministic-mode flags  ───────────────────────────
            "deterministic_fire_target": True,
            "fixed_fire_target_angle": 90,  # None ⇒ spacing rule
            "state_times": {
                "Taxi to Runway": 5,
                "Takeoff&Climb": 8,
                "Dropping": 5,
                "Landing": 5,
                "Taxi to Station": 5,
                "Refill": 12,
                "Refuel and Refill": 20
            },

            "is_24_7": True,
            "fuel_threshold": 0.3

        }
        super().__init__(unique_id, model, base_position, time_step=time_step, config = config)



