# Dash 8-400MRE agent
from airtankeragent import AirtankerAgent

class Dash8_400MREAgent(AirtankerAgent):
    def __init__(self, unique_id, model, base_position, time_step):
        config = {
            "drop_capacity": 336,
            "cruise_speed_knots": 360,
            "buffer_time": 10,
            "buffer_distance": 200,
            "fuel_capacity": 1492,
            "burn_rates": {
                "Taxi": 2,
                "Takeoff&Climb": 11,
                "Cruise": 6,
                "Landing": 5,
            },
            "state_times": {
                "Taxi to Runway": 5,
                "Takeoff&Climb": 16,
                "Dropping": 5,
                "Landing": 5,
                "Taxi to Station": 5,
                "Refill": 12,
                "Refuel and Refill": 20
            },

            "is_24_7": False,
            "fuel_threshold": 0.3
        }
        super().__init__(unique_id, model, base_position, time_step=time_step, config = config)
