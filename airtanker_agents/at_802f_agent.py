# Air Tractor AT-802F agent
from airtankeragent import AirtankerAgent

class AT802FAgent(AirtankerAgent):
    def __init__(self, unique_id, model, base_position, time_step):
        config = {
            "drop_capacity": 105,
            "cruise_speed_knots": 200,
            "buffer_time": 10,
            "buffer_distance": 200,
            "fuel_capacity": 379,
            "burn_rates": {
                "Taxi": 0.2,
                "Takeoff&Climb": 1.6,
                "Cruise": 1,
                "Landing": 0.5,
            },
            "state_times": {
                "Taxi to Runway": 5,
                "Takeoff&Climb": 8,
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
