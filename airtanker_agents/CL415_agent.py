from scooper_agent import ScooperAgent

class CL415Agent(ScooperAgent):
    def __init__(self, unique_id, model, base_position, time_step):
        config = {
            "drop_capacity": 200,
            "cruise_speed_knots": 180,
            "buffer_time": 10,
            "buffer_distance": 200,
            "fuel_capacity": 1530,
            "burn_rates": {
                "Taxi": 1,
                "Takeoff&Climb": 5,
                "Cruise": 3,
                "Landing": 2,
            },
            "state_times": {
                "Taxi to Runway": 5,
                "Takeoff&Climb": 8,
                "Dropping": 5,
                "Landing": 5,
                "Taxi to Station": 5,
                "Refill from Lake": 5,
                "Refuel": 20
            },

            "fuel_threshold": 0.3
        }
        super().__init__(unique_id, model, base_position, time_step=time_step, config=config)
