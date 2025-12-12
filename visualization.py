"""
Visualization that can be used to run and observe a single ABM simulation in action with no tree search

"""

import plotly.graph_objects as go
import panel as pn
import threading
from model import WildfireModel
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from wind_schedule_utils import load_wind_schedule_from_csv_mean
pn.extension("plotly", theme="dark")


# Function to run the simulation loop in a separate thread
def simulation_loop(model, plot_panel, case_folder):
    while model.step():  # Step the model until the time limit is reached
        # if model.time == 0 or model.time % 1 == 0:  # Update the plot every 10 minutes
        #     plot_panel.object = model.plot_fig

        # if model.time == 0 or model.time % 1 == 0:  # Update the plot every 10 minutes
        plot_panel.object = model.plot_fig


if __name__ == "__main__":
    # Simulation parameters
    airtanker_counts = {
        "C130J": 0,
        "FireHerc": 4,
        "Scooper": 0,
        "AT802F": 0,
        "Dash8_400MRE": 0
    }
    lake_positions = [(5000, 5000)]  # Arbitrary lake locations
    # base_positions = [
    #     (20000, 20000),
    #     (-20000, 20000),
    #     (-20000, -20000),
    #     (20000, -20000),
    # ]  # Different base locations for each airtanker
    base_positions = [
        (20000, 20000)
    ]  # Different base locations for each airtanker




    time_step = 1  # Simulation time step in minutes
    buffer_time = 300  # Buffer time for agents
    buffer_distance = 6000
    overall_time_limit = 3000  # Simulation time limit in minutes
    fire_spread_sim_time = 3000
    operational_delay=0

    groundcrew_count = 1# spin up 3 ground-crew agents
    groundcrew_speed = 30

    # Define the start time of the simulation, e.g., 10:00 AM
    start_time = datetime.datetime.strptime("00:00", "%H:%M")
    wind_schedule = load_wind_schedule_from_csv_mean("wind_schedule_from_raws.csv")

    wind_schedule = load_wind_schedule_from_csv_mean("wind_schedule_camp_four.csv")

    # print("wind_schedule", wind_schedule)
    # Define the case folder for saving outputs
    case_folder = Path("Visualization_Case")
    case_folder.mkdir(parents=True, exist_ok=True)
    # groundcrew_sector_mapping = [0,1,2,3]
    second_groundcrew_sector_mapping = [2,4]

    crew_sector_map = [1,3]  # ← 2 crews → sector 0
    # Initialize the wildfire model
    model = WildfireModel(
        airtanker_counts=airtanker_counts,
        wind_speed=45,
        wind_direction= 230,
        base_positions=base_positions,
        lake_positions=lake_positions,
        time_step=time_step,
        debug=False,
        start_time=start_time,
        case_folder=case_folder,
        overall_time_limit=overall_time_limit,
        fire_spread_sim_time= fire_spread_sim_time,
        operational_delay=operational_delay,
        enable_plotting=True,
        groundcrew_count=groundcrew_count,  # spin up 3 ground-crew agents
        groundcrew_speed=groundcrew_speed,
        elapsed_minutes=200,
        groundcrew_sector_mapping = crew_sector_map,
        second_groundcrew_sector_mapping = second_groundcrew_sector_mapping,
        wind_schedule=wind_schedule,
        # wind_schedule=None,
        num_sectors=4
    )
    # (for example, to re-task on a button press)
    from groundcrew import GroundCrewAgent

    for gc, sector in zip(
            (ag for ag in model.schedule.agents if isinstance(ag, GroundCrewAgent)),
            crew_sector_map):
        if sector >= 0:  # skip “un-assigned”
            gc.assign_to_sector(sector)

    # # Set simulation start time to 3000 minutes
    # model.time = 2000
    # model.current_time = model.start_time + datetime.timedelta(minutes=2000)
    # Create a Panel object for interactive visualization
    plot_panel = pn.panel(model.plot_fig, width=1100, height=800)

    # Start the simulation loop in a separate thread
    simulation_thread = threading.Thread(
        target=simulation_loop, args=(model, plot_panel, case_folder), daemon=True
    )
    simulation_thread.start()

    # Serve the Panel layout
    pn.serve(plot_panel, show=True)
