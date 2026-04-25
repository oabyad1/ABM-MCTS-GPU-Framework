import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pathlib import Path

class Cmaps:
    def __init__(self):
        # Use the Reds colormap, but override the first color with pure red.
        original_cmap = mpl.colormaps["Reds"]
        fire_cmap = original_cmap(np.arange(original_cmap.N))
        red = np.array([1.0, 0.0, 0.0, 1.0])
        fire_cmap[0] = red
        self.fire_cmap = ListedColormap(fire_cmap)

        # Load the fuel colormap from the CSV file located in the main project directory.
        fuel_cmap_csv_path = Path(__file__).parent.joinpath("fuel_cmap.csv").resolve()
        fuel_cmap_df = pd.read_csv(fuel_cmap_csv_path, sep=",")
        fuel_model_list = list(
            zip(
                fuel_cmap_df["FF_RED"],
                fuel_cmap_df["FF_GREEN"],
                fuel_cmap_df["FF_BLUE"],
            )
        )
        self.fuel_cmap = ListedColormap(fuel_model_list)

        # Create an aspect colormap from black (opaque) to white (transparent).
        c1 = (1.0, 1.0, 1.0, 0.0)
        c2 = (0.0, 0.0, 0.0, 1.0)
        self.aspect_cmap = LinearSegmentedColormap.from_list("aspect_cmap", [c2, c1])

        # Create a drop colormap: transparent for clear cells, blue for drops.
        clear = (0.0, 0.0, 0.0, 0.0)
        drop = (0.0, 0.0, 1.0, 1.0)
        self.drop_cmap = ListedColormap([clear, drop])


def plot_fire(surrogate, time=0, max_time=None, contour_time=800,
              aspect_transparency=0.5, fire_transparency=0.5,
              contour_color="black", ax=None, cmaps=None):
    """
    Plot fire growth for a SurrogateFireModel.

    This function overlays the fuel model, an aspect (if available), and the current fire state
    (cells with arrival times <= 'time' are burned, and if max_time is provided, cells with arrival
    times above max_time are masked) using imshow. It also adds a contour outlining the fire's boundary.

    Parameters:
      surrogate: An instance of SurrogateFireModel.
      time: Time threshold (e.g. current simulation time) to determine the burned area.
      max_time: If provided, cells with arrival times above this value will be masked.
      contour_time: (Optional) used here to set contouring details (for future extensions).
      aspect_transparency: Transparency for the aspect overlay.
      fire_transparency: Transparency for the fire state overlay.
      contour_color: Color for the contour line.
      ax: A matplotlib Axes instance. If None, a new figure and axes are created.
      cmaps: An instance of Cmaps for color mapping. If None, a new one is created.

    Returns:
      fig, ax: The matplotlib Figure and Axes containing the plot.
    """
    if cmaps is None:
        cmaps = Cmaps()
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title(f"Fire growth at t = {time} min")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
    else:
        fig = ax.figure

    # Get the extent from the surrogate's TIFF file bounds.
    with rasterio.open(surrogate.tif_path) as src:
        bounds = src.bounds
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    # Plot the fuel model.
    ax.imshow(surrogate.fuel_model, cmap=cmaps.fuel_cmap,
              extent=extent, origin='upper')

    # Optionally overlay the aspect if available.
    if hasattr(surrogate, "aspect_np"):
        ax.imshow(surrogate.aspect_np, cmap="Greys",
                  alpha=aspect_transparency, extent=extent, origin='upper')

    # Plot the current fire state, limiting the arrival time if max_time is provided.
    fire_state = surrogate.current_fire(time, max_time=max_time)
    im = ax.imshow(fire_state, cmap=cmaps.fire_cmap,
                   alpha=fire_transparency, extent=extent, origin='upper')

    # Create coordinate arrays to be used in the contour.
    nrows, ncols = surrogate.fuel_model.shape
    x = np.linspace(extent[0], extent[1], ncols)
    y = np.linspace(extent[3], extent[2], nrows)

    # Add a contour line at the boundary (e.g., where fire_state == 0).
    cs = ax.contour(x, y, fire_state, levels=[0], colors=contour_color, linewidths=1)

    # Add a colorbar for the fire state.
    fig.colorbar(im, ax=ax, label='Arrival Time')
    return fig, ax
