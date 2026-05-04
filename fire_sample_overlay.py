"""
fire_sample_overlay.py
======================
Drop-in addition for dashboard.py that produces a figure showing:

  • The TRUTH fire perimeter (solid, opaque)
  • The MCTS-sampled fire perimeters stacked as semi-transparent overlays
    so your audience can see the *uncertainty envelope* the MCTS is planning over.

────────────────────────────────────────────────────────────────────────────
HOW TO INTEGRATE
────────────────────────────────────────────────────────────────────────────

1.  Copy this file into your project folder.

2.  In dashboard.py add near the top (after existing imports):

        from fire_sample_overlay import (
            FireSampleCollector,
            plot_truth_vs_sampled_fires,
        )
        fire_sample_collector = FireSampleCollector(max_samples=50)
        fire_overlay_panel    = pn.pane.Plotly(height=600,
                                               sizing_mode="stretch_width")

3.  In mcts.py, inside the `rollout()` function, just BEFORE `return reward`
    add ONE line:

        # ── collect sampled fire grid for the overlay figure ──────────────
        if callable(getattr(rollout, "_on_sample", None)):
            rollout._on_sample(current_state)
        # ──────────────────────────────────────────────────────────────────

    Then in dashboard.py, after you build `fire_sample_collector`, wire it up:

        import mcts as _mcts_module
        _mcts_module.rollout._on_sample = fire_sample_collector.add_sample

    (This uses a lightweight function attribute instead of a full callback
    registry – no changes to the mcts() signature needed.)

4.  In simulation_loop(), after the `root, _ = mcts(...)` call, add:

        fig = plot_truth_vs_sampled_fires(
            truth_model   = model,
            collector     = fire_sample_collector,
            current_time  = model.time,
        )
        fire_overlay_panel.object = fig
        fire_sample_collector.reset()        # clear for the next decision step

5.  Add fire_overlay_panel to your dashboard layout wherever you like, e.g.:

        pn.Column(..., fire_overlay_panel, ...)

────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Collector  ── gathers sampled fire grids during MCTS rollouts
# ─────────────────────────────────────────────────────────────────────────────

class FireSampleCollector:
    """
    Lightweight store for fire grids produced by MCTS rollouts.

    Each "sample" is a dict:
      {
        "fire_grid" : 2-D ndarray  (arrival-time grid, same shape as truth),
        "sim_time"  : float        (wall-clock sim time of the rollout endpoint),
        "schedule"  : list | None  (the wind schedule used, for labelling),
      }

    Call  .add_sample(rollout_sim_state)  from inside mcts.rollout().
    Call  .reset()  after you've consumed the samples for one decision step.
    """

    def __init__(self, max_samples: int = 60):
        self.max_samples = max_samples
        self._samples: list[dict] = []

    # ── public API ────────────────────────────────────────────────────────────

    def add_sample(self, sim_state) -> None:
        """
        Called from inside mcts.rollout() once the rollout has finished.
        sim_state  is the rollout's `current_state` (a WildfireModel clone).
        """
        if len(self._samples) >= self.max_samples:
            return  # cap the collection so memory stays bounded

        try:
            # Pull the fire arrival-time grid for the FULL simulation horizon
            grid = sim_state.fire.current_fire(
                sim_state.fire_spread_sim_time,
                max_time=sim_state.fire_spread_sim_time,
            )
            # current_fire may return a CuPy array – bring it to CPU
            if hasattr(grid, "get"):
                grid = grid.get()
            grid = np.array(grid, dtype=float)
        except Exception:
            return  # silently skip if anything goes wrong

        self._samples.append({
            "fire_grid" : grid,
            "sim_time"  : float(sim_state.fire_spread_sim_time),
            "schedule"  : getattr(sim_state, "_wind_schedule_used", None),
        })

    @property
    def samples(self) -> list[dict]:
        return self._samples

    def reset(self) -> None:
        self._samples = []

    def __len__(self) -> int:
        return len(self._samples)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Plotting function
# ─────────────────────────────────────────────────────────────────────────────

def plot_truth_vs_sampled_fires(
    truth_model,
    collector: FireSampleCollector,
    current_time: float,
    save_path: str,
    *,
    downsample: int = 1,
) -> None:
    """
    Save a matplotlib figure showing truth fire vs MCTS sampled fires.
    Saves directly to save_path (PNG).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fire     = truth_model.fire
    sim_time = truth_model.fire_spread_sim_time
    bounds   = fire.bounds

    def to_numpy(arr):
        if hasattr(arr, "get"):
            arr = arr.get()
        return np.array(arr, dtype=float)

    # ── grids ─────────────────────────────────────────────────────────────────
    fuel               = to_numpy(fire.fuel_model)
    truth_arrival      = to_numpy(fire.current_fire(current_time, max_time=sim_time))
    truth_full_arrival = to_numpy(fire.current_fire(sim_time,     max_time=sim_time))

    # ── downsample ────────────────────────────────────────────────────────────
    D              = max(1, int(downsample))
    fuel           = fuel          [::D, ::D]
    truth_arrival  = truth_arrival [::D, ::D]
    truth_full_arr = truth_full_arrival[::D, ::D]
    target_shape   = truth_arrival.shape

    # spatial extent for imshow
    extent  = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    samples  = collector.samples
    n_samples = len(samples)

    SAMPLE_COLORS = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#000075",
    ]

    burned_now = np.isfinite(truth_arrival) & (truth_arrival <= current_time)

    def _render_and_save(include_perimeter: bool, path: str):
        fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
        ax.set_facecolor("white")

        legend_elements = []
        for i, s in enumerate(samples):
            color = SAMPLE_COLORS[i % len(SAMPLE_COLORS)]
            g = to_numpy(s["fire_grid"])[::D, ::D]
            g = _match_shape(g, target_shape)
            burned_mask = np.ma.masked_where(~np.isfinite(g), np.ones_like(g))
            ax.imshow(burned_mask, extent=extent, origin="upper",
                      cmap=LinearSegmentedColormap.from_list(f"s{i}", [color, color]),
                      vmin=0, vmax=1, alpha=0.35, aspect="auto")
            legend_elements.append(Patch(facecolor=color, alpha=0.7, label=f"Sample {i + 1}"))

        if burned_now.sum() > 0:
            truth_burned_vals = np.where(burned_now, truth_arrival, np.nan)
            ax.imshow(truth_burned_vals, extent=extent, origin="upper",
                      cmap=LinearSegmentedColormap.from_list("truth", ["#ff0000", "#ff0000"]),
                      vmin=np.nanmin(truth_arrival), vmax=float(current_time),
                      alpha=1.0, aspect="auto")

        if include_perimeter and np.isfinite(truth_full_arr).any():
            binary = np.where(np.isfinite(truth_full_arr), 1.0, 0.0)
            ax.contour(binary, levels=[0.5],
                       colors=["black"], linewidths=[2.0],
                       extent=extent, origin="upper")

        legend_elements.append(Patch(facecolor="#ff0000", alpha=1.0,
                                     label=f"Truth fire (t={current_time:.0f} min)"))
        if include_perimeter:
            legend_elements.append(Line2D([0], [0], color="black", lw=2.0,
                                          label="Truth fire final perimeter"))

        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        ax.set_xlabel("X Coordinate (m)", color="black")
        ax.set_ylabel("Y Coordinate (m)", color="black")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
        ax.set_title(
            f"Truth Fire vs. MCTS-Sampled Fires  —  "
            f"t = {current_time:.0f} min  |  {n_samples} rollout samples",
            color="black", fontsize=14, pad=12,
        )
        ax.legend(handles=legend_elements, loc="lower right",
                  facecolor="white", edgecolor="black",
                  labelcolor="black", fontsize=9)

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[OVERLAY] Saved → {path}")

    # save both versions
    base, ext = save_path.rsplit(".", 1)
    _render_and_save(include_perimeter=True,  path=f"{base}_with_perimeter.{ext}")
    _render_and_save(include_perimeter=False, path=f"{base}_no_perimeter.{ext}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Alternative: side-by-side comparison figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_truth_and_samples_sidebyside(
    truth_model,
    collector: FireSampleCollector,
    current_time: float,
    n_cols: int = 4,
    dark_theme: bool = True,
) -> go.Figure:
    """
    Panel grid showing:
      • Top-left  : truth fire (large)
      • Remaining : individual sample fires at reduced size

    Useful for presentations where you want to show ACTUAL sample fires
    rather than an aggregate frequency map.

    Parameters
    ----------
    n_cols
        Number of columns in the sample grid.
    """
    samples = collector.samples
    n_show  = min(len(samples), n_cols * 2)   # at most 2 rows of samples
    if n_show == 0:
        return plot_truth_vs_sampled_fires(truth_model, collector,
                                           current_time, dark_theme=dark_theme)

    fire     = truth_model.fire
    sim_time = truth_model.fire_spread_sim_time
    bounds   = fire.bounds
    transform= fire.transform
    dx = abs(transform[0]); dy = abs(transform[4])
    x = np.arange(bounds.left,  bounds.right, dx)
    y = np.arange(bounds.top,   bounds.bottom, -dy)

    truth_arrival = np.array(fire.current_fire(current_time, max_time=sim_time), dtype=float)
    if hasattr(truth_arrival, "get"):
        truth_arrival = truth_arrival.get()

    template = "plotly_dark" if dark_theme else "plotly_white"

    # Total subplots: 1 truth + n_show samples
    n_total  = 1 + n_show
    n_rows   = (n_total + n_cols - 1) // n_cols
    titles   = [f"<b>TRUTH  (t={current_time:.0f} min)</b>"] + \
               [f"Sample {i+1}" for i in range(n_show)]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    def _add_fire_heatmap(row, col, grid, *, is_truth=False):
        """Add one fire heatmap to a subplot cell."""
        g = np.array(grid, dtype=float)
        g = _match_shape(g, truth_arrival.shape)
        burned = np.where((g > 0) & (g < np.inf), 1.0, np.nan)

        colorscale = (
            [[0, "rgba(255,220,0,0.9)"], [1, "rgba(255,100,0,0.9)"]]
            if is_truth else
            [[0, "rgba(255,80,0,0.45)"], [1, "rgba(160,0,0,0.45)"]]
        )

        fig.add_trace(
            go.Heatmap(z=burned, x=x, y=y,
                       colorscale=colorscale,
                       showscale=False,
                       hoverinfo="skip"),
            row=row, col=col,
        )

    # truth
    _add_fire_heatmap(1, 1, truth_arrival, is_truth=True)

    # samples
    for idx, s in enumerate(samples[:n_show]):
        cell  = idx + 2                         # 1-indexed, cell 1 = truth
        r     = (cell - 1) // n_cols + 1
        c     = (cell - 1) %  n_cols + 1
        _add_fire_heatmap(r, c, s["fire_grid"])

    fig.update_layout(
        template=template,
        title=dict(
            text=f"Truth Fire vs. MCTS Sampled Fires  —  t = {current_time:.0f} min",
            font=dict(size=16),
            x=0.5,
        ),
        height=320 * n_rows,
        margin=dict(t=60, l=10, r=10, b=30),
        showlegend=False,
    )
    # hide tick labels for all subplots (too cluttered)
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _match_shape(arr: np.ndarray, target: tuple) -> np.ndarray:
    """Crop or pad arr to match target shape."""
    r, c = target
    ar, ac = arr.shape

    # Crop if larger
    arr = arr[:r, :c]

    # Pad with NaN if smaller
    if arr.shape != (r, c):
        padded = np.full((r, c), np.nan)
        padded[:arr.shape[0], :arr.shape[1]] = arr
        arr = padded

    return arr


def _add_perimeter_contour(
    fig: go.Figure,
    arrival: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t_now: float,
    line_color: str,
) -> None:
    """
    Add a crisp perimeter contour for the fire front at t_now.
    Uses Plotly Contour with a single level at t_now.
    """
    arrival_safe = np.where(arrival <= 0, np.nan, arrival)
    fig.add_trace(go.Contour(
        z=arrival_safe,
        x=x, y=y,
        contours=dict(
            coloring="none",
            showlabels=False,
            start=t_now,
            end=t_now,
            size=1,
        ),
        line=dict(color=line_color, width=2.5),
        showscale=False,
        name=f"Truth perimeter (t={t_now:.0f} min)",
        hoverinfo="skip",
    ))
