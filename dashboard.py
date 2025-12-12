# truth_dashboard.py
#
# A single-file Panel dashboard that lets you
#   1. choose all fire-model parameters once,
#   2. lock them in with “Set Simulation Parameters”,
#   3. run/pause Monte-Carlo Tree-Search loops indefinitely
#      without ever rebuilding the model (so the tree stays intact).
#

import panel as pn
import threading
import datetime
import math
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import time as t
from forecast_provider import get_forecast
from wind_schedule_utils import load_wind_schedule_from_csv_random
from wind_schedule_utils import load_wind_schedule_from_csv_mean
from wind_schedule_utils import load_wind_schedule_from_csv_sigma
from wind_schedule_utils import _print_schedule
import plotly.io as pio

# ── use Kaleido & give it defaults ──────────────────────────────
# pio.kaleido.scope.default_format = "png"
# pio.kaleido.scope.default_scale  = 2          # 2× pixel density
# pio.kaleido.scope.default_width  = 800        # optional global defaults
# pio.kaleido.scope.default_height = 450
import kaleido                     # make sure import succeeds
import plotly.io as pio
pio.kaleido.scope.mathjax = None   # tiny speed improvement
from model import WildfireModel
from mcts import (
    clone_simulation,
    mcts,
    simulate_in_place,
    hierarchy_pos_tree,
    count_expandable_assets,
)


from pathlib import Path

# ── wind-rose config ───────────────────────────────────────────
NUM_WIND_ROSES = 16          # ← change this to any number you like
ROSES_PER_ROW  = 4          # 4 keeps each row narrow enough for most screens
# ───────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
# CUDA / CuPy warm-up – do this ONCE when the process starts
# ────────────────────────────────────────────────────────────────
import cupy as cp, time
# import cupy as cp
# from surrogate_fire_model_CK2_multi_phase import SurrogateFireModelROS_CK2Multi
#
# TARGET_POOL_BYTES = 4_800_000_000         # hard-coded or imported
#
# def _cuda_warmup():
#     # 1) install a private memory pool
#     mpool = cp.cuda.MemoryPool()
#     cp.cuda.set_allocator(mpool.malloc)
#
#     # 2) pre-allocate the desired amount in nice big slabs
#     chunk = 1 << 26                         # 64 MiB
#     allocated = []
#     while sum(a.nbytes for a in allocated) < TARGET_POOL_BYTES:
#         allocated.append(cp.empty(chunk, dtype=cp.uint8))
#     del allocated[:]                        # return to the pool
#
#     # 3) trigger JIT compilation once
#     _ = SurrogateFireModelROS_CK2Multi(
#             "TEST_TIF.tif",
#             sim_time           = 60,
#             wind_speed         = 0,
#             wind_direction_deg = 0,
#             max_iter           = 20,
#             tol                = 1e-3,
#             wind_schedule      = None)
#     cp.cuda.Stream.null.synchronize()
#
# _cuda_warmup()          # ← runs exactly once, before Panel starts
# ────────────────────────────────────────────────────────────────


pn.extension("plotly", theme="dark", notifications=True)
# truth_schedule = load_wind_schedule_from_csv("wind_schedule_natural.csv")


# truth_schedule = load_wind_schedule_from_csv_mean("wind_schedule_from_raws.csv")

# TRUTH_SEED      =10   # 42 → any seed you like
# truth_schedule = load_wind_schedule_from_csv_random("wind_schedule_from_raws.csv",
#                                              seed=TRUTH_SEED)
# from forecast_provider import set_truth_schedule
# set_truth_schedule(truth_schedule)


#
# TRUTH_SEED  = 17         # Pick any integer; change → different ± pattern
# SIGMA_LEVEL = 3.0        # “3 σ” extreme scenario
#
# truth_schedule = load_wind_schedule_from_csv_sigma(
#     "wind_schedule_from_raws.csv",
#     sigma=SIGMA_LEVEL,
#     seed=TRUTH_SEED)
#
#
# from forecast_provider import (
#     set_truth_schedule,
# )
# _print_schedule(truth_schedule, title=f"⚑ {SIGMA_LEVEL}σ truth schedule")
# set_truth_schedule(truth_schedule)






from forecast_provider import (
    set_truth_schedule,
    set_background_schedule
)
from wind_schedule_utils import (
    load_wind_schedule_from_csv_mean,
    load_wind_schedule_from_csv_sigma,  # or _random / whatever produces truth
)

# 1) background “climatology” (means)
BACKGROUND_SCHED = load_wind_schedule_from_csv_mean("wind_schedule_camp.csv")
_print_schedule(BACKGROUND_SCHED, title=f"⚑ background sched")
set_background_schedule(BACKGROUND_SCHED)

# 2) truth schedule (extreme, random, etc.)
TRUTH_SEED  = 25
# truth_schedule = load_wind_schedule_from_csv_sigma(
#                     "wind_schedule_camp.csv",
#                     sigma=3.0,
#                     seed=TRUTH_SEED)
truth_schedule = load_wind_schedule_from_csv_mean("wind_schedule_camp.csv")


_print_schedule(truth_schedule, title=f"⚑ hard σ truth schedule")
set_truth_schedule(truth_schedule)


# ────────────────────────────────────────────────────────────────
def _print_schedule_tuples(schedule: list[tuple], *, title=""):
    """
    Pretty-print a (start, end, speed, dir) list.
    Times are minutes since t = 0; speed & dir are whatever units
    you store (kt, m s⁻¹, deg true …).
    """
    import pandas as pd
    if title:
        print(f"\n{title}")
    df = pd.DataFrame(schedule, columns=["start", "end", "speed", "dir"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
# ────────────────────────────────────────────────────────────────



# ───── Forecast-evolution helpers ───────────────────────────────────────────
# ── put this near the top of truth_dashboard.py (after the imports) ─────────
def _debug_dump_schedule(model, *, header=""):
    """Pretty-print the entire wind-schedule of *model*."""
    import pandas as pd
    cols = ["start", "end", "speed", "direction"]
    df = pd.DataFrame(model.wind_schedule, columns=cols)
    print(f"\n{header}  —  t = {model.time:.0f} min")
    print(df.to_string(index=False))
# ────────────────────────────────────────────────────────────────────────────

def _debug_dump_forecast(fore_df: pd.DataFrame, *, now: int, hdr=""):
    """Pretty-print μ/σ for every future segment *before* sampling."""
    if hdr:
        print(f"\n{hdr}  —  t = {now} min")
    show = fore_df.copy()
    show["lead"] = show.start_min - now          # minutes ahead
    cols = ["lead", "start_min", "end_min",
            "speed_mean", "speed_std", "dir_mean", "dir_std"]
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:6.2f}"))


def _make_baseline_schedule(truth_model, forecast_df):
    """Return full 0-→end schedule made of past truth + future means."""
    past = [seg for seg in truth_model.wind_schedule            # end ≤ now
            if seg[1] <= truth_model.time]

    future_mu = [
        (int(r.start_min), int(r.end_min),
         float(r.speed_mean), float(r.dir_mean))
        for r in forecast_df.itertuples()
        if r.end_min > truth_model.time
    ]

    return past + future_mu

from plotly.subplots import make_subplots
forecast_history: list[tuple[int, pd.DataFrame]] = []   # (issue_minute, df)
forecast_evo_panel    = pn.pane.Plotly(height=320, sizing_mode="stretch_width")

def record_forecast(issue_minute: int, df: pd.DataFrame):
    """Keep every forecast table we get."""
    forecast_history.append((issue_minute, df.copy()))

from scipy.stats import norm
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def _build_evolution_fig() -> go.Figure | None:
    if not forecast_history:
        return None

    BANDS       = [0.674, 1.036, 1.645]                     # 50 %, 70 %, 90 %
    BAND_NAMES  = ["50 %", "70 %", "90 %"]
    BAND_COLORS = ["rgba(255,200,0,0.60)",
                   "rgba(255,120,0,0.40)",
                   "rgba(255, 60,0,0.25)"]

    # ---------- long table --------------------------------------------------
    rows = []
    for issue_min, fdf in forecast_history:
        for r in fdf.itertuples():
            rows.append(dict(issue=issue_min,
                             mid=0.5*(r.start_min+r.end_min),
                             mu_s=r.speed_mean,  sd_s=r.speed_std,
                             mu_d=r.dir_mean,    sd_d=r.dir_std))
    df = pd.DataFrame(rows)
    first_issue = df.issue.min()

    # ---------- figure skeleton --------------------------------------------
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=("wind-speed – probability fan chart",
                                        "wind-direction – probability fan chart"))

    # ---------- speed panel -------------------------------------------------
    for iss, sub in df.groupby("issue"):
        # 🔽 NEW — unwrap 0/360° jumps so Plotly won't draw long diagonals
        sub = sub.copy()  # keep the original intact
        sub["mu_d_unwrap"] = (
                np.unwrap(np.deg2rad(sub.mu_d), discont=np.pi) * 180 / np.pi
        )
        for z, name, col in zip(BANDS[::-1], BAND_NAMES[::-1], BAND_COLORS[::-1]):
            hi = sub.mu_s + z*sub.sd_s
            lo = sub.mu_s - z*sub.sd_s
            fig.add_trace(
                go.Scatter(
                    x = pd.concat([sub.mid, sub.mid[::-1]]),
                    y = pd.concat([hi,       lo[::-1]]),
                    fill="toself", line=dict(width=0),
                    fillcolor=col,
                    name=f"{name} band",                     # legend label
                    legendgroup="spd",
                    showlegend=bool(iss == first_issue)),    # only once
                row=1, col=1)
        # median
        fig.add_trace(go.Scatter(x=sub.mid, y=sub.mu_s,
                                 mode="lines", line=dict(width=1),
                                 name="μ", legendgroup="spd",
                                 showlegend=False),
                      row=1, col=1)

    # ---------- direction panel --------------------------------------------
    # ---------- direction panel --------------------------------------------
    latest_issue = df.issue.max()  # draw the median only for newest run

    for iss, sub in df.groupby("issue"):

        # unwrap so the lines stay inside the plot (no 0↔360° jumps)
        sub = sub.copy()
        sub["mu_d_unwrap"] = (
                np.unwrap(np.deg2rad(sub.mu_d), discont=np.pi) * 180 / np.pi
        )

        #  fan bands ──────────────────────────────────────────────────────
        for z, col in zip(BANDS[::-1], BAND_COLORS[::-1]):
            hi = sub.mu_d_unwrap + z * sub.sd_d
            lo = sub.mu_d_unwrap - z * sub.sd_d

            fig.add_trace(
                go.Scatter(
                    x=pd.concat([sub.mid, sub.mid[::-1]]),
                    y=pd.concat([hi, lo[::-1]]),
                    fill="toself",
                    fillcolor=col,
                    line=dict(width=0),
                    showlegend=False),
                row=2, col=1)

        #  median line (only for *latest* forecast) ───────────────────────
        if iss == latest_issue:
            fig.add_trace(
                go.Scatter(
                    x=sub.mid,
                    y=sub.mu_d_unwrap,
                    mode="lines",
                    line=dict(width=2, color="white"),
                    name="μ (latest)"),
                row=2, col=1)

    # ---------- cosmetics ---------------------------------------------------
    fig.update_yaxes(title_text="m s⁻¹", row=1, col=1)
    fig.update_yaxes(title_text="deg",   row=2, col=1, range=[0, 360])
    fig.update_xaxes(title_text="minutes from now", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=320,
                      margin=dict(t=42, l=8, r=8, b=0),
                      legend=dict(orientation="h", y=-0.18,
                                  title="central-probability band"))
    return fig
def update_forecast_evo_panel():
    fig = _build_evolution_fig()
    if fig is not None:
        forecast_evo_panel.object = fig



###################################################################################################
###################################################################################################
###################################################################################################
################# -----PROBABILITY APPROAH ---- #####################################################
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import norm

# _BANDS       = [0.674, 1.036, 1.645]                 # 50 %, 70 %, 90 %
# _BAND_NAMES  = ["50 %", "70 %", "90 %"]
# _BAND_COLORS = ["rgba(255,200,0,0.60)",
#                 "rgba(255,120,0,0.40)",
#                 "rgba(255, 60,0,0.25)"]
#
# def _fan_chart_from_forecast(fore_df: pd.DataFrame, *, now_min: int) -> go.Figure:
#     """
#     One stacked figure:
#       row 1 → wind-speed fan (μ ± z·σ)
#       row 2 → wind-direction fan, unwrapped
#     Only the **latest** forecast is shown.
#     The y-axes are auto-zoomed to the 90 % band (+5 % padding).
#     """
#     latest = fore_df.copy()
#     latest = latest[latest.end_min > now_min]          # future only
#     latest["mid"] = 0.5 * (latest.start_min + latest.end_min)
#     latest["dir_mu_unwrap"] = (
#         np.unwrap(np.deg2rad(latest.dir_mean), discont=np.pi) * 180/np.pi
#     )
#
#     fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
#                         vertical_spacing=0.08,
#                         subplot_titles=("wind-speed – probability fan chart",
#                                         "wind-direction – probability fan chart"))
#
#     # ── speed panel ──────────────────────────────────────────────
#     for z, name, col in zip(_BANDS[::-1], _BAND_NAMES[::-1], _BAND_COLORS[::-1]):
#         hi = latest.speed_mean + z*latest.speed_std
#         lo = latest.speed_mean - z*latest.speed_std
#         fig.add_trace(
#             go.Scatter(x=pd.concat([latest.mid, latest.mid[::-1]]),
#                        y=pd.concat([hi,         lo[::-1]]),
#                        fill="toself", line=dict(width=0),
#                        fillcolor=col,
#                        name=f"{name} band",
#                        legendgroup="spd",
#                        showlegend=(name == "90 %")),
#             row=1, col=1)
#     fig.add_trace(go.Scatter(x=latest.mid, y=latest.speed_mean,
#                              mode="lines", line=dict(width=2, color="white"),
#                              name="μ (latest)"),
#                   row=1, col=1)
#
#     # ── direction panel ─────────────────────────────────────────
#     for z, col in zip(_BANDS[::-1], _BAND_COLORS[::-1]):
#         hi = latest.dir_mu_unwrap + z*latest.dir_std
#         lo = latest.dir_mu_unwrap - z*latest.dir_std
#         fig.add_trace(
#             go.Scatter(x=pd.concat([latest.mid, latest.mid[::-1]]),
#                        y=pd.concat([hi,         lo[::-1]]),
#                        fill="toself", fillcolor=col,
#                        line=dict(width=0), showlegend=False),
#             row=2, col=1)
#     fig.add_trace(go.Scatter(x=latest.mid, y=latest.dir_mu_unwrap,
#                              mode="lines", line=dict(width=2, color="white"),
#                              showlegend=False),
#                   row=2, col=1)
#
#     # ── dynamic y-ranges ────────────────────────────────────────
#     # 90 % band = the widest one (z = 1.645)
#     sp_hi = (latest.speed_mean + _BANDS[-1]*latest.speed_std).max()
#     sp_lo = max(0, (latest.speed_mean - _BANDS[-1]*latest.speed_std).min())
#     pad_s = 0.05 * (sp_hi - sp_lo)
#     fig.update_yaxes(range=[sp_lo - pad_s, sp_hi + pad_s], title_text="m s⁻¹",
#                      row=1, col=1)
#
#     dir_hi = (latest.dir_mu_unwrap + _BANDS[-1]*latest.dir_std).max()
#     dir_lo = (latest.dir_mu_unwrap - _BANDS[-1]*latest.dir_std).min()
#     pad_d  = 0.05 * (dir_hi - dir_lo)
#     fig.update_yaxes(range=[dir_lo - pad_d, dir_hi + pad_d], title_text="deg",
#                      row=2, col=1)
#
#     # ── cosmetics ───────────────────────────────────────────────
#     fig.update_xaxes(title_text="minutes from now", row=2, col=1)
#     fig.update_layout(template="plotly_dark",
#                       height=440,                    # ← bigger!
#                       margin=dict(t=50, l=8, r=8, b=0),
#                       legend=dict(orientation="h", y=-0.20,
#                                   title="central-probability band"))
#     return fig
#
#
# # ── panel updater (unchanged usage) ─────────────────────────────
# def update_fan_plot_panel(forecast_df: pd.DataFrame, now_minute: int):
#     """Replace *wind_rose_panel* with the new fan chart."""
#     global wind_rose_panel
#     fig = _fan_chart_from_forecast(forecast_df, now_min=now_minute)
#     wind_rose_panel.objects = [
#         pn.pane.Markdown(
#             "### <span style='color:white'>Wind forecast fan chart "
#             "(μ ± σ bands, next 8 h)</span>",
#             sizing_mode="stretch_width"),
#         pn.pane.Plotly(fig, height=540, sizing_mode="stretch_width"),
#     ]


###################################################################################################
###################################################################################################
###################################################################################################
################# -----SIGMA  APPROAH ---- #####################################################



# ±1 σ, ±2 σ, ±3 σ bands (dark→light)
_STD_SIGMAS  = [1, 2, 3]
_BAND_NAMES  = ["±1 σ", "±2 σ", "±3 σ"]
_BAND_COLORS = ["rgba(255,200,0,0.60)",
                "rgba(255,120,0,0.40)",
                "rgba(255, 60,0,0.25)"]


def _fan_chart_from_forecast(fore_df: pd.DataFrame, *, now_min: int) -> go.Figure:
    """
    Return a stacked fan-chart figure:
      • Row 1  wind-speed  (mean ± k·σ, k = 1,2,3)
      • Row 2  wind-direction (mean ± k·σ, unwrapped)
    Only rows with end_min > now_min are plotted (i.e. the future forecast).
    Y-axes auto-zoom to ±3 σ with 5 % padding.
    """
    latest = fore_df[fore_df.end_min > now_min].copy()
    latest["mid"] = 0.5 * (latest.start_min + latest.end_min)
    latest["dir_mu_unwrap"] = (
        np.unwrap(np.deg2rad(latest.dir_mean), discont=np.pi) * 180 / np.pi
    )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=("wind-speed – fan chart (σ)",
                                        "wind-direction – fan chart (σ)"))

    # ── Row 1 · speed ──────────────────────────────────────────────
    for k, name, col in zip(_STD_SIGMAS[::-1], _BAND_NAMES[::-1], _BAND_COLORS[::-1]):
        hi = latest.speed_mean + k * latest.speed_std
        lo = latest.speed_mean - k * latest.speed_std
        fig.add_trace(
            go.Scatter(x=pd.concat([latest.mid, latest.mid[::-1]]),
                       y=pd.concat([hi,         lo[::-1]]),
                       fill="toself", line=dict(width=0),
                       fillcolor=col, name=name,
                       legendgroup="spd",
                       showlegend=(name == "±3 σ")),          # one legend label
            row=1, col=1)
    fig.add_trace(go.Scatter(x=latest.mid, y=latest.speed_mean,
                             mode="lines", line=dict(width=2, color="white"),
                             name="μ", legendgroup="spd", showlegend=False),
                  row=1, col=1)

    # ── Row 2 · direction ─────────────────────────────────────────
    for k, col in zip(_STD_SIGMAS[::-1], _BAND_COLORS[::-1]):
        hi = latest.dir_mu_unwrap + k * latest.dir_std
        lo = latest.dir_mu_unwrap - k * latest.dir_std
        fig.add_trace(
            go.Scatter(x=pd.concat([latest.mid, latest.mid[::-1]]),
                       y=pd.concat([hi,         lo[::-1]]),
                       fill="toself", line=dict(width=0),
                       fillcolor=col, showlegend=False),
            row=2, col=1)
    fig.add_trace(go.Scatter(x=latest.mid, y=latest.dir_mu_unwrap,
                             mode="lines", line=dict(width=2, color="white"),
                             showlegend=False),
                  row=2, col=1)

    # ── Dynamic y-ranges (±3 σ + 5 % pad) ─────────────────────────
    sp_hi = (latest.speed_mean + 3 * latest.speed_std).max()
    sp_lo = max(0, (latest.speed_mean - 3 * latest.speed_std).min())
    pad_s = 0.05 * (sp_hi - sp_lo)
    fig.update_yaxes(range=[sp_lo - pad_s, sp_hi + pad_s],
                     title_text="m s⁻¹", row=1, col=1)

    dir_hi = (latest.dir_mu_unwrap + 3 * latest.dir_std).max()
    dir_lo = (latest.dir_mu_unwrap - 3 * latest.dir_std).min()
    pad_d  = 0.05 * (dir_hi - dir_lo)
    fig.update_yaxes(range=[dir_lo - pad_d, dir_hi + pad_d],
                     title_text="deg", row=2, col=1)

    # ── cosmetics ────────────────────────────────────────────────
    fig.update_xaxes(title_text="minutes from now", row=2, col=1)
    fig.update_layout(template="plotly_dark",
                      height=440,
                      margin=dict(t=52, l=8, r=8, b=0),
                      legend=dict(orientation="h", y=-0.22,
                                  title="σ-band"))

    return fig


def update_fan_plot_panel(forecast_df: pd.DataFrame, now_minute: int) -> None:
    """
    Build / refresh the fan chart and replace the content of
    the global ``wind_rose_panel`` (now a *fan-plot panel*).
    """
    global wind_rose_panel
    fig = _fan_chart_from_forecast(forecast_df, now_min=now_minute)
    wind_rose_panel.objects = [
        pn.pane.Markdown(
            "### <span style='color:white'>Wind forecast fan chart "
            "(μ ± σ bands, next 8 h)</span>",
            sizing_mode="stretch_width"),
        pn.pane.Plotly(fig, height=540, sizing_mode="stretch_width"),
    ]




# ────────────────────────────────────────────────────────────────
# Globals & synchronisation primitives
# ────────────────────────────────────────────────────────────────
pause_event           = threading.Event()   # “soft” pause (does NOT kill the thread)
pause_event.set()                            # dashboard launches in the paused state
simulation_stop_event = threading.Event()   # used when we rebuild a fresh model
sim_thread            = None                # handle of the background loop
truth_model           = None                # will be constructed by the user

# # ────────────────────────────────────────────────────────────────
# # Wind-rose helpers (fake data for now)
# # ────────────────────────────────────────────────────────────────
# def generate_fake_wind_data(hour: int) -> pd.DataFrame:
#     directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
#     speeds     = ['0-5', '5-10', '10-15', '15-20']
#     rows       = []
#     for d in directions:
#         for s in speeds:
#             rows.append(
#                 dict(direction=d,
#                      strength=s,
#                      frequency=np.random.uniform(0, 15),   # fake probability
#                      hour=f'Hour {hour}')
#             )
#     return pd.DataFrame(rows)
#
# speed_bins = ['0-5', '5-10', '10-15', '15-20']
# color_map  = {
#     '0-5'   : px.colors.sequential.Plasma_r[0],
#     '5-10'  : px.colors.sequential.Plasma_r[3],
#     '10-15' : px.colors.sequential.Plasma_r[6],
#     '15-20' : px.colors.sequential.Plasma_r[9],
# }
#
# def make_wind_rose(hour: int, *, show_legend=True):
#     df = generate_fake_wind_data(hour)
#     df['frequency'] = 100 * df['frequency'] / df['frequency'].sum()
#     df['strength']  = pd.Categorical(df['strength'], categories=speed_bins, ordered=True)
#
#     fig = px.bar_polar(
#         df, r="frequency", theta="direction", color="strength",
#         template="plotly_dark",
#         category_orders={"strength": speed_bins},
#         color_discrete_map=color_map,
#     )
#     fig.update_layout(
#         title=f"Wind Forecast – Hour {hour}",
#         showlegend=show_legend,
#         margin=dict(t=40, l=10, r=10, b=10),
#         height=300,
#         polar=dict(radialaxis=dict(ticksuffix="%", showticklabels=True))
#     )
#     return pn.pane.Plotly(fig, sizing_mode='stretch_width', height=300)
# ────────────────────────────────────────────────────────────────
# Wind-rose helpers (real μ/σ from the forecast)
# ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")                    # safe for headless servers

# speed bins (kt or mph – whatever your µ/σ units are)
SPEED_BIN_EDGES  = [0, 5, 10, 15, 20, 999]
SPEED_BIN_LABELS = ["0-5", "5-10", "10-15", "15-20", "20+"]

DIR_BEARINGS = np.arange(0, 360, 45)     # 8-point compass Rose

def _sample_from_row(row, n=400):
    """Monte-Carlo sample n draws from one forecast row’s μ/σ."""
    rng = np.random.default_rng()
    speeds = rng.normal(row.speed_mean, row.speed_std, n)
    dirs   = rng.normal(row.dir_mean,   row.dir_std,   n) % 360
    return speeds, dirs

def wind_rose_from_forecast(forecast_df, window_start, window_end):
    subset = forecast_df[
        (forecast_df.start_min < window_end) &
        (forecast_df.end_min   > window_start)
    ]
    if subset.empty:
        return None

    # ── Monte-Carlo sample ────────────────────────────────────────────
    speeds, dirs = [], []
    for r in subset.itertuples():
        s, d = _sample_from_row(r, n=500)
        speeds.append(s); dirs.append(d)
    speeds = np.concatenate(speeds)
    dirs   = np.concatenate(dirs)

    # ── Bin into categories ───────────────────────────────────────────
    speed_bins = pd.cut(speeds, SPEED_BIN_EDGES, labels=SPEED_BIN_LABELS)
    dir_bins = pd.Series(
        pd.cut(dirs,
               bins=np.append(DIR_BEARINGS, 360),
               labels=[str(b) for b in DIR_BEARINGS]),
        name="direction")
    # drop points that did not fall into a bin
    mask = (~speed_bins.isna()) & (~dir_bins.isna())
    if mask.sum() == 0:
        return None

    table = (
        pd.crosstab(dir_bins[mask], speed_bins[mask], normalize="all") * 100
    )
    if table.empty:
        return None

    table = (table
             .reset_index()
             .melt(id_vars="direction",
                   var_name="strength",
                   value_name="frequency"))
    return table



# ───────────────────────────────────────────────────────────────
# Probability wind-rose (stacked barpolar)
# ───────────────────────────────────────────────────────────────\
NUM_DIR_SECTORS = 24          # 8 → 45° ; 16 → 22.5° ; 24 → 15°, etc.

SPEED_BIN_EDGES  = [0, 5, 10, 15, 20, 999]
SPEED_BIN_LABELS = ["0-5", "5-10", "10-15", "15-20", "20+"]
DIR_BEARINGS  = np.arange(0, 360, 360 / NUM_DIR_SECTORS)
BAR_WIDTH_DEG = 360 / NUM_DIR_SECTORS
_COLOR_MAP = {
    lab: col for lab, col in zip(
        SPEED_BIN_LABELS,
        px.colors.sequential.Plasma_r[:len(SPEED_BIN_LABELS)])
}

def _bin_samples(speeds, dirs):
    """Return dict{speed_bin -> counts array[len DIR_BEARINGS]}"""
    # digitize once for speed & direction
    s_idx = np.digitize(speeds, SPEED_BIN_EDGES) - 1    # 0..len-2
    d_idx = np.digitize(dirs  % 360, DIR_BEARINGS, right=False) - 1
    d_idx[d_idx == -1] = len(DIR_BEARINGS) - 1          # wrap 360°

    counts = {
        lab: np.zeros(len(DIR_BEARINGS), dtype=int)
        for lab in SPEED_BIN_LABELS
    }
    for s, d in zip(s_idx, d_idx):
        if 0 <= s < len(SPEED_BIN_LABELS):
            counts[SPEED_BIN_LABELS[s]][d] += 1
    return counts

def probability_wind_rose(forecast_df, window_start, window_end):
    subset = forecast_df[
        (forecast_df.start_min < window_end) &
        (forecast_df.end_min   > window_start)
    ]
    if subset.empty:
        return pn.pane.Markdown(f"*no forecast data*")

    # Monte-Carlo sample
    rng = np.random.default_rng()
    speeds, dirs = [], []
    for r in subset.itertuples():
        s = rng.normal(r.speed_mean, r.speed_std, 800)
        d = rng.normal(r.dir_mean,   r.dir_std,   800)
        speeds.append(s); dirs.append(d)
    speeds = np.concatenate(speeds); dirs = np.concatenate(dirs)

    bins = _bin_samples(speeds, dirs)
    total = len(speeds)
    theta_centres = DIR_BEARINGS + BAR_WIDTH_DEG / 2



    fig = go.Figure()
    cum = np.zeros(len(DIR_BEARINGS))            # stack offset
    for lab in SPEED_BIN_LABELS[::-1]:           # draw largest first
        probs = 100 * bins[lab] / total
        fig.add_trace(go.Barpolar(
            r     = probs,
            theta = theta_centres,
            width = BAR_WIDTH_DEG,
            base  = cum,
            name  = lab,
            marker_color=_COLOR_MAP[lab],
            hovertemplate="%{theta:.0f}°<br>"+lab+" kt<br>%{r:.1f}%<extra></extra>"
        ))
        cum += probs

    fig.update_layout(
        template="plotly_dark",
        title=f"{window_start}-{window_end} min  •  draw probability (%)",
        height=260,
        legend=dict(orientation="h", y=-0.13),
        polar=dict(
            angularaxis=dict(direction="clockwise", rotation=90),
            radialaxis=dict(showticklabels=True, ticksuffix="%"),
        ),
        margin=dict(t=40, b=40, l=10, r=10),
    )
    return pn.pane.Plotly(fig, height=260, sizing_mode="stretch_width")

def make_wind_rose(table, title, *, show_legend=False):
    """
    Convert the aggregated table into a Plotly polar bar chart.
    """
    if table is None:                     # no rows for that window
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title=f"{title}<br>(no data)")
        return pn.pane.Plotly(fig, height=260)

    colors = {l: c for l, c in zip(
        SPEED_BIN_LABELS, px.colors.sequential.Plasma_r[:len(SPEED_BIN_LABELS)]
    )}

    fig = px.bar_polar(
        table, r="frequency", theta="direction", color="strength",
        category_orders={"strength": SPEED_BIN_LABELS},
        color_discrete_map=colors,
        template="plotly_dark",
    )
    fig.update_layout(
        title=title,
        showlegend=show_legend,
        margin=dict(t=40, l=0, r=0, b=0),
        height=260,
        polar=dict(radialaxis=dict(ticksuffix="%", showticklabels=True,
                                   gridcolor="rgba(200,200,200,0.2)"))
    )
    return pn.pane.Plotly(fig, height=260, sizing_mode="stretch_width")

# def update_wind_rose_panel(forecast_df, now_minute):
#     """
#     Replace the global wind_rose_panel content with 4 roses covering:
#       now→+120, +120→+240, +240→+360, +360→+480 minutes.
#     """
#     global wind_rose_panel
#     roses = []
#     for k in range(4):
#         w_start = now_minute + k * decision_interval
#         w_end   = w_start   + decision_interval
#         tbl = wind_rose_from_forecast(forecast_df, w_start, w_end)
#         roses.append(make_wind_rose(
#             tbl,
#             title=f"{w_start}-{w_end} min",
#             show_legend=(k == 3)          # legend only on the last plot
#         ))
#     wind_rose_panel.objects = [           # wipe & refill
#         pn.pane.Markdown(
#             "### <span style='color:white'>Wind Forecasts (next 8 h)</span>",
#             sizing_mode="stretch_width"),
#         pn.Row(*roses, sizing_mode="stretch_width"),
#     ]
# def update_wind_rose_panel(forecast_df, now_minute):
#     global wind_rose_panel
#     plots = []
#     for k in range(4):
#         w_start = now_minute + k * decision_interval
#         w_end   = w_start   + decision_interval
#         plots.append(probability_wind_rose(forecast_df, w_start, w_end))
#
#     wind_rose_panel.objects = [
#         pn.pane.Markdown(
#             "### <span style='color:white'>Wind-draw probability (next 8 h)</span>",
#             sizing_mode="stretch_width"),
#         pn.Row(*plots, sizing_mode="stretch_width"),
#     ]
def update_wind_rose_panel(forecast_df, now_minute,
                           *, n_roses: int = NUM_WIND_ROSES) -> None:
    """
    Build n_roses probability wind-roses starting at *now_minute* and
    replace the contents of the global ``wind_rose_panel``.  Every rose
    shows a full 0–360 deg compass and its own legend.
    """
    global wind_rose_panel

    # ── make the individual plots ──────────────────────────────
    plots = []
    for k in range(n_roses):
        w_start = now_minute + k * decision_interval
        w_end   = w_start   + decision_interval
        pane = probability_wind_rose(forecast_df, w_start, w_end)

        # ensure the legend is ON and the compass is complete
        if isinstance(pane.object, go.Figure):
            pane.object.update_layout(
                showlegend=True,
                polar=dict(
                    angularaxis=dict(
                        direction="clockwise",
                        rotation=90,
                        tickmode="array",
                        tickvals=np.arange(0, 360, 45),   # 0-315 every 45°
                        ticks="outside",
                        showline=True,
                        linewidth=1,
                    ),
                    radialaxis=dict(showticklabels=True, ticksuffix="%"),
                ),
            )
        plots.append(pane)

    # ── lay them out in rows of ROSES_PER_ROW ──────────────────
    rows = [pn.Row(*plots[i:i+ROSES_PER_ROW], sizing_mode="stretch_width")
            for i in range(0, len(plots), ROSES_PER_ROW)]

    wind_rose_panel.objects = [
        pn.pane.Markdown(
            f"### <span style='color:white'>Wind-draw probability "
            f"(next {n_roses * decision_interval // 60} h)</span>",
            sizing_mode="stretch_width"),
        *rows,
    ]
# ────────────────────────────────────────────────────────────────
# Default (placeholder) parameters – will be overwritten by widgets
# ────────────────────────────────────────────────────────────────
lake_positions  = [(5000, 5000)]
# base_positions  = [( 20000,  20000),
#                    (-20000,  20000),
#                    (-20000, -20000),
#                    ( 20000, -20000)]


base_positions  = [( 20000,  20000)]
start_time      = datetime.datetime.strptime("00:00", "%H:%M")
case_folder     = "Dashboard_Case"
groundcrew_sector_mapping        = [0]
second_groundcrew_sector_mapping = None

# ────────────────────────────────────────────────────────────────
# MCTS defaults (also controlled by widgets)
# ────────────────────────────────────────────────────────────────
decision_interval = 120   # minutes between replanning
mcts_iterations    = 50
mcts_max_depth     = 3
exploration_constant = 1 / math.sqrt(2)

# ────────────────────────────────────────────────────────────────
# Input widgets – Simulation first
# ────────────────────────────────────────────────────────────────
elapsed_minutes_widget     = pn.widgets.IntInput  (name='Elapsed Minutes',          value=240,  start=0)
fire_sim_time_widget       = pn.widgets.IntInput  (name='Fire Spread Sim Time',     value=2000, start=1)
overall_time_limit_widget  = pn.widgets.IntInput  (name='Overall Time Limit',       value=2000, start=1)
groundcrew_count_widget    = pn.widgets.IntInput  (name='Ground-crew Count',        value=0,     start=0)
num_sectors_widget = pn.widgets.IntInput(name='Sectors', value=4, start=1)

c130j_widget   = pn.widgets.IntInput(name='C130J Count',        value=0, start=0)
fireherc_widget= pn.widgets.IntInput(name='FireHerc Count',     value=1, start=0)
scooper_widget = pn.widgets.IntInput(name='Scooper Count',      value=0, start=0)
at802f_widget  = pn.widgets.IntInput(name='AT-802F Count',      value=0, start=0)
dash8_widget   = pn.widgets.IntInput(name='Dash-8 400MRE Count',value=0, start=0)

time_step_widget         = pn.widgets.IntInput(name='Time Step (min)',      value=1,     start=1)
operational_delay_widget = pn.widgets.IntInput(name='Operational Delay',    value=0,     start=0)
groundcrew_speed_widget  = pn.widgets.IntInput(name='Ground-crew Speed',    value=30,    start=1)
wind_speed_widget        = pn.widgets.IntSlider(name='Wind Speed',          start=0, end=50, step=1, value=0)
wind_dir_widget          = pn.widgets.IntSlider(name='Wind Direction (°)',  start=0, end=360, step=1, value=220)

sim_inputs_panel = pn.Column(
    pn.pane.Markdown("### Simulation Inputs"),
    fire_sim_time_widget,
    overall_time_limit_widget,
    elapsed_minutes_widget,
    groundcrew_count_widget,
    fireherc_widget,                # keeping only one airtanker type for brevity
    groundcrew_speed_widget,
    wind_speed_widget,
    wind_dir_widget,
    num_sectors_widget,
    sizing_mode='stretch_width'
)

# ────────────────────────────────────────────────────────────────
# MCTS widgets
# ────────────────────────────────────────────────────────────────
decision_interval_widget   = pn.widgets.IntInput  (name='Decision Interval (min)', value=decision_interval, start=1)
mcts_iter_widget           = pn.widgets.IntInput  (name='MCTS Iterations',         value=mcts_iterations,   start=1)
mcts_depth_widget          = pn.widgets.IntInput  (name='MCTS Max Depth',          value=mcts_max_depth,    start=1)
exploration_constant_widget= pn.widgets.FloatInput(name='Exploration Constant (UCB)', value=exploration_constant, step=0.1, start=0)
# ── NEW widgets ───────────────────────────────────────────────
auto_iter_toggle = pn.widgets.Checkbox(name='Auto-set MCTS iterations')
iters_for_1_asset = pn.widgets.IntInput(name='Iters if 1 asset', value=50,  start=1)
iters_for_2_assets = pn.widgets.IntInput(name='Iters if 2 assets', value=100, start=1)
iters_for_3_assets = pn.widgets.IntInput(name='Iters if 3 assets', value=200, start=1)

# mcts_inputs_panel = pn.Column(
#     pn.pane.Markdown("### MCTS Inputs"),
#     decision_interval_widget,
#     mcts_iter_widget,
#     mcts_depth_widget,
#     exploration_constant_widget,
#     sizing_mode='stretch_width'
# )

mcts_inputs_panel = pn.Column(
    pn.pane.Markdown("### MCTS Inputs"),
    decision_interval_widget,
    mcts_iter_widget,            # ← kept so you can still type manually
    auto_iter_toggle,            # ← NEW
    iters_for_1_asset,
    iters_for_2_assets,
    iters_for_3_assets,
    mcts_depth_widget,
    exploration_constant_widget,
    sizing_mode='stretch_width'
)

# ────────────────────────────────────────────────────────────────
# Buttons
# ────────────────────────────────────────────────────────────────
apply_button = pn.widgets.Button(name='Set Simulation Parameters',
                                 button_type='primary', width=180)
run_button   = pn.widgets.Button(name='Run',
                                 button_type='success', width=100)
pause_button = pn.widgets.Button(name='Pause',
                                 button_type='warning', width=100)

# ────────────────────────────────────────────────────────────────
# Panels that will be updated live
# ────────────────────────────────────────────────────────────────
empty_fig    = go.Figure()
empty_fig.update_layout(template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)')
map_panel          = pn.pane.Plotly(empty_fig, width=800, height=600)
tree_panel         = pn.pane.Plotly(empty_fig, height=400,
                                    sizing_mode='stretch_width')
progress           = pn.widgets.Progress(name="Sim Progress",
                                         value=0, max=100, width=300)

mcts_history_panel = pn.Row(sizing_mode='stretch_width',
                            styles={'overflow-x': 'auto'})
history_entries: list[tuple[pn.widgets.Button, go.Figure]] = []

# wind_rose_panel = pn.Column(
#     pn.pane.Markdown("### <span style='color:white'>Wind Forecasts</span>",
#                      sizing_mode='stretch_width'),
#     pn.Row(make_wind_rose(1, show_legend=False),
#            make_wind_rose(2, show_legend=False),
#            make_wind_rose(3, show_legend=True),
#            sizing_mode='stretch_width'),
#     sizing_mode='stretch_width'
# )
wind_rose_panel = pn.Column()   # will be filled by update_wind_rose_panel()

# ────────────────────────────────────────────────────────────────
# Functions to build/reset the model
# ────────────────────────────────────────────────────────────────
def build_model_from_widgets() -> WildfireModel:
    """
    Construct a brand-new WildfireModel using *current* widget values.
    """
    return WildfireModel(
        airtanker_counts={
            "C130J":        c130j_widget.value,
            "FireHerc":     fireherc_widget.value,
            "Scooper":      scooper_widget.value,
            "AT802F":       at802f_widget.value,
            "Dash8_400MRE": dash8_widget.value,
        },
        wind_speed           = wind_speed_widget.value,
        wind_direction       = wind_dir_widget.value,
        base_positions       = base_positions,
        lake_positions       = lake_positions,
        time_step            = time_step_widget.value,
        debug                = False,
        start_time           = start_time,
        case_folder          = case_folder,
        overall_time_limit   = overall_time_limit_widget.value,
        fire_spread_sim_time = fire_sim_time_widget.value,
        operational_delay    = operational_delay_widget.value,
        enable_plotting      = True,
        groundcrew_count     = groundcrew_count_widget.value,
        groundcrew_speed     = groundcrew_speed_widget.value,
        elapsed_minutes      = elapsed_minutes_widget.value,
        groundcrew_sector_mapping        = groundcrew_sector_mapping,
        second_groundcrew_sector_mapping = second_groundcrew_sector_mapping,
        # wind_schedule = None,
        wind_schedule = truth_schedule,
        fuel_model_override=None,
        num_sectors=num_sectors_widget.value,
    )

# ────────────────────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────────────────────
def count_nodes(node):
    """Recursively count descendants (including root)."""
    return 1 + sum(count_nodes(c) for c in getattr(node, "children", []))

def on_mcts_iter(root, iteration):
    """Live update of the Plotly tree during MCTS search."""
    tree_panel.object = plot_mcts_tree_plotly(root)

def plot_mcts_tree_plotly(root):
    """Return a Plotly figure of the current MCTS tree."""
    G = nx.DiGraph()

    # build NetworkX graph with node statistics
    def build(n):
        G.add_node(id(n),
                   visits=n.visits,
                   avg_reward=(n.reward/n.visits) if n.visits else 0.0,
                   label=n.node_name)
        for child in n.children:
            G.add_edge(id(n), id(child))
            build(child)
    build(root)

    pos = hierarchy_pos_tree(G, id(root))  # your custom layout

    # edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(color='gray', width=1),
                            hoverinfo='none')

    # nodes (size ∝ visits, color ∝ avg reward)
    node_x, node_y, node_size, node_color, hover_text = [], [], [], [], []
    for n_id, data in G.nodes(data=True):
        x, y = pos[n_id]
        node_x.append(x)
        node_y.append(y)
        visits = data['visits']
        avg_rwd = data['avg_reward']
        # node_size.append(10 + visits * 2)
        node_size.append(10 + 0.5* ( visits * 2))

        node_color.append(avg_rwd)
        hover_text.append(
            f"{data['label']}<br>Visits: {visits}<br>Avg Rwd: {avg_rwd:.2f}"
        )
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(size=node_size,
                    color=node_color,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Avg Reward')),
        text=hover_text, hoverinfo='text'
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(template='plotly_dark',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      title="MCTS Tree  (size ∝ visits, color ∝ avg reward)",
                      xaxis=dict(showgrid=False, zeroline=False,
                                 showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False,
                                 showticklabels=False),
                      margin=dict(l=0, r=0, t=30, b=0))
    return fig


# ★ NEW ------------------------------------------------------------------
def end_simulation(model: WildfireModel):
    """
    Final-ise the run once no more decisions can be made.
    • flush any buffered ground-crew work
    • calculate & print the final fire-score
    • capture the last frame (only if plotting on)
    """
    from groundcrew import GroundCrewAgent

    # 1) commit pending clear-buffers so score is correct
    for ag in model.schedule.agents:
        if isinstance(ag, GroundCrewAgent):
            ag.flush_clear_buffer()

    # # 2) final score
    # score = model.fire.calculate_fire_score(model.time)
    # print(f"[SIM] Finished at t={model.time:.0f} min – final fire-score = {score:.2f}")

    import json
    # final metrics
    area_tot = model.fire.calculate_fire_score(model.time)
    try:
        bldg_tot = model.fire.calculate_building_score(model.time)
        bldg_map = model.fire.calculate_building_breakdown(model.time)
    except AttributeError:  # tif had <10 bands
        bldg_tot, bldg_map = 0, {}

    print(
        f"[SIM] Finished at t={model.time:.0f} min – "
        f"final fire-score = {area_tot:.2f}  "
        f"final buildings-destroyed = {bldg_tot}  "
        f"final buildings-breakdown = {json.dumps(bldg_map, separators=(',', ':'))}"
    )

    # 3) last PNG
    if model.enable_plotting:
        model.plot_fire()
        out = Path(model.case_folder) / "images"
        out.mkdir(exist_ok=True)
        fname = out / f"final_{int(model.time)}.png"
        model.plot_fig.write_image(str(fname), scale=4)
        print(f"[SIM] Saved final frame → {fname}")
# ------------------------------------------------------------------------


# ────────────────────────────────────────────────────────────────
# Core simulation loop (runs in a background thread)
# ────────────────────────────────────────────────────────────────
def simulation_loop(model: WildfireModel):

    schedule_enabled = model.wind_schedule is not None

    """
    Background loop that drives the truth simulation and runs MCTS
    whenever the clock reaches the next_decision_time.
    ─────────────────────────────────────────────────────────────
    Fixes:
      • first tree now starts immediately (t = model.time)
      • every tree is built exactly on the decision boundary, not +1 min
    """
    next_decision_time = model.time          # ← start planning right away
    overall_limit      = model.overall_time_limit

    # ⇩ NEW FORECAST ----------------------------------------------------
    if schedule_enabled:
        forecast_df = get_forecast(current_minute=int(model.time))
        model.latest_forecast_df = forecast_df
        record_forecast(int(model.time), forecast_df)
        update_forecast_evo_panel()
        # update_wind_rose_panel(forecast_df, int(model.time))
        update_fan_plot_panel(forecast_df, int(model.time))
    else:
        forecast_df = None
    # -------------------------------------------------------------------

    while not simulation_stop_event.is_set():

        # honour dashboard pause
        while pause_event.is_set() and not simulation_stop_event.is_set():
            t.sleep(0.1)
        if simulation_stop_event.is_set():
            break

        # ────────────────────────────────────────────────────────────
        # NEW ► If we don’t have a *full* decision slice left, jump
        #       straight to the end (or break the loop).
        #       This prevents launching an MCTS tree that can’t expand.
        # ----------------------------------------------------------------
        # if model.time >= overall_limit - decision_interval:  # ← NEW
        print("MCTS expansion time ", model.time)
        print("overall_limit ", overall_limit)
        # if model.time >= overall_limit - decision_interval:  # ← NEW
        #     print("[SIM] Not enough time for another decision slice; "
        #           "fast-forwarding to the end.")  # ← NEW
        #     while model.time < overall_limit and model.step():  # ← NEW
        #         pass  # ← NEW
        #     break  # ← NEW

        # ────────────────────────────────────────────────────────────
        # 0 · leave loop if there is *no* full decision slice left
        # ────────────────────────────────────────────────────────────
        if model.time >= overall_limit - decision_interval:  # unchanged
            print("[SIM] < 1 decision slice left – terminating.")  # ★ NEW
            end_simulation(model)  # ★ NEW
            break  # ★ NEW
        # ────────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────────

        # ─── 1 · (re)plan BEFORE advancing one tick ───────────────────────
        if model.time >= next_decision_time:
            # ⇩ update forecast for this decision boundary
            # forecast_df = get_forecast(current_minute=int(model.time))
            # model.latest_forecast_df = forecast_df
            # record_forecast(int(model.time), forecast_df)  # NEW
            # update_forecast_evo_panel()
            # update_wind_rose_panel(forecast_df, int(model.time))

            if schedule_enabled:
                forecast_df = get_forecast(current_minute=int(model.time))
                model.latest_forecast_df = forecast_df
                record_forecast(int(model.time), forecast_df)
                update_forecast_evo_panel()
                # update_wind_rose_panel(forecast_df, int(model.time))
                update_fan_plot_panel(forecast_df, int(model.time))

                # ⇩ DEBUG: μ/σ table that MCTS will sample from
                _debug_dump_forecast(forecast_df, now=int(model.time),
                                     hdr="⚑ forecast going into MCTS")

            # ⇩ DEBUG: show the exact schedule that MCTS will see
            _debug_dump_schedule(model, header="⚑ deterministic schedule going into MCTS")


            # --- EARLY-EXIT: nothing left to contain -------------------------
            open_secs = [s for s in range(model.num_sectors) if not model.is_sector_contained(s)]

            if not open_secs:
                print("[SIM] All sectors contained – ending simulation.")
                end_simulation(model)
                break  # leave while-loop
            # ----------------------------------------------------------------


            clone = clone_simulation(model)
            if schedule_enabled:
                clone.latest_forecast_df = forecast_df

            cp.cuda.Stream.null.synchronize()
            t0 = t.time()

            # ── decide how many iterations to run ────────────────────────
            if auto_iter_toggle.value:
                n_assets = count_expandable_assets(model)
                mapping = {
                    1: iters_for_1_asset.value,
                    2: iters_for_2_assets.value,
                    3: iters_for_3_assets.value,
                }
                mcts_iterations_effective = mapping.get(n_assets, mcts_iter_widget.value)
            else:
                mcts_iterations_effective = mcts_iter_widget.value

            root, _ = mcts(
                clone,
                iterations=mcts_iterations_effective,
                max_depth=mcts_max_depth,
                duration=decision_interval,
                exploration_constant=exploration_constant,
                building_weight=1,
                rollout_depth_adjustment=6,
                track_progress=False,

                on_iteration=on_mcts_iter,
            )

            # root, _ = mcts(
            #     clone,
            #     iterations=mcts_iterations,
            #     max_depth=mcts_max_depth,
            #     duration=decision_interval,
            #     exploration_constant=exploration_constant,
            #     track_progress=False,
            #     on_iteration=on_mcts_iter,
            # )
            cp.cuda.Stream.null.synchronize()
            mcts_duration = t.time() - t0


            # --- GUARD: tree has no children ---------------------------------
            if not root.children:  # ← nothing was expandable
                print("[SIM] MCTS produced no children – skipping ahead.")
                next_decision_time += decision_interval
                continue  # back to main while-loop
            # -----------------------------------------------------------------

            # gather stats
            total_nodes  = count_nodes(root)
            best_child   = max(root.children,
                               key=lambda c: (c.reward/c.visits)
                               if c.visits else -float('inf'))
            avg_best     = (best_child.reward / best_child.visits
                            if best_child.visits else 0.0)
            rewards      = [(c.reward/c.visits) if c.visits else 0
                            for c in root.children]

            # update UI
            progress.value = min(100, int((model.time/overall_limit)*100))
            stats_md = pn.pane.Markdown(f"""
            **Sim Time:** {model.time:.0f} min  
            **Nodes:** {total_nodes}  
            **Iterations:** {mcts_iterations}  
            **Depth:** {mcts_max_depth}  
            **Best Action:** `{best_child.node_name}`  
            **Avg Reward:** {avg_best:.2f}  
            **Best Rwd:** {max(rewards):.2f}  
            **Worst Rwd:** {min(rewards):.2f}  
            **Time Taken:** {mcts_duration:.2f} s""",
            width=250)

            current_fig     = plot_mcts_tree_plotly(root)
            tree_panel.object = current_fig

            # # ─── NEW: export the figure to PNG (uses Kaleido) ──────────────
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # img_name = f"mcts_tree_t{int(model.time)}m_{timestamp}.png"
            # current_fig.write_image(img_name, scale=2)  # 2× pixel density

            #   → file is saved alongside truth_dashboard.py; adjust path if needed
            # clickable history entry
            idx = len(history_entries) + 1
            btn = pn.widgets.Button(
                name=f"#{idx} | t={model.time:.0f} min | best={avg_best:.2f}",
                button_type="primary", width=280)
            btn.on_click(lambda _evt, fig=current_fig: setattr(tree_panel, "object", fig))
            mcts_history_panel.append(pn.Column(btn, stats_md, width=300))
            history_entries.append((btn, current_fig))


            # TODO
            #NEED TO REMOVE
            # ---------------------------------------------------------------
            # 1)  (re)plan fresh start/finish for every sector that is NOT yet
            #     contained – the planner returns a dict {sec_idx -> {...}}
            #
            # open_secs = [s for s in range(4)  # 0–3 sectors
            #              if not model.is_sector_contained(s)]
            # new_plan = model.groundcrew_planner.get_planned_cells(open_secs)
            #
            # # 2)  translate that into the sector-action dict the crew logic expects
            # #     {'GroundCrewAgent': (sec+1, sec+1, …),   # 1-based !
            # #      'FireHerc'       : (... airtankers ...)}
            # #
            # crew_actions = tuple(sec + 1 for sec in open_secs)  # 1-based
            # best_child.action.setdefault("GroundCrewAgent", crew_actions)

            # apply action to the real model
            simulate_in_place(model, best_child.action, duration=decision_interval)
            map_panel.object = model.plot_fig

            next_decision_time += decision_interval
            continue              # ← skip the extra one‑minute tick this loop

        # ─── 2 · no planning due → advance one ordinary tick ──────────────
        if not model.step():      # returns False on termination
            break

    progress.value = 100


# ────────────────────────────────────────────────────────────────
# Button callbacks
# ────────────────────────────────────────────────────────────────
def apply_params(event):
    """
    Build a fresh model from widget values (only allowed
    when no simulation thread is running), then DISABLE
    this button so parameters stay frozen.
    """
    global truth_model, sim_thread

    # if a simulation is already running, insist on pausing first
    if sim_thread and sim_thread.is_alive():
        pn.state.notifications.warning("Pause the simulation first.", duration=4000)
        return

    # clear any previous thread completely
    simulation_stop_event.set()
    if sim_thread and sim_thread.is_alive():
        sim_thread.join()
    simulation_stop_event.clear()

    # build new model
    truth_model = build_model_from_widgets()
    truth_model.latest_forecast_df = None  # disable forecasts

    # ‹NEW› Compute the baseline fire score for normalization.
    # If a wind_schedule is provided (forecast enabled), compute a baseline using the mean-only forecast.
    if truth_model.wind_schedule is not None:
        forecast_df = get_forecast(current_minute=0)
        baseline_sched = _make_baseline_schedule(truth_model, forecast_df)
        _print_schedule_tuples(baseline_sched,
                               title="⚑ Baseline schedule fed to surrogate")
        baseline_model = clone_simulation(truth_model,
                                          new_wind_schedule=baseline_sched)
        truth_model.baseline_fire_score = baseline_model.fire.calculate_fire_score(
            baseline_model.fire_spread_sim_time)
        print("Baseline FIRE SCORE (forecast-based):", truth_model.baseline_fire_score)
    else:
        truth_model.baseline_fire_score = truth_model.fire.calculate_fire_score(truth_model.fire_spread_sim_time)
        print("Baseline FIRE SCORE (truth-based):", truth_model.baseline_fire_score)


    maxfirescore = truth_model.fire.calculate_fire_score(truth_model.fire_spread_sim_time)
    print("MAX FIRE SCORE:", maxfirescore)
    #This is the firescore you get using the actual truth schedule
    truth_model.groundcrew_planner = None
    truth_model.plot_fire()

    # reset UI
    map_panel.object          = truth_model.plot_fig
    mcts_history_panel.objects= []
    history_entries.clear()
    tree_panel.object         = empty_fig
    progress.value            = 0

    # freeze parameters
    apply_button.disabled = True
    pn.state.notifications.success("Parameters locked in – ready to run.")


def run_sim(event):
    """
    Start the background thread if not already running,
    or simply resume after a pause.
    """
    global sim_thread, decision_interval, mcts_iterations
    global mcts_max_depth, exploration_constant

    if truth_model is None:
        pn.state.notify("Click “Set Simulation Parameters” first.",
                        duration=4000)
        return

    # pull latest MCTS widget values every time ‘Run’ is pressed
    decision_interval    = decision_interval_widget.value
    mcts_iterations      = mcts_iter_widget.value
    mcts_max_depth       = mcts_depth_widget.value
    exploration_constant = exploration_constant_widget.value

    # if no thread yet, start one
    if not (sim_thread and sim_thread.is_alive()):
        simulation_stop_event.clear()
        sim_thread = threading.Thread(
            target=simulation_loop,
            args=(truth_model,),
            daemon=True
        )
        sim_thread.start()

    # resume if we were paused
    pause_event.clear()

def pause_sim(event):
    """Soft pause; thread keeps running but does no work."""
    pause_event.set()

# connect buttons
apply_button.on_click(apply_params)
run_button.on_click(run_sim)
pause_button.on_click(pause_sim)

# ────────────────────────────────────────────────────────────────
# Dashboard layout
# ────────────────────────────────────────────────────────────────
controls_row = pn.Row(apply_button, run_button, pause_button)

dashboard = pn.Row(
    pn.Column(
        map_panel,
        pn.Row(sim_inputs_panel,
               pn.Column(mcts_inputs_panel,
                         controls_row)),
        width=800
    ),
    pn.Column(
        progress,
        mcts_history_panel,
        tree_panel,
        wind_rose_panel,
        # forecast_evo_panel,
        sizing_mode='stretch_width'
    ),
)

# ────────────────────────────────────────────────────────────────
# Serve the app
# ────────────────────────────────────────────────────────────────
if __name__.startswith("__main__"):
    pn.serve(dashboard, show=True)
