#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
──────────────────────────────────────────────────────────────────────────────
Runs *N* complete sweeps of every strategy script.  Features:

  • Each sweep gets its own RNG  → unique truth wind schedule
  • All strategies within one sweep share that frozen schedule
  • Logs + PNGs saved under  <OUT_ROOT>/runXX/
  • summary.csv grows IMMEDIATELY after each strategy row
  • Safe to interrupt: on next launch the script skips finished rows
    and resumes with the same seeds (perfect reproducibility)

.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, json, os, pathlib, re, runpy, shutil
import subprocess, sys, tempfile, textwrap
from typing import Any, Dict, List, Tuple
import pandas as pd, numpy as np
# from surrogate_fire_model_CK2_multi_phase import SurrogateFireModelROS_CK2Multi
from FIRE_MODEL_CUDA import SurrogateFireModelROS



DEFAULT_RUNS = 50
DEFAULT_OUT  = "results_3005_esperenza_med"


#set rollout depth for the mcts
MCTS_ROLLOUT_DEPTH_ADJUSTMENT = 7
# MCTS_ROLLOUT_DEPTH_ADJUSTMENT = 6
# MCTS_ROLLOUT_DEPTH_ADJUSTMENT = 17

#set number of sectors
DEFAULT_SECTORS = 6


# ───────────────────────────────────────────────────────────────
# Baseline score: “let it burn” under the truth schedule
# ───────────────────────────────────────────────────────────────
def _baseline_score(truth_sched,
                    full_sim_minutes: int) -> tuple[int,int,dict]:
    """
    Run fire model for selected schedule
    no airtankers, no crews, and return its fire-score at full_sim_minutes.
    Select correct landscape here
    """
    dummy_speed, dummy_dir = truth_sched[-1][2], truth_sched[-1][3]  # ignored

    fm = SurrogateFireModelROS(
        tif_path="cali_test_big_enhanced.tif",  # <<< give valid landscape
        #     model constructor expects
        sim_time=full_sim_minutes,
        wind_speed=dummy_speed,
        wind_direction_deg=dummy_dir,
        max_iter=250,
        tol=1e-3,
        wind_schedule=truth_sched
    )
    area = fm.calculate_fire_score(full_sim_minutes)
    try:
        bldg = fm.calculate_building_score(full_sim_minutes)
        bkdn = fm.calculate_building_breakdown(full_sim_minutes)
    except AttributeError:
        bldg, bkdn = 0, {}
    return area, bldg, bkdn



# ───────────────────────────────────────────────────────────────
# 1) WildfireModel parameters: Simulation parameters for the agent based mdoel
# ───────────────────────────────────────────────────────────────
SIM_PARAMS: Dict[str, Any] = {
    "airtanker_counts": dict(C130J=0, FireHerc=1, Scooper=0,
                             AT802F=0, Dash8_400MRE=0),
    "wind_speed": 0,
    "wind_direction": 220,
    "groundcrew_count": 2,
    "overall_time_limit": 3000,
    "fire_spread_sim_time": 3000,
    "elapsed_minutes": 240,
}

# ───────────────────────────────────────────────────────────────
# 2) CLI
# ───────────────────────────────────────────────────────────────

#select appropriate wind schedule
# DEF_SCHED = "wind_schedule_marshall.csv"
DEF_SCHED = "wind_schedule_esper_med.csv"
# DEF_SCHED = "wind_schedule_from_raws.csv"
# DEF_SCHED = "wind_schedule_camp.csv"

cli = argparse.ArgumentParser()
cli.add_argument("--schedule", default=DEF_SCHED,
                 help=f"wind-schedule CSV (default: {DEF_SCHED})")
cli.add_argument("--out", default=None,
                 help=f"folder for runs + summary.csv (default: {DEFAULT_OUT})")
cli.add_argument("--params", metavar="FILE",
                 help="JSON or YAML with WildfireModel kwargs")
cli.add_argument("--python", default=sys.executable,
                 help="python interpreter for child processes")
cli.add_argument("--runs", type=int, default=None,
                 help=f"number of sweeps (default: {DEFAULT_RUNS})")
cli.add_argument("--base-seed", type=int,
                 help="first RNG seed (omit → high-res time)")

cli.add_argument("--sectors", type=int, default=DEFAULT_SECTORS,
                 help=f"number of sectors (default: {DEFAULT_SECTORS})")


args = cli.parse_args()




RUN_COUNT = args.runs if args.runs is not None else DEFAULT_RUNS
OUT_ROOT  = args.out  if args.out  is not None else DEFAULT_OUT
SECTOR_CFG = {"num_sectors": int(args.sectors)}


# ───────────────────────────────────────────────────────────────
# 3) Strategy registry  (label, script)
# ───────────────────────────────────────────────────────────────
STRATEGIES: list[tuple[str, str]] = [
    ("mcts",                       "dashboard_NO_PLOTTING.py"),
    ("ics_dynamic",                "ics_dynamic_NO_PLOTTING.py"),
    ("ics_dynamic_mean",           "ics_dynamic_mean_NO_PLOTTING.py"),
    ("ics_dynamic_mean_lookahead", "ics_dynamic_mean_lookahead_NO_PLOTTING.py"),
    ("ics_ros_weighted_mean",      "ics_ros_weighted_mean_NO_PLOTTING.py"),
    ("ics_dynamic_truth_lookahead","ics_dynamic_truth_lookahead_NO_PLOTTING.py"),
    ("ics_ros_weighted",           "ics_ros_weighted_NO_PLOTTING.py"),
    ("random",                     "random_allocations_NO_PLOTTING.py"),
    ("ics_truth_burned_buildings",       "ics_truth_burned_buildings_NO_PLOTTING.py"),
    ("ics_mean_burned_buildings",        "ics_mean_burned_buildings_NO_PLOTTING.py"),
    ("ics_buildings", "ics_buildings_NO_PLOTTING.py"),
]
STRATEGIES = list(reversed(STRATEGIES))          # keep prior call order

# ───────────────────────────────────────────────────────────────
# 4) Truth-schedule sampler
# ───────────────────────────────────────────────────────────────
def _sample_truth(csv_path: pathlib.Path,
                  rng: np.random.Generator) -> list[tuple]:
    df = pd.read_csv(csv_path)
    return [(int(r.start_min), int(r.end_min),
             float(max(rng.normal(r.speed_mean, r.speed_std), 0.0)),
             float(rng.normal(r.dir_mean, r.dir_std) % 360.0))
            for r in df.itertuples()]

# ───────────────────────────────────────────────────────────────
# 5) Child-process wrapper (brace-escaped)
# ───────────────────────────────────────────────────────────────

WRAP = textwrap.dedent("""
    import json, runpy, sys, atexit, importlib, pathlib, matplotlib
    matplotlib.use("Agg", force=True)

    _TRUTH = json.loads({truth_json})
    _SIM_PARAMS = json.loads({sim_params_json})
    
    _SECT = json.loads({sector_json})
    _SECT_N = int(_SECT.get("num_sectors", 4))
    
    _SEED  = {seed}
    _LABEL = {label!r}.upper()
    _OUT   = pathlib.Path({out_dir!r}); _OUT.mkdir(parents=True, exist_ok=True)
    _SCHED_PATH = pathlib.Path({sched_path!r}).resolve()

    

    # --- import target strategy script -------------------------------------
    try:
        import dashboard as _dash                     # legacy name
    except ModuleNotFoundError:
        _dash = importlib.import_module(pathlib.Path({target!r}).stem)

    # overwrite its module-level truth_schedule (if present)
    if hasattr(_dash, "truth_schedule"):
        _dash.truth_schedule = _TRUTH
    if hasattr(_dash, "truth_model"):
        _dash.truth_model = None  # clear cached model if any
        
        
    # --- register background + truth with forecast_provider ----------------
    from forecast_provider import set_background_schedule, set_truth_schedule, set_forecast_hyperparams
    import wind_schedule_utils as _wsu
    _BG = _wsu.load_wind_schedule_from_csv_mean(str(_SCHED_PATH))
    set_background_schedule(_BG)
    set_truth_schedule(_TRUTH)
    
    
    set_forecast_hyperparams(
        speed_std_min={speed_std_min},
        speed_std_max={speed_std_max},
        dir_std_min={dir_std_min},
        dir_std_max={dir_std_max},
    )
    
    import os
    os.environ["BACKGROUND_SCHEDULE_JSON"] = json.dumps(_BG)
    os.environ["BACKGROUND_SCHEDULE_PATH"] = str(_SCHED_PATH)
    os.environ["ROLLOUT_DEPTH_ADJUSTMENT"] = str({rollout_depth_adjustment})


    # --- plotting snapshot helper (unchanged) ------------------------------
    import matplotlib.pyplot as _plt
    from surrogate_plotting import plot_fire
    _latest = None
    def _snap(tag, mod):
        fig, ax = plot_fire(mod.fire, time=mod.time,
                            max_time=mod.fire_spread_sim_time)
        ax.set_title(f"{{_LABEL}} – {{tag}}  (t={{mod.time:.0f}} min, seed={{_SEED}})")
        fig.savefig(_OUT / f"{{_LABEL.lower()}}_{{tag}}.png", dpi=300)
        _plt.close(fig)

    # patch WildfireModel.__init__ so we force the truth schedule + capture
    _mdl       = importlib.import_module("model")
    _orig_init = _mdl.WildfireModel.__init__
    def _new_init(self, *a, **k):
        global _latest
        k.update(_SIM_PARAMS)
        k["wind_schedule"] = _TRUTH
        k["case_folder"]   = _OUT
        k["num_sectors"]   = _SECT_N
        _orig_init(self, *a, **k)
        
        try:
            self._background_schedule = _BG
        except Exception:
            pass
        
        
        _latest = self; _snap("initial", self)
    _mdl.WildfireModel.__init__ = _new_init

    atexit.register(lambda: _latest and _snap("final", _latest))

    print("RUN SEED:", _SEED)
    runpy.run_path({target!r}, run_name="__main__")
""")


FINAL_RE = re.compile(r"final fire-score\s*=\s*([0-9.]+)", re.I)
FINAL_AREA_RE   = re.compile(r"final fire-score\s*=\s*([0-9.]+)", re.I)
FINAL_BLDG_RE   = re.compile(r"final buildings-destroyed\s*=\s*([0-9.]+)", re.I)

FINAL_BKDN_RE = re.compile(r"final buildings-breakdown\s*=\s*(\{.*\})", re.I)

# ───────────────────────────────────────────────────────────────
# 6) Strategy runner
# ───────────────────────────────────────────────────────────────
def run_strategy(truth_json: str, seed: int, label: str, script: str,
                 out_dir: pathlib.Path) -> tuple[float | str, float | str, str, int, int, str]:
    print("Using schedule:", sched_path)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".py",
                                      delete=False, encoding="utf-8")

    tmp.write(WRAP.format(truth_json=repr(truth_json),
                          sim_params_json=repr(json.dumps(SIM_PARAMS)),
                          label=label,
                          target=script,
                          out_dir=str(out_dir),
                          seed=seed,
                          sched_path=str(sched_path),
                          sector_json=repr(json.dumps(SECTOR_CFG)),
                          rollout_depth_adjustment=MCTS_ROLLOUT_DEPTH_ADJUSTMENT,
                          speed_std_min=SPEED_STD_MIN_VAL,
                          speed_std_max=SPEED_STD_MAX_VAL,
                          dir_std_min=DIR_STD_MIN_VAL,
                          dir_std_max=DIR_STD_MAX_VAL,
                          ))  # ← NEW

    wrapper = tmp.name; tmp.close()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{ts}_{label}.log"; log_path = out_dir / log_name
    start = dt.datetime.now()
    proc  = subprocess.Popen([args.python, "-u", wrapper],
                             cwd=pathlib.Path(__file__).parent,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", bufsize=1)
    area, bldg, bkdn = "N/A", "N/A", "{}"
    with open(log_path, "w", encoding="utf-8") as log:
        for line in proc.stdout:
            print(f"[{label}] {line}", end="");
            log.write(line)
            m1 = FINAL_AREA_RE.search(line);
            m2 = FINAL_BLDG_RE.search(line)
            m3 = FINAL_BKDN_RE.search(line)
            if m1: area = float(m1.group(1))
            if m2: bldg = float(m2.group(1))
            if m3: bkdn = m3.group(1)
    proc.wait()
    secs = int((dt.datetime.now() - start).total_seconds())
    pathlib.Path(wrapper).unlink(missing_ok=True)
    return area, bldg, bkdn, secs, proc.returncode, log_name

# ───────────────────────────────────────────────────────────────
# 7) Prepare output folder & summary
# ───────────────────────────────────────────────────────────────
sched_path = pathlib.Path(args.schedule).expanduser().resolve()

# Read schedule CSV to get first/last-bin stds
_sched_df = pd.read_csv(sched_path).sort_values("start_min")
# Use first non-null row and last non-null row for stds
_first = _sched_df[_sched_df["speed_std"].notna() & _sched_df["dir_std"].notna()].iloc[0]
_last  = _sched_df[_sched_df["speed_std"].notna() & _sched_df["dir_std"].notna()].iloc[-1]

SPEED_STD_MIN_VAL = float(_first["speed_std"])
SPEED_STD_MAX_VAL = float(_last["speed_std"])
DIR_STD_MIN_VAL   = float(_first["dir_std"])
DIR_STD_MAX_VAL   = float(_last["dir_std"])

if not sched_path.exists():
    sys.exit(f"[ERR] schedule not found: {sched_path}")

out_root = pathlib.Path(OUT_ROOT).resolve()
out_root.mkdir(parents=True, exist_ok=True)
shutil.copy2(sched_path, out_root / sched_path.name)

FIELDNAMES = (
    "run", "seed",
    "baseline_area", "baseline_buildings", "baseline_bldg_breakdown",
    "strategy", "script",
    "area_score", "buildings_destroyed", "bldg_breakdown",
    "seconds", "returncode", "log_file"
)


csv_path = out_root / "summary.csv"
if not csv_path.exists():
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, FIELDNAMES).writeheader()

# ───────────────────────────────────────────────────────────────
# 8) Resume information
# ───────────────────────────────────────────────────────────────
done: dict[tuple[int, str], bool] = {}
run_seed_map: dict[int, int] = {}       # run_id → seed already used
done_rows: list[dict] = []              # ←  keep complete rows for baseline lookup

with open(csv_path, newline="", encoding="utf-8") as fh:
    rdr = csv.DictReader(fh)
    for row in rdr:
        r  = int(row["run"])
        sd = int(row["seed"])
        done[(r, row["strategy"])] = True
        run_seed_map[r] = sd
        done_rows.append(row)

BASE_SEED = args.base_seed if args.base_seed is not None \
            else int(dt.datetime.now().timestamp() * 1e9) & 0xFFFFFFFF

# --- FORCE IDE OVERRIDES (place AFTER the args-based BASE_SEED line) ---
# RUN_COUNT = 1                 # only one run
# BASE_SEED = 3919038470        # the exact seed you want
# OUT_ROOT  = "results_rerun_run02"
# -----------------------------------------------------------------------


print(f"\nBatch configuration → runs={RUN_COUNT}  base-seed={BASE_SEED}  out={out_root}\n")

# ───────────────────────────────────────────────────────────────
# 9) Main loop
# ───────────────────────────────────────────────────────────────
for run_idx in range(RUN_COUNT):
    seed = run_seed_map.get(run_idx, BASE_SEED + run_idx) #use for regular runs
    # seed = BASE_SEED + run_idx  # always use your forced seed for RERUNS in IDE runs

    run_seed_map[run_idx] = seed

    rng = np.random.default_rng(seed)
    truth_sched = _sample_truth(sched_path, rng)
    truth_json  = json.dumps(truth_sched)

    # # 3) compute or retrieve baseline
    # if run_idx in run_seed_map and any(
    #         (run_idx, lbl) in done for lbl, _ in STRATEGIES):
    #     # already began this run earlier → baseline must be in CSV ► read once
    #     baseline = next(
    #         float(r["baseline_score"]) for r in done_rows
    #         if r["run"] == str(run_idx))  # done_rows collected when we
    #     # parsed summary.csv (see below)
    # else:
    #     baseline = _baseline_score(
    #         truth_sched,
    #         SIM_PARAMS["fire_spread_sim_time"]
    #     )
    #     print(f"[BASELINE] run {run_idx}  score = {baseline:.2f}")

    # baseline = _baseline_score(
    #     truth_sched,
    #     SIM_PARAMS["fire_spread_sim_time"]
    # )

    baseline_area, baseline_bldg, baseline_bkdn = _baseline_score(
        truth_sched, SIM_PARAMS["fire_spread_sim_time"])
    # print(f"[BASELINE] run {run_idx}  score = {baseline:.2f}")

    run_out = out_root / f"run{run_idx:02d}"
    run_out.mkdir(parents=True, exist_ok=True)
    print(f"══════════ Run {run_idx}  (seed={seed}) ══════════")

    for label, script in STRATEGIES:
        if (run_idx, label) in done:
            print(f"• {label:<25} already complete – skipping")
            continue

        #for debbugging
        strategy_out = run_out / label
        strategy_out.mkdir(parents=True, exist_ok=True)
        print(f"▶ {label:<25} ({script}) → {strategy_out}")


        print(f"▶ {label:<25} ({script})")
        # score, secs, rc, log = run_strategy(truth_json, seed,
        #                                     label, script, run_out)

        # for debbugging
        area, bldg, bkdn, secs, rc, log = run_strategy(
            truth_json, seed, label, script, strategy_out)


        status = "OK" if rc == 0 else f"ERR {rc}"
        print(f"   finished in {secs}s   area={area}   buildings={bldg}   {status}\n")

        # row = dict(run=run_idx, seed=seed, strategy=label, script=script,
        #            fire_score=score, seconds=secs, returncode=rc,
        #            log_file=str(run_out / log))

        row = dict(
            run=run_idx, seed=seed,
            baseline_area=baseline_area,
            baseline_buildings=baseline_bldg,
            baseline_bldg_breakdown=json.dumps(baseline_bkdn),
            strategy=label, script=script,
            area_score=area,
            buildings_destroyed=bldg,
            bldg_breakdown=bkdn,
            seconds=secs, returncode=rc,
            log_file=str(run_out / log)
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, FIELDNAMES).writerow(row)
        done[(run_idx, label)] = True

print("\nAll sweeps complete.")
print(" summary →", csv_path)
