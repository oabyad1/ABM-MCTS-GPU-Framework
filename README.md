# ABM-MCTS-GPU-Framework

**ABM-MCTS-GPU-Framework** is a research-oriented software framework that combines Agent-Based Modeling (ABM), Monte Carlo Tree Search (MCTS), and GPU acceleration using CUDA via CuPy. It simulates wildfire spread and suppression using aerial and ground firefighting agents. The core fire spread model runs on the GPU, enabling efficient simulations required for decision-making under uncertainty.

---

## 🔥 Framework Overview

### Wildfire Spread Simulation (GPU Surrogate Model)
- Implements a Rothermel-based fire spread model in CUDA via CuPy.
- Simulates fire spread across a raster landscape using real topographic and fuel data.
- The `SurrogateFireModelROS` class manages GPU computation.

### Agent-Based Firefighting Model
- Built with the Mesa ABM library.
- Agents include airtankers and ground crews.
- Agents can move, drop retardant, or build fire lines.

### Monte Carlo Tree Search (MCTS) Planner
- Explores action sequences to minimize burned area.
- Performs fast forward simulations using GPU fire model.
- Builds and traverses a search tree to select optimal agent actions.

### Wind Schedules and Uncertainty
- Wind conditions are loaded from CSV schedules.
- Each benchmark uses two wind schedules: low uncertainty and high uncertainty.
- Supports stochastic wind forecasts via sampling.

---

## 📁 Benchmark Scenarios
Three real wildfire landscapes were used:

| Fire Name       | GeoTIFF File                     | Description                          |
|----------------|----------------------------------|--------------------------------------|
| Marshall Fire  | `!marshall_enhanced.tif`         | Colorado fire, rapid urban spread.   |
| Camp Fire      | `!camp_fire_three_enhanced.tif`  | California fire, high damage event.  |
| Esperanza Fire | `!cali_test_big_enhanced.tif`    | Complex terrain and wind dynamics.   |

Each scenario is run with two wind settings:
- **Medium Uncertainty** (low variability forecast)
- **High Uncertainty** (greater wind prediction error)

---

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support

### 1. Clone the Repository
```bash
git clone https://github.com/oabyad1/ABM-MCTS-GPU-Framework.git
cd ABM-MCTS-GPU-Framework
```

### 2. Install CUDA Toolkit
- Download from: https://developer.nvidia.com/cuda-downloads
- Ensure drivers are updated.

### 3. Set Up Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Install CuPy (based on your CUDA version)
Examples:
```bash
pip install cupy-cuda118  # for CUDA 11.8
pip install cupy-cuda115  # for CUDA 11.5
```

---

## 🚀 Running Simulations

### Option 1: Interactive Dashboard
```bash
python dashboard.py
```
- Web-based GUI to set landscape, wind, and agent parameters.
- Starts/stops MCTS-controlled simulation.

### Option 2: Batch Experiments
```bash
python run_all_strategies_batch.py --schedule wind_schedule_esper_high.csv --runs 10
```
- Runs multiple strategies over chosen wind schedule.
- Outputs results to CSVs and images.

### Option 3: Manual Scripts
Use predefined scripts like:
```bash
python ics_dynamic_NO_PLOTTING.py
python random_allocations_NO_PLOTTING.py
```

---

## 📊 Outputs and Visualization
- **Fire Spread Maps**: PNGs showing final burn perimeters.
- **MCTS Tree Diagrams**: Visualizes decision paths and rewards.
- **Logs**: JSON files tracking rollout timings and fire metrics.
- **CSV Summaries**: Run-by-run statistics (area burned, drops used).

---

## 🧪 Project Structure

| File / Folder                     | Description                                        |
|----------------------------------|----------------------------------------------------|
| `FIRE_MODEL_CUDA.py`             | GPU fire spread logic (CuPy-based).                |
| `model.py`                       | Main Mesa wildfire environment.                   |
| `mcts.py`                        | MCTS planner implementation.                      |
| `airtankeragent.py`              | Airtanker agent behavior.                         |
| `groundcrew.py`                  | Ground crew agent logic.                          |
| `dashboard.py`                   | Interactive web GUI with Panel.                   |
| `run_all_strategies_batch.py`   | Batch simulation runner.                          |
| `wind_schedule_utils.py`        | Helpers for handling wind forecasts.              |
| `uncertainty_quantification.py` | Tool for uncertainty analysis with visual plots.  |

---

## 📈 Research Usage
- Framework supports statistical testing under uncertainty.
- Designed for adaptive decision-making under dynamic conditions.
- Validated against real wildfire scenarios.

---

## 🙌 Contributing
Feel free to fork and extend the framework for other disasters, planning models, or improved fire simulations. Please cite relevant research if publishing.

---

## 📧 Contact
For questions or collaboration, reach out via GitHub Issues or contact the project maintainers.

---

**License**: MIT (or as specified in the repository)

