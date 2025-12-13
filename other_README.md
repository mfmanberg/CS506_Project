# CS506 Project
BU CS506 Final Project - NYISO Load Forecasting

## Quick Start

**Automated Execution (Recommended):**
```powershell
.\RunNotebooks.ps1
```

This executes all 7 notebooks in WSL and automatically logs results to `model_results.log`.

**Reproducible Setup:**
All scripts use dynamic path detection - they work from any directory on any machine with WSL installed. No hardcoded paths!

---

## Model Performance Results

| Model | MSE | RMSE | R² | Notes |
|-------|-----|------|----|----- |
| Linear Regression | 84.80 | 84.80 | 0.311 | Baseline |
| SVM Truncated | 156.35 | 156.35 | 0.545 | Hourly |
| SVM Daily | 108.99 | 108.99 | 0.761 | Daily aggregation |
| SVM Daily (no weather) | 83.91 | 83.91 | **0.859** | Best R² |
| XGBoost PostMid | **64.18** | 64.18 | - | **Lowest MSE** |
| XGBoost Testing | 96.33 | 96.33 | - | Full features |

---

## Project Structure

```
├── RunNotebooks.ps1          # Main execution script
├── extract_results.py        # Metric extraction & logging
├── path_utils.py            # Cross-platform path handling
├── model_results.log        # Performance log (timestamped)
├── 1_LIB/master/            # Data (master.parquet - 38MB)
└── 3_OUTPUT/                # Analysis notebooks (7 total)
    ├── 3_linear_regression/
    ├── 3_svr/
    └── 3_xg_boost/
```

---

## Analysis Notebooks

1. **linear_regression.ipynb** - Linear regression baseline
2. **SVM_Trunc.ipynb** - SVM truncated hourly model
3. **SVMDaily.ipynb** - SVM daily predictions with weather
4. **SVMDailywoutMeso.ipynb** - SVM daily without weather data
5. **ComparisonMetrics.ipynb** - Model comparison visualizations
6. **XGBoost_PostMid.ipynb** - XGBoost post-midterm model
7. **XGBoost_Testing.ipynb** - XGBoost full feature testing

---

## Setup on Any Machine

**Prerequisites:**
- Windows with WSL (Ubuntu) installed
- Python 3.8+ in WSL

**First-time setup:**
```bash
# 1. Clone/download project to any location
cd /path/to/CS506_Project

# 2. Create WSL virtual environment
wsl bash -c "python3 -m venv .venv_wsl"

# 3. Install dependencies
wsl bash -c "source .venv_wsl/bin/activate && pip install -r requirements.txt"

# 4. Run notebooks
.\RunNotebooks.ps1
```

All paths are automatically detected - no configuration needed!

---

## Logging System

Results are automatically logged to `model_results.log` with timestamps, allowing you to track performance improvements over time.

**Manual metric extraction:**
```bash
wsl python3 extract_results.py
```

**View results:**
```bash
cat model_results.log
```

---

## Alternative Manual Execution

If you prefer manual execution or encounter WSL issues:

```bash
# Open Jupyter
python run_makefile.py open-all

# Then manually:
# 1. Navigate to 3_OUTPUT folder
# 2. Open each notebook
# 3. Cell -> Run All
# 4. Save (Ctrl+S)
```

## Commands

| Command | Description |
|---------|-------------|
| `status` | Show project status (default) |
| `open-all` | Open Jupyter in project folder |
| `open <path>` | Open specific notebook |
| `clean` | Remove master.parquet |

## Why Manual Execution?

Windows uses `ProactorEventLoop` for asyncio (Python 3.8+), but Jupyter's ZMQ communication requires `SelectorEventLoop`. This causes the kernel to crash immediately during automated execution.

**Solution**: Run notebooks interactively in Jupyter, which handles the event loop correctly.

## Project Structure

- `run_makefile.py` - Jupyter launcher
- `Makefile` - Make shortcuts
- `1_LIB/master/master.parquet` - Master dataset (38 MB)
- `3_OUTPUT/` - Analysis notebooks (7 total)

## Analysis Notebooks

1. `linear_regression.ipynb` - Linear regression analysis
2. `SVM_Trunc.ipynb` - SVM truncated model
3. `SVMDaily.ipynb` - SVM daily predictions  
4. `SVMDailywoutMeso.ipynb` - SVM without mesonet data
5. `ComparisonMetrics.ipynb` - Model comparison
6. `XGBoost_PostMid.ipynb` - XGBoost post-midterm
7. `XGBoost_Testing.ipynb` - XGBoost testing

## Workflow

```bash
# 1. Check status
python run_makefile.py status

# 2. Open Jupyter
python run_makefile.py open-all

# 3. In Jupyter browser:
#    - Navigate to 3_OUTPUT folder
#    - Open each .ipynb file
#    - Run all cells (Cell -> Run All)
#    - Save when complete
```
