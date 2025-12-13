# Build System Documentation

Automated execution system for 9 Jupyter notebooks with dependency management and metric extraction.

## Quick Start

```bash
# Windows with WSL - VALIDATED!!!!!
Build\run_build.bat 

# Linux / WSL Terminal
make -f Build/Makefile.wsl run
```

**First-time setup**: Auto-installs dependencies from `Dependencies/requirements.txt`  
**Execution time**: 30-60 minutes for all 9 notebooks  
**Output**: Results logged to `Build/model_results.log`

---

## Files

```
Build/
├── Makefile.wsl              # Build automation (WSL/Linux)
├── run_build.bat             # Windows wrapper
├── path_utils.py             # Cross-platform paths
├── extract_results.py        # Metric extraction
├── test_notebook_execution.py# Diagnostics
├── export_svr_animations.py  # SVR animation generator
├── run_export_animations.sh  # Animation launcher (Bash)
├── run_export_animations.ps1 # Animation launcher (PowerShell)
└── model_results.log         # Results (generated)
```

---

## How It Works

### Makefile.wsl
Main build orchestration for WSL/Linux:
- Auto-detects `PROJECT_ROOT` from Makefile location
- Checks/installs dependencies (compares `requirements.txt` timestamp)
- Executes 9 notebooks via papermill (3600s timeout each)
- Exports `PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/Build"`
- Outputs to `/tmp/papermill_output/` then copies back (prevents file locks)

**Key Targets**:
```bash
make -f Build/Makefile.wsl run              # Execute all notebooks
make -f Build/Makefile.wsl run-linear       # Single notebook
make -f Build/Makefile.wsl setup            # Create venv
make -f Build/Makefile.wsl clean-outputs    # Clear outputs
make -f Build/Makefile.wsl help             # Show all targets
```

### run_build.bat
Windows wrapper that:
1. Detects script location using `%~dp0` (no hardcoded paths)
2. Converts Windows path to WSL: `C:\Users\...` → `/mnt/c/Users/...`
3. Calls WSL: `wsl bash -c "cd <wsl_path> && make -f Build/Makefile.wsl run"`

### path_utils.py
Cross-platform path resolver used in all notebooks:
```python
from path_utils import get_project_root

project_root = get_project_root()  # Auto-detects from Build/ location
data_path = project_root / "1_LIB" / "master" / "master.parquet"
```

### extract_results.py
Parses executed notebooks (JSON format), extracts:
- Cell execution counts
- Metrics: MSE, RMSE, R², MAE (via regex)
- Appends to `Build/model_results.log` with timestamp

---

## Dependency Management

Auto-installs before each build:
```bash
if requirements.txt newer than .venv_wsl/.deps_installed:
  pip install -r requirements.txt
  touch .venv_wsl/.deps_installed
```

Manual setup:
```bash
make -f Build/Makefile.wsl setup
```

---

## Reproducibility

✅ No hardcoded paths (auto-detected from script location)  
✅ Auto dependency management  
✅ Isolated `.venv_wsl` environment  
✅ Works on WSL and Linux with same commands  

**Setup on new machine**:
```bash
git clone <repo_url>
cd CS506_Project
Build\run_build.bat  # Windows + WSL
# OR
make -f Build/Makefile.wsl run  # Linux
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails immediately | `make -f Build/Makefile.wsl setup` |
| Notebooks abort | Don't run commands in same terminal during build<br>Use `run_build.bat` (separate window) |
| Import errors | Check `PYTHONPATH` exported in Makefile |
| File locks | Close notebooks in VS Code before build |
| Dependency errors | `rm .venv_wsl/.deps_installed && make -f Build/Makefile.wsl run` |

---

## Advanced Usage

```bash
# Individual notebooks
make -f Build/Makefile.wsl run-linear
make -f Build/Makefile.wsl run-svm-daily

# Clear outputs and rerun
make -f Build/Makefile.wsl clean-outputs run

# View results
cat Build/model_results.log
tail -50 Build/model_results.log

# Diagnostics
python3 Build/test_notebook_execution.py
```

---

## Performance

- **Total time**: 30-60 minutes (9 notebooks)
- **Memory**: 300-500 MB per notebook
- **Data**: master.parquet (38 MB file, 339 MB in memory)
- **Execution**: Sequential (one notebook at a time)

---

## Adding Notebooks

1. Add to `Makefile.wsl` NOTEBOOKS list
2. Update `run` target (increment TOTAL, add case)
3. Add to `extract_results.py` notebooks list
4. Use `path_utils` in notebook:
   ```python
   from path_utils import get_project_root
   project_root = get_project_root()
   ```

---

---

## WSL vs Linux Commands

### Running on WSL (Windows Subsystem for Linux)

```bash
# From Windows (PowerShell/CMD)
Build\run_build.bat

# Or from WSL terminal
cd /mnt/c/Users/<username>/path/to/CS506_Project
make -f Build/Makefile.wsl run
```

### Running on Native Linux

```bash
# From project root
cd ~/CS506_Project  # or wherever you cloned the repo
make -f Build/Makefile.wsl run
```

**Key Differences**:
- **WSL**: Paths use `/mnt/c/Users/...`, filesystem bridge adds slight overhead
- **Linux**: Paths use `/home/user/...`, native filesystem (faster)
- **Makefile**: Identical for both (auto-detects environment)

---

## Support

**Troubleshooting**:
```bash
python3 Build/test_notebook_execution.py  # Run diagnostics
cat Build/model_results.log               # View results
make -f Build/Makefile.wsl help           # Show all targets
```

---

## SVR Animation Generation

Generate animated GIFs showing SVR model predictions vs actual load.

### Quick Run

**WSL/Linux:**
```bash
bash Build/run_export_animations.sh
```

**Windows PowerShell:**
```powershell
.\Build\run_export_animations.ps1
```

### Manual Run

```bash
# Activate environment first
source .venv_wsl/bin/activate
# Or use helper
source Dependencies/activate_env.sh

# Run generator
python Build/export_svr_animations.py
```

### Output

Generates 6 GIFs in `2_FIGURES/FIGURES/svr_animations/`:
- `svr_5min_animation.gif` - 5-minute resolution (0.28% MAPE)
- `svr_15min_animation.gif` - 15-minute resolution (~0.35% MAPE)
- `svr_hourly_animation.gif` - Hourly resolution (0.28% MAPE)
- `svr_hourly_trunc_animation.gif` - Truncated dataset
- `svr_daily_weather_animation.gif` - Daily + weather (~3.5% MAPE)
- `svr_daily_loadonly_animation.gif` - Daily load-only (~5.2% MAPE)

**Settings:** 5 fps, 15 seconds duration, auto-looping

---

**Developer Environment Details**
WSL2 Configuration
WSL Version: 2.6.1.0
Kernel: 6.6.87.2-1 (microsoft-standard-WSL2)
WSLg: 1.0.66 (GUI support)
Windows Version: 10.0.26200.7462
Ubuntu Distribution
Version: Ubuntu 24.04.3 LTS (Noble Numbat)
Codename: noble
Release: 24.04
Python Environment
Python Version: 3.12.3
Virtual Environment: .venv_wsl (WSL-specific venv)
Location: /mnt/c/Users/Matt/Desktop/CS506/CS506_Project/.venv_wsl
Total Packages: 141 installed packages