# Build System Documentation

Automated execution system for 9 Jupyter notebooks with dependency management and metric extraction.

## Quick Start

```bash
# Linux / WSL Terminal
make -f Build/Makefile.simple run
```
Example Commmand: wsl bash -c "cd /mnt/c/Users/Matt/Desktop/CS506/CS506_Project && make -f Build/Makefile.simple run"

**Execution time**: 30-60 minutes for 9 notebooks  
**Output**: Results logged to `Build/model_results.log`
---

## How It Works

### Makefile.simple
Simplified build orchestration for WSL/Linux:
- Auto-detects `PROJECT_ROOT` from Makefile location
- Executes 9 notebooks via papermill (3600s timeout each)
- Exports `PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/Build"`
- Outputs to `/tmp/` then copies back (prevents file locks)
- Sequential execution with error tracking
- Automatic dependency checking (pandas, numpy, sklearn, xgboost, papermill)
- Auto-creates `.venv_wsl` if missing

**Available Targets**:
```bash
make -f Build/Makefile.simple run              # Execute all notebooks
make -f Build/Makefile.simple all              # Same as run
make -f Build/Makefile.simple clean            # Remove temp files
```

### run_build.bat
Windows wrapper that:
1. Detects script location using `%~dp0` (no hardcoded paths)
2. Converts Windows path to WSL: `C:\Users\...` → `/mnt/c/Users/...`
3. Calls: `make -f Build/Makefile.simple run`

### path_utils.py
Cross-platform path resolver for notebooks:
```python
from pathlib import Path
project_root = Path.cwd()  # Papermill sets cwd to project root
data_path = project_root / "1_LIB" / "master" / "master.parquet"
```

### extract_results.py
Parses executed notebooks (JSON format), extracts:
- Cell execution counts
- Metrics: MSE, RMSE, R², MAE (via regex)
- Appends to `Build/model_results.log` with timestamp

---

## Dependency Management

**Automatic setup** (handled by Makefile.simple):
- Checks for `.venv_wsl` virtual environment
- Creates it if missing
- Verifies required packages (pandas, numpy, sklearn, xgboost, papermill)
- Installs from `Dependencies/requirements.txt` if needed

**Manual setup** (if needed):
```bash
python3 -m venv .venv_wsl
source .venv_wsl/bin/activate
pip install -r Dependencies/requirements.txt
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
python3 -m venv .venv_wsl
source .venv_wsl/bin/activate
pip install -r Dependencies/requirements.txt
make -f Build/Makefile.simple run  # Linux/WSL
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails immediately | Ensure `.venv_wsl` exists with dependencies installed |
| Notebooks abort | Don't run commands in same terminal during build |
| Import errors | Check `PYTHONPATH` exported in Makefile |
| File locks | Close notebooks in VS Code before build |
| Dependency errors | `pip install -r Dependencies/requirements.txt` |

---

## Advanced Usage

```bash
# Run all notebooks
make -f Build/Makefile.simple run

# Clean temp files
make -f Build/Makefile.simple clean

# View results
cat Build/model_results.log
tail -50 Build/model_results.log

# Diagnostics
python3 Build/check_nb_status.py
python3 Build/check_papermill_output.py
```

---

## Performance

- **Total time**: 30-60 minutes (9 notebooks)
- **Memory**: 300-500 MB per notebook
- **Data**: master.parquet (38 MB file, 339 MB in memory)
- **Execution**: Sequential (one notebook at a time, 3600s timeout per notebook)

---

## Adding Notebooks

1. Add to `Makefile.simple` NOTEBOOKS list
2. Update `TOTAL` count in `run` target
3. Add to `extract_results.py` notebooks list (if extracting metrics)
4. Use simple path in notebook (papermill sets cwd):
   ```python
   from pathlib import Path
   project_root = Path.cwd()
   data_path = project_root / "1_LIB" / "master" / "master.parquet"
   ```

---

---

## WSL vs Linux Commands

### Running on WSL (Windows Subsystem for Linux)

```bash
# From WSL terminal
cd /mnt/c/Users/<username>/path/to/CS506_Project
source .venv_wsl/bin/activate
make -f Build/Makefile.simple run
```

### Running on Native Linux

```bash
# From project root
cd ~/CS506_Project  # or wherever you cloned the repo
make -f Build/Makefile.simple run
```

**Key Differences**:
- **WSL**: Paths use `/mnt/c/Users/...`, filesystem bridge adds slight overhead
- **Linux**: Paths use `/home/user/...`, native filesystem (faster)
- **Makefile**: Identical for both (auto-detects environment)

---

## Support

**Troubleshooting**:
```bash
python3 Build/check_nb_status.py          # Check notebook status
cat Build/model_results.log               # View results
make -f Build/Makefile.simple clean       # Clean temp files
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