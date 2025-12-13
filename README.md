# CS506 Project - Complete Setup & Usage Guide

## 🎯 Super Quick Start

**Three simple steps:**

1. **Setup (one time):**
   ```
   Double-click: setup.bat
   ```

2. **Wait** (5-10 min first time, instant after)

3. **Use** - The activated terminal opens automatically:
   ```cmd
   make status
   make run-analysis
   ```

**That's it!** Everything is automated.

---

## 📚 Table of Contents

1. [First Time Setup](#first-time-setup)
2. [Daily Workflow](#daily-workflow)
3. [Makefile Commands](#makefile-commands)
4. [Project Structure](#project-structure)
5. [Troubleshooting](#troubleshooting)

---

## 🚀 First Time Setup

### Prerequisites

- **Windows 10/11**
- **Python 3.8+** (Download: https://python.org/downloads)
  - ⚠️ During install: Check "Add Python to PATH"

### Installation

**Just run:**

*Command Prompt or Double-click:*
```cmd
setup.bat
```

*PowerShell:*
```powershell
.\setup.bat
```

This single script:
- ✅ Checks Python
- ✅ Creates virtual environment
- ✅ Installs all dependencies
- ✅ Verifies installation
- ✅ Activates environment

**No user interaction needed!**

---

## 💼 Daily Workflow

### Quick Activation (Recommended)
```cmd
setup.bat activate          # Command Prompt
.\setup.bat activate        # PowerShell
```
Opens an activated terminal - just run your commands!

### Run Make Commands Directly
```cmd
setup.bat make status              # Command Prompt
.\setup.bat make status            # PowerShell

setup.bat make run-analysis        # Command Prompt
.\setup.bat make run-analysis      # PowerShell
```

### Use the Setup Window
After running `setup.bat`, the activated terminal stays open. Just use it!

### Manual Activation
```cmd
venv\Scripts\activate.bat
make status
deactivate
```

## 🎯 setup.bat - All-in-One Tool

The `setup.bat` script handles everything:

```cmd
setup.bat                  # Full setup + activate (first time)
setup.bat                  # Verify + activate (subsequent runs)
setup.bat activate         # Just activate environment
setup.bat make [command]   # Activate and run make command
```

**Examples:**
```cmd
setup.bat make status
setup.bat make run-analysis
setup.bat make help
```

---

## 🛠️ Makefile Commands

### Check Status
```cmd
make status          # Check master.parquet and notebooks
make list-status     # Detailed completion status
make check-master    # Just check master.parquet
```

### Run Pipeline
```cmd
make                 # Run everything (check-master + run-analysis)
make process         # Create master.parquet (if missing)
make run-analysis    # Run all analysis notebooks
```

### Manage Notebooks
```cmd
make mark-complete NB=path\to\notebook.ipynb    # Mark as done
make clean-all       # Reset all completion markers
make clean-master    # Delete master.parquet
```

### Configuration
Edit `Makefile` configuration section:
```makefile
ENABLE_TIMEOUT = TRUE    # Set FALSE for no timeout
TIMEOUT_SECONDS = 600    # Adjust timeout (seconds)
```

### Help
```cmd
make help           # Show all commands
```

---

## 📁 Project Structure

```
CS506_Project/
├── setup.bat                    # ⭐ ALL-IN-ONE TOOL (setup/activate/make)
├── requirements.txt             # Python dependencies
├── Makefile                     # Pipeline automation
│
├── 1_LIB/
│   └── master/
│       └── master.parquet       # Combined dataset (created by pipeline)
│
├── 2_FIGURES/
│   ├── 1_data_wrangling/
│   │   └── 1st_pass.ipynb       # Data processing notebook
│   ├── linear_regression.ipynb
│   ├── SVM_*.ipynb
│   └── XGBoost_*.ipynb
│
├── venv/                        # Virtual environment (auto-created)
└── .make_completion/            # Completion markers (auto-created)
```

---

## 🔍 Troubleshooting

### Python Not Found
```
[ERROR] Python is not installed or not in PATH!
```

**Solution:**
1. Download Python: https://python.org/downloads
2. During install: ✅ Check "Add Python to PATH"
3. Run `setup.bat` again

### Virtual Environment Issues
```
[ERROR] Failed to create virtual environment
```

**Solution:**
```cmd
rmdir /s /q venv
setup.bat
```

### Package Import Errors in Notebooks
```python
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
Make sure Jupyter is running in the virtual environment:
```cmd
venv\Scripts\activate.bat
jupyter notebook
```

### Makefile Not Found
```
'make' is not recognized as an internal or external command
```

**Solution - Install make:**
```cmd
choco install make
```
(First install Chocolatey: https://chocolatey.org/install)

### Notebooks Not Found
```
[WARNING] notebook.ipynb not found - skipping
```

**Solution:**
1. Find notebooks: `find_notebooks.bat`
2. Update paths in `Makefile` line 18

---

## 📦 Installed Packages

From `requirements.txt`:

### Core Data Science
- numpy, pandas, scipy
- matplotlib, seaborn, plotly

### Machine Learning
- scikit-learn, xgboost

### Jupyter
- jupyter, notebook, ipykernel, nbconvert

### Data I/O
- pyarrow, fastparquet

### Utilities
- requests, beautifulsoup4, tqdm, pytz

---

## 🎓 Tips & Best Practices

1. **Always use the virtual environment**
   - Keeps project dependencies isolated
   - Prevents conflicts with other projects

2. **Run `setup.bat` before starting work**
   - Opens activated terminal
   - Ensures everything is installed

3. **Use `make` commands for pipeline automation**
   - Handles notebook execution
   - Tracks completion state

4. **Check status frequently**
   ```cmd
   make list-status
   ```

5. **Mark heavy computations as complete**
   ```cmd
   make mark-complete NB=heavy_notebook.ipynb
   ```

---

## 📖 Documentation Files

- **`QUICKSTART.md`** - Ultra-concise getting started
- **`README.md`** - This file (complete guide)
- **`SETUP_README.md`** - Detailed environment setup
- **`MAKEFILE_README.md`** - Makefile usage guide

---

## ✨ Summary

**Setup:**
```
setup.bat
```

**Daily use:**
```cmd
setup.bat activate        # Open activated terminal
setup.bat make status     # Run make status
setup.bat make run-analysis
jupyter notebook          # In activated window
```

**That's it!** The rest is automated. 🎉

---

## 🆘 Need Help?

1. Check `make help`
2. Read `QUICKSTART.md`
3. Check troubleshooting section above
4. Verify Python: `py --version`
5. Verify packages: `setup.bat` (re-run to verify)

---

**Questions?** Check the documentation files or run validation:
```
validate_makefile.bat
find_python.bat
find_notebooks.bat
```
