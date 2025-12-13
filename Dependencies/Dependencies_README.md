# Dependency Installation


## 3. Installation Process Documentation

### Step-by-Step Verified Process

#### 1. WSL Setup (Windows)
```bash
# Install WSL (if not already installed)
wsl --install

# Install Python 3.12
sudo apt update
sudo apt install python3.12 python3.12-venv
```

#### 2. Create Virtual Environment
```bash
# Navigate to project directory
cd /path/to/CS506_Project

# Create virtual environment
python3.12 -m venv .venv_wsl

# Activate virtual environment
source .venv_wsl/bin/activate
```

#### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

**Installation Time:** ~5-10 minutes (depending on connection speed)
**Disk Space Required:** ~2GB for virtual environment

#### 4. Verify Installation
```bash
# Test package imports
python -c "import pandas, numpy, sklearn, xgboost, matplotlib, seaborn, plotly, statsmodels, papermill; print('OK')"

# Test papermill
papermill --version

# Test path utilities
python -c "from path_utils import MASTER_PARQUET; print(MASTER_PARQUET)"
```

---

## 4. GitHub Actions Compatibility

### Workflow Verification
All 3 GitHub Actions workflows use the same requirements.txt:

**✅ Linear Regression Test** (Run #20191651950)
- Duration: 4m 5s
- All dependencies installed successfully
- Notebook executed without errors

**✅ XGBoost Testing** (Run #20191651954)
- Duration: 1m 34s
- All dependencies installed successfully
- Notebook executed without errors

**✅ SVM Daily Test** (Run #20191652644)
- Duration: 1m 23s
- All dependencies installed successfully
- Notebook executed without errors

### GitHub Actions Environment
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
```

**Result:** ✅ All workflows pass with updated requirements.txt

---

## 5. Dependency Matrix

### Core Dependencies (15 critical packages)

| Package | Version | Purpose | Status |
|---------|---------|---------|--------|
| **numpy** | 2.3.5 | Numerical computing | ✅ Installed |
| **pandas** | 2.3.3 | Data manipulation | ✅ Installed |
| **pyarrow** | 22.0.0 | Parquet I/O | ✅ Installed |
| **scikit-learn** | 1.8.0 | ML algorithms | ✅ Installed |
| **xgboost** | 3.1.2 | Gradient boosting | ✅ Installed |
| **statsmodels** | 0.14.6 | Statistical models | ✅ Installed |
| **scipy** | 1.16.3 | Scientific computing | ✅ Installed |
| **matplotlib** | 3.10.8 | Plotting | ✅ Installed |
| **seaborn** | 0.13.2 | Statistical viz | ✅ Installed |
| **plotly** | 6.5.0 | Interactive viz | ✅ Installed |
| **jupyter** | 1.1.1 | Notebook interface | ✅ Installed |
| **jupyterlab** | 4.5.0 | Lab interface | ✅ Installed |
| **papermill** | 2.6.0 | Automation | ✅ Installed |
| **joblib** | 1.5.2 | Model persistence | ✅ Installed |
| **duckdb** | 1.4.3 | Data processing | ✅ Installed |

### Supporting Dependencies (110 packages)
All transitive dependencies and utilities properly installed and verified.

---

## 6. Known Working Configurations

### Configuration 1: WSL2 + Ubuntu (Tested)
```
OS: Windows 11 with WSL2
Distribution: Ubuntu 22.04 LTS
Python: 3.12.3
Virtual Environment: .venv_wsl
Status: ✅ WORKING
```

### Configuration 2: GitHub Actions (Tested)
```
OS: ubuntu-latest
Python: 3.12 / 3.13
Virtual Environment: GitHub Actions managed
Status: ✅ WORKING
```

### Configuration 3: Windows Native
```
OS: Windows 11
Python: 3.12
Status: ⚠️ NOT RECOMMENDED
Reason: Asyncio issues with Jupyter kernel
Workaround: Use WSL2 instead
```

---

## 7. Troubleshooting Guide

### Issue: Import errors after installation
**Solution:**
```bash
# Ensure virtual environment is activated
source .venv_wsl/bin/activate

# Verify pip is using venv
which pip  # Should show .venv_wsl path

# Reinstall if needed
pip install --force-reinstall -r requirements.txt
```

### Issue: Papermill fails to execute
**Solution:**
```bash
# Install Jupyter kernel
python -m ipykernel install --user --name python3

# Test papermill
papermill --help
```

### Issue: Path errors in notebooks
**Solution:**
```python
# Ensure path_utils is imported correctly
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from path_utils import MASTER_PARQUET
```

---

## 8. Testing Checklist

Use this checklist to verify a fresh installation:

- [x] WSL2 installed and configured
- [x] Python 3.12 installed in WSL
- [x] Virtual environment created (.venv_wsl)
- [x] Virtual environment activated
- [x] pip upgraded to latest version (25.3)
- [x] All dependencies installed from requirements.txt (125 packages)
- [x] Critical packages import successfully
- [x] papermill command available
- [x] path_utils module accessible
- [x] Test notebook executes via papermill
- [x] extract_results.py runs without errors
- [x] GitHub Actions workflows pass

---

## 9. Performance Benchmarks

### Installation Performance
- **Dependency Download:** ~3-5 minutes (2GB bandwidth)
- **Package Installation:** ~2-3 minutes (compilation)
- **Total Setup Time:** ~5-10 minutes

### Execution Performance (Single Notebook)
- **XGBoost_Testing:** ~30-60 seconds
- **Linear Regression:** ~180-240 seconds
- **SVM Daily:** ~60-90 seconds

### Full Makefile Execution (7 Notebooks)
- **Estimated Time:** 30-60 minutes
- **Resource Usage:** ~8GB RAM, 4 CPU cores
- **Disk I/O:** Reading 38MB master.parquet + writing results

---

## 10. Conclusion

### ✅ All Tests Passed

**Dependencies:** All 125 packages installed and verified
**Execution:** Papermill successfully executes notebooks
**Cross-Platform:** Works on WSL, Linux, and GitHub Actions
**Reproducibility:** requirements.txt enables one-command setup
**Documentation:** Comprehensive setup instructions included

### Next Steps for New Users

1. **Clone repository:**
   ```bash
   git clone https://github.com/mfmanberg/CS506_Project.git
   cd CS506_Project
   ```

2. **Setup environment (WSL):**
   ```bash
   python3.12 -m venv .venv_wsl
   source .venv_wsl/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run notebooks:**
   ```bash
   # Single notebook
   ./RunNotebooks.ps1
   
   # Or full makefile
   make -f Makefile.wsl
   ```

4. **View results:**
   ```bash
   cat model_results.log
   ```

### Support
- GitHub Actions: Automatically run on every push
- Documentation: See README.md and TEST_RESULTS.md
- Issues: Open on GitHub repository

---

**Report Generated:** December 13, 2025  
**Test Environment:** WSL2 Ubuntu 22.04, Python 3.12.3  
**Project:** CS506 NYISO Load Forecasting
