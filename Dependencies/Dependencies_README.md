# Dependencies Documentation

## 1. Overview

**Last Updated:** December 13, 2025  
**Python Version:** 3.12.3  
**Virtual Environment:** `.venv_wsl`  
**Total Packages:** 121 explicit + 18 transitive = 139 installed  
**Requirements File:** [requirements.txt](requirements.txt)

### Current Environment Status
✅ **All dependencies installed and verified**  
✅ **No broken requirements** (`pip check` passed)  
✅ **GitHub Actions compatible**  
✅ **WSL2 + Ubuntu 24.04.3 LTS tested**

---

## 2. System Configuration

### WSL2 Environment
- **WSL Version:** 2.6.1.0
- **Kernel:** 6.6.87.2-1 (microsoft-standard-WSL2)
- **Distribution:** Ubuntu 24.04.3 LTS (Noble Numbat)
- **Python:** 3.12.3
- **pip:** 25.3

---

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
**Bandwidth:** ~500MB download

#### 4. Verify Installation
```bash
# Check pip version (should be 25.3+)
pip --version

# Verify no conflicts
pip check

# Test critical package imports
python -c "import pandas, numpy, sklearn, xgboost, matplotlib, seaborn, plotly, statsmodels, papermill; print('✓ All imports successful')"

# Test papermill
papermill --version

# Test path utilities (if in project root)
python -c "import sys; sys.path.insert(0, 'Build'); from path_utils import get_project_root; print(get_project_root())"

# Run comprehensive test script
bash Dependencies/test_dependencies.sh
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
    python -m pip instal7 critical packages)

| Package | Version | Purpose | Size | Status |
|---------|---------|---------|------|--------|
| **numpy** | 2.3.5 | Numerical computing foundation | ~50MB | ✅ Installed |
| **pandas** | 2.3.3 | Data manipulation & analysis | ~40MB | ✅ Installed |
| **pyarrow** | 22.0.0 | Parquet I/O (fast columnar) | ~35MB | ✅ Installed |
| **scikit-learn** | 1.8.0 | ML algorithms (SVM, regression) | ~30MB | ✅ Installed |
| **xgboost** | 3.1.2 | Gradient boosting (primary model) | ~10MB | ✅ Installed |
| **statsmodels** | 0.14.6 | Statistical models & tests | ~25MB | ✅ Installed |
| **scipy** | 1.16.3 | Scientific computing (optimization) | ~60MB | ✅ Installed |
| **matplotlib** | 3.10.8 | Plotting & visualization | ~20MB | ✅ Installed |
| **seaborn** | 0.13.2 | Statistical visualization | ~5MB | ✅ Installed |
| **plotly** | 6.5.0 | Interactive plots | ~15MB | ✅ Installed |
| **jupyter** | 1.1.1 | Notebook interface | ~2MB | ✅ Installed |
| **jupyterlab** | 4.5.0 | Advanced notebook UI | ~30MB | ✅ Installed |
| **papermill** | 2.6.0 | Notebook automation (build system) | ~2MB | ✅ Installed |
| **joblib** | 1.5.2 | Model persistence (.joblib files) | ~1MB | ✅ Installed |
| **duckdb** | 1.4.3 | Fast data processing | ~25MB | ✅ Installed |
| **fastparquet** | 2024.11.0 | Alternative parquet engine | ~5MB | ✅ Installed |
| **tqdm** | 4.67.1 | Progress bars | ~1MB | ✅ Installed |

### Explicit Dependencies
- **121 packages** explicitly listed in [requirements.txt](requirements.txt)
- All versions pinned for reproducibility
- Grouped by function (data, ML, visualization, Jupyter)

### Transitive Dependencies
- **18 additional packages** auto-installed as dependencies
- Examples: `aiohappyeyeballs`, `aiohttp`, `aiosignal`, `ansicolors`
- All verified with `pip check` (no conflicts)

### Total Installation Footprint
- **Installed Packages:** 139 total
- **Virtual Environment Size:** ~2GB
- **Download Size:** ~500MB (varies by mirror)
| **seaborn** | 0.13.2 | Statistical viz | ✅ Installed |
| **plotly** | 6.5.0 | Interactive 24.04 (Current Production)
```
OS: Windows 11 with WSL2 version 2.6.1.0
Distribution: Ubuntu 24.04.3 LTS (Noble Numbat)
Kernel: 6.6.87.2-microsoft-standard-WSL2
Python: 3.12.3
pip: 25.3
Virtual Environment: .venv_wsl
Total Packages: 139 (121 explicit + 18 transitive)
Status: ✅ WORKING (verified December 13, 2025)
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
**WSL2 installed and configured** (version 2.6.1.0)
- [x] **Ubuntu 24.04.3 LTS** distribution active
- [x] **Python 3.12.3** installed in WSL
- [x] **Virtual environment** created (`.venv_wsl`)
- [x] **Virtual environment activated** (`source .venv_wsl/bin/activate`)
- [x] **pip upgraded** to latest version (25.3)
- [x] **Dependencies installed** from requirements.txt (121 explicit packages)
- [x] **No conflicts** (`pip check` returns "No broken requirements found")
- [x] **Critical imports** working (pandas, numpy, sklearn, xgboost)
- [x] **papermill** command available and functional
- [x] **path_utils** module accessible from Build folder
- [x] **Test script** executes (`bash Dependencies/test_dependencies.sh`)
- [x] **Full build** completes (9 notebooks via `Build\run_build.bat`)
- [x] **extract_results.py** runs without errors
- [x] **GitHub Actions** workflows pass (if applicable)

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
Package Details

### Key Package Versions (Verified December 13, 2025)

**Data Processing:**
- pandas 2.3.3 | numpy 2.3.5 | pyarrow 22.0.0 | duckdb 1.4.3

**Machine Learning:**
- scikit-learn 1.8.0 | xgboost 3.1.2 | statsmodels 0.14.6 | scipy 1.16.3

**Visualization:**
- matplotlib 3.10.8 | seaborn 0.13.2 | plotly 6.5.0

**Jupyter Ecosystem:**
- jupyter 1.1.1 | jupyterlab 4.5.0 | papermill 2.6.0 | ipykernel 7.1.0

**Utilities:**
- joblib 1.5.2 | tqdm 4.67.1 | fastparquet 2024.11.0

### Dependency Health Check
```bash
# Run from project root with venv activated
pip check
# Output: "No broken requirements found" ✅
```

---

## 11. Conclusion

### ✅ All Tests Passed

**Dependencies:** All 121 explicit packages + 18 transitive = 139 total installed and verified  
**Conflicts:** None (`pip check` clean)  
**E# Ensure Python 3.12 is installed
   sudo apt update && sudo apt install python3.12 python3.12-venv
   
   # Create and activate virtual environment
   python3.12 -m venv .venv_wsl
   source .venv_wsl/bin/activate
   
   # Install dependencies (121 packages)
   pip install --upgrade pip
   pip install -r Dependencies/requirements.txt
   
   # Verify installation
   pip check  # Should show "No broken requirements found"
   ```

3. **Run notebooks:**
   ```bash
   # Check build results
   cat Build/model_results.log
   
   # View installed packages
   pip list
   
   # Check for updates
   pip list --outdated
   ```

### Support & Resources
- **Build System:** See [Build/Build_README.md](../Build/Build_README.md)
- **Test Script:** [test_dependencies.sh](test_dependencies.sh)
- **Requirements:** [requirements.txt](requirements.txt) (121 pinned packages)
- **GitHub Actions:** Automatically run on every push (if configured)
- **Issues:** Report via GitHub repository

---

**Report Generated:** December 13, 2025  
**Test Environment:** WSL2 2.6.1.0, Ubuntu 24.04.3 LTS, Python 3.12.3, pip 25.3  
**Total Packages:** 139 installed (121 explicit + 18 transitive)  
**Status:** ✅ All dependencies verified, no conflicts
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
