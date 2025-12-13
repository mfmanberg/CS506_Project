# GitHub Workflows Documentation

**Last Updated:** December 13, 2025  
**Status:** ✅ All 3 workflows PASSING - Code is reproducible

---

## Quick Start

### Check Workflow Status
```powershell
# Windows (PowerShell)
.\.github\workflows\Check-WorkflowStatus.ps1

# WSL/Linux (Bash)
bash .github/workflows/check_workflow_status.sh
```

### Test Before Pushing
```bash
# Quick validation (30 seconds)
bash .github/workflows/test_reproducibility.sh

# Full simulation (5+ minutes, executes 3 notebooks)
bash .github/workflows/test_full_workflow.sh
```

### Trigger Workflows Manually
```powershell
# PowerShell
.\.github\workflows\trigger_workflows.ps1

# Or use GitHub CLI
gh workflow run "Linear Regression Test"
gh workflow run "SVM Daily Test"
gh workflow run "XGBoost Testing"
```

---

## Files in This Directory

### Workflow Definitions (YAML)
| File | Purpose | Runtime | Status |
|------|---------|---------|--------|
| **workflows_linear_regression.yml** | Tests linear regression model | ~4-5 min | ✅ Passing |
| **workflows_svmdaily.yml** | Tests SVM daily forecasting | ~1-2 min | ✅ Passing |
| **workflows_xgboost_testing.yml** | Tests XGBoost model | ~1-2 min | ✅ Passing |

**What they do:**
- Auto-run on push to `main` branch
- Install dependencies from `Dependencies/requirements.txt`
- Execute notebook via papermill with `PYTHONPATH` set
- Extract metrics to `Build/model_results.log`
- Upload artifacts (executed notebooks + logs)

### Testing & Monitoring Scripts

| File | Platform | Purpose | Usage |
|------|----------|---------|-------|
| **Check-WorkflowStatus.ps1** | PowerShell | Monitor workflow status, auto-wait for completion | `.\.github\workflows\Check-WorkflowStatus.ps1` |
| **check_workflow_status.sh** | Bash | Same as above for Linux/WSL | `bash .github/workflows/check_workflow_status.sh` |
| **test_reproducibility.sh** | Bash | 9-point checklist + auto-install dependencies | `bash .github/workflows/test_reproducibility.sh` |
| **test_full_workflow.sh** | Bash | Full GitHub Actions simulation locally | `bash .github/workflows/test_full_workflow.sh` |
| **trigger_workflows.ps1** | PowerShell | Manually trigger all workflows | `.\.github\workflows\trigger_workflows.ps1` |

### Documentation
| File | Contents |
|------|----------|
| **Workflow_README.md** | This file - complete workflow documentation |
| **TROUBLESHOOTING.md** | 6 common issues with solutions (300+ lines) |

---

## How to Run Workflows

### Option 1: Automatic (Push to GitHub)
```bash
git add .
git commit -m "Update notebooks"
git push origin main
```
**Result:** All 3 workflows trigger automatically

### Option 2: Manual Trigger (GitHub UI)
1. Go to https://github.com/mfmanberg/CS506_Project/actions
2. Select workflow from left sidebar
3. Click "Run workflow" button
4. Select branch (main)
5. Click "Run workflow"

### Option 3: Manual Trigger (GitHub CLI)
```bash
# Single workflow
gh workflow run "XGBoost Testing"

# Check status
gh run list --limit 5

# View logs
gh run view <run_id> --log
```

### Option 4: PowerShell Script (Trigger All)
```powershell
.\.github\workflows\trigger_workflows.ps1
```

---

## Testing Workflow Scripts

### Test 1: Reproducibility Check (30 seconds)
```bash
bash .github/workflows/test_reproducibility.sh
```

**Tests performed:**
1. ✅ Virtual environment exists/created
2. ✅ Dependencies installed from requirements.txt
3. ✅ Critical packages verified (pandas, numpy, scikit-learn, xgboost, papermill)
4. ✅ Workflow files exist
5. ✅ PYTHONPATH configured in all workflows
6. ✅ Papermill `--cwd` flag present
7. ✅ Dependencies path correct
8. ✅ Git LFS enabled
9. ✅ No hardcoded paths
10. ✅ Required files present
11. ✅ path_utils imports successfully
12. ✅ Notebooks use path_utils

### Test 2: Full Workflow Simulation (5+ minutes)
```bash
bash .github/workflows/test_full_workflow.sh
```

**Steps executed:**
## What Each File Does (Detailed)

### Workflow YAML Files

**workflows_linear_regression.yml**
- **Trigger:** Push/PR to main, manual dispatch
- **Actions:**
  1. Checkout repo with Git LFS
  2. Setup Python 3.12 with pip cache
  3. Install dependencies from `Dependencies/requirements.txt`
  4. Verify notebook exists
  5. Export `PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"`
  6. Execute notebook: `papermill linear_regression.ipynb linear_regression_output.ipynb --cwd "$PWD"`
  7. Run `Build/extract_results.py` to parse metrics
  8. Upload artifacts (executed notebook + model_results.log)
  9. Verify execution with `Build/test_notebook_execution.py`

**workflows_svmdaily.yml**
- **Trigger:** Push/PR to main, manual dispatch
- **Actions:**
  1. Checkout repo with Git LFS
  2. Setup Python 3.13 with pip cache
  3. Install dependencies + papermill
  4. Export `PYTHONPATH` 
  5. Execute: `papermill SVMDaily.ipynb SVMDaily_output.ipynb --cwd "$PWD"`
  6. Extract results to log
  7. Upload artifacts

**workflows_xgboost_testing.yml**
- **Trigger:** Push/PR to main, manual dispatch
- **Actions:**
  1. Checkout repo with Git LFS
  2. Setup Python 3.12 with pip cache
  3. Install dependencies
  4. Verify notebook exists
  5. Export `PYTHONPATH`
  6. Execute: `papermill XGBoost_Testing.ipynb XGBoost_Testing_output.ipynb --cwd "$PWD"`
  7. Extract results
  8. Upload artifacts
  9. Verify execution

### Monitoring Scripts

**Check-WorkflowStatus.ps1 (PowerShell)**
- Checks if GitHub CLI is installed/authenticated
- Lists all available workflows
- Identifies in-progress workflows
- **Auto-waits** up to 5 minutes for completion (30s intervals)
- Gets latest status for each workflow (Linear Regression, SVM Daily, XGBoost)
- Displays success (✅) or failure (❌) with run IDs
- Shows error logs for failed runs
- **Exit code:** 0 if all pass, 1 if any fail
- **Best for:** Real-time monitoring, CI/CD integration

**check_workflow_status.sh (Bash)**
- Same functionality as PowerShell version
- Works in WSL/Linux environments
- Auto-waits up to 10 minutes for completion
- Provides detailed failure logs
- **Best for:** Linux servers, WSL environments

**test_reproducibility.sh (Bash)**
- **Step 0:** Check/create virtual environment, install dependencies
- **Steps 1-9:** Validate workflow configuration
- Tests: workflow files, PYTHONPATH, --cwd flag, paths, Git LFS, hardcoded paths, required files, imports
- **Auto-fixes:** Creates venv, installs deps if missing
- **Exit code:** 0 if all pass, 1 if errors found
- **Best for:** Pre-push validation, new machine setup

**test_full_workflow.sh (Bash)**
- **Full simulation** of GitHub Actions environment
- Creates venv, installs deps, verifies packages
- Checks Git LFS data file (master.parquet)
- Executes 3 notebooks locally (5 min timeout each)
- Extracts results, verifies environment
- **Cleanup:** Removes temp files on exit
- **Exit code:** 0 if all pass, 1 if any fail
- **Best for:** Testing changes before push, debugging workflow issues

**trigger_workflows.ps1 (PowerShell)**
- Triggers all 3 workflows manually via GitHub CLI
- Prompts for confirmation before triggering
- Shows triggered workflow IDs
- **Best for:** Forcing re-runs without pushing code

---

## Reproducibility Requirements

### Critical Configuration
All 3 workflows require:
1. **PYTHONPATH export** before Python commands:
   ```yaml
   run: |
     export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"
   ```

2. **Papermill --cwd flag**:
   ```yaml
   papermill notebook.ipynb output.ipynb --cwd "$PWD"
   ```

3. **Git LFS checkout**:
   ```yaml
   uses: actions/checkout@v4
   with:
     lfs: true
   ```

4. **Correct dependency path**:
   ```yaml
   pip install -r Dependencies/requirements.txt
   ```

### Why Each Part Matters

**PYTHONPATH:** Allows notebooks to `from path_utils import get_project_root` without errors

**--cwd flag:** Sets working directory so relative paths resolve correctly

**Git LFS:** Downloads actual master.parquet file (38MB) instead of pointer file

**Dependencies path:** Installs all 121 packages from correct location

---

## Common Workflows

### Setup on New Machine
```bash
git clone https://github.com/mfmanberg/CS506_Project.git
cd CS506_Project
bash .github/workflows/test_reproducibility.sh  # Auto-creates venv, installs deps
```

### Before Pushing Changes
```bash
bash .github/workflows/test_reproducibility.sh  # 30 seconds
# If all pass:
git add .
git commit -m "Update"
git push origin main
```

### After Push - Check Status
```powershell
.\.github\workflows\Check-WorkflowStatus.ps1  # Auto-waits for completion
```

### Debug Failed Workflow
```bash
# Get run ID from GitHub Actions page
gh run view <run_id> --log

# Or simulate locally:
bash .github/workflows/test_full_workflow.sh
```

### Force Re-run All Workflows
```powershell
.\.github\workflows\trigger_workflows.ps1
```

---

## Troubleshooting

**Common Issues:**
- ModuleNotFoundError: Check PYTHONPATH in workflow
- File not found: Verify --cwd flag and Git LFS
- Dependencies fail: Check `Dependencies/requirements.txt` path
- Timeout: Increase `--execution-timeout` value

**See:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions

---

**View Live Status:** https://github.com/mfmanberg/CS506_Project/actions  
**Current Status:** ✅ All 3 workflows passing (December 13, 2025)
1.  `Dependencies/requirements.txt` - All Python packages (121 explicit)
2.  `Build/path_utils.py` - Cross-platform path resolution
3.  `Build/extract_results.py` - Metric extraction script
4.  `Build/test_notebook_execution.py` - Environment verification
5.  `1_LIB/master/master.parquet` - Data file (tracked via Git LFS)

### Workflow Requirements
1.  Git LFS checkout enabled (`lfs: true`)
2. `PYTHONPATH` exported to include project root and Build folder
3.  Papermill executed with `--cwd "$PWD"` flag
4.  All scripts use relative paths (no hardcoded paths)

### Notebooks Requirements
All notebooks must:
- Import `path_utils` for data paths: `from path_utils import get_project_root`
- Use relative paths from project root
- Not contain hardcoded absolute paths

---

## Workflow Status

View live status at: https://github.com/mfmanberg/CS506_Project/actions

| Workflow | Python | Timeout | Artifacts |
|----------|--------|---------|-----------|
| Linear Regression | 3.12 | 1800s | notebook + log |
| SVM Daily | 3.13 | 1800s | notebook + log |
| XGBoost Testing | 3.12 | 1800s | notebook + log |

---

## Testing Locally

### Automated Reproducibility Test

Run the comprehensive test script to verify all workflows are configured correctly:

```bash
bash .github/workflows/test_reproducibility.sh
```

This script verifies:
-  All workflow files exist
-  PYTHONPATH is configured in each workflow
-  Papermill uses `--cwd` flag
-  Dependencies path is correct (Dependencies/requirements.txt)
-  Git LFS is enabled
-  No hardcoded paths exist
-  All required files present
-  path_utils imports successfully
-  Notebooks use path_utils

### Manual Testing

### Before Pushing to GitHub

1. **Test path_utils access:**
   ```bash
   export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"
   python3 -c "from path_utils import get_project_root; print(get_project_root())"
   ```

2. **Test notebook execution:**
   ```bash
   export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"
   papermill \
     3_OUTPUT/3_xg_boost/XGBoost_Testing.ipynb \
     /tmp/test_output.ipynb \
     --kernel python3 \
     --execution-timeout 1800 \
     --cwd "$PWD"
   ```

3. **Test extract_results:**
   ```bash
   export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"
   python3 Build/extract_results.py
   ```

4. **Run full test suite:**
   ```bash
   bash Dependencies/test_dependencies.sh
   python3 Build/test_notebook_execution.py
   ```

---
