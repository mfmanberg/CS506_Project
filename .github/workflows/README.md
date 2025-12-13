# GitHub Workflows Documentation

**Last Updated:** December 13, 2025  
**Status:** ✅ All workflows configured for reproducibility

This directory contains GitHub Actions workflow files for automated CI/CD testing.

---

## Workflow Files

### Active Workflows
- **workflows_linear_regression.yml** - Linear regression model (4-5 min)
- **workflows_svmdaily.yml** - SVM daily forecasting (1-2 min)
- **workflows_xgboost_testing.yml** - XGBoost model (1-2 min)

### Configuration
All workflows are configured to run automatically on push to `main` branch or manually via `workflow_dispatch`.

---

## Reproducibility Features

###  Environment Setup
- **Python:** 3.12 or 3.13 (ubuntu-latest)
- **Dependencies:** Installed from `Dependencies/requirements.txt`
- **Git LFS:** Enabled for large files (master.parquet)
- **Cache:** pip cache enabled for faster builds

###  Path Configuration
All workflows export `PYTHONPATH` to ensure `path_utils.py` is accessible:
```bash
export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"
```

###  Notebook Execution
- Uses **papermill** with `--cwd "$PWD"` flag for correct working directory
- Timeout: 1800 seconds (30 minutes) per notebook
- Separate output files to avoid overwriting source notebooks

###  Result Extraction
- Runs `Build/extract_results.py` to parse metrics
- Logs saved to `Build/model_results.log`
- Artifacts retained for 30 days

---

## Requirements for Reproducibility

### Repository Requirements
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
