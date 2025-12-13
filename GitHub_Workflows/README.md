# GitHub Workflows Directory

This directory contains all GitHub Actions workflow files and documentation for CI/CD automation.

## Workflow Files

### Active Workflows (in `.github/workflows/`)
- **linear_regression.yml** - Tests linear regression model (4-5 min)
- **svmdaily.yml** - Tests SVM daily forecasting model (1-2 min)
- **xgboost_testing.yml** - Tests XGBoost model (1-2 min)

### Local Copies (for reference)
- **workflows_linear_regression.yml**
- **workflows_svmdaily.yml**
- **workflows_xgboost_testing.yml**

## Workflow Status

All workflows run automatically on `push` to `main` branch:

| Workflow | Status | Duration | Last Run |
|----------|--------|----------|----------|
| Linear Regression | ✅ PASSING | 4m 5s | Run #20191651950 |
| XGBoost Testing | ✅ PASSING | 1m 34s | Run #20191651954 |
| SVM Daily | ✅ PASSING | 1m 23s | Run #20191652644 |

## Features

**All workflows include:**
- ✅ Python 3.12/3.13 setup
- ✅ Dependency installation from `Dependencies/requirements.txt`
- ✅ Automated notebook execution via papermill
- ✅ Result extraction and logging
- ✅ Artifact upload (executed notebooks)
- ✅ Execution verification

## Viewing Results

**Via GitHub:**
1. Go to Actions tab: https://github.com/mfmanberg/CS506_Project/actions
2. Click on a workflow run
3. Download artifacts from the bottom of the page

**Via GitHub CLI:**
```bash
# List recent runs
gh run list --limit 5

# View specific run
gh run view <run_id>

# Download artifacts
gh run download <run_id>
```

## Triggering Workflows

**Automatic:** Push to `main` branch

**Manual:**
```bash
gh workflow run "Linear Regression Test"
gh workflow run "SVM Daily Test"
gh workflow run "XGBoost Testing"
```

## Configuration

Workflows use:
- **OS:** ubuntu-latest
- **Python:** 3.12 or 3.13
- **Timeout:** 1800s per notebook
- **Artifacts:** Retained for 90 days

## Requirements

- Dependencies from `../Dependencies/requirements.txt`
- Data file: `1_LIB/master/master.parquet` (tracked via Git LFS)
- Scripts: `../Makefile/extract_results.py`, `../Makefile/test_execution.py`
