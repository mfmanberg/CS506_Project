# GitHub Workflows Troubleshooting Guide

**Last Updated:** December 13, 2025

---

## Common Issues & Solutions

### 1. ModuleNotFoundError: No module named 'path_utils'

**Symptom:** Notebook fails with import error for `path_utils`

**Cause:** `PYTHONPATH` not set to include Build folder

**Solution:**
```yaml
- name: Execute notebook
  run: |
    export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"
    papermill notebook.ipynb output.ipynb --cwd "$PWD"
```

**Verify locally:**
```bash
export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"
python -c "from path_utils import get_project_root; print(get_project_root())"
```

---

### 2. File not found: master.parquet

**Symptom:** Notebook fails loading data file

**Cause:** Git LFS not enabled in workflow checkout

**Solution:**
```yaml
- name: Checkout repository
  uses: actions/checkout@v4
  with:
    lfs: true  # Enable Git LFS
```

**Verify locally:**
```bash
git lfs install
git lfs pull
ls -lh 1_LIB/master/master.parquet
```

---

### 3. Dependencies installation fails

**Symptom:** `pip install -r requirements.txt` fails

**Cause:** Wrong path to requirements file

**Solution:**
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r Dependencies/requirements.txt  # Correct path
```

**Verify locally:**
```bash
test -f Dependencies/requirements.txt && echo "Found" || echo "Not found"
```

---

### 4. Papermill execution timeout

**Symptom:** Workflow fails after 30 minutes

**Cause:** Default timeout (1800s) exceeded

**Solution:**
```yaml
- name: Execute notebook
  run: |
    papermill notebook.ipynb output.ipynb \
      --execution-timeout 3600  # Increase to 60 minutes
```

**Adjust based on local testing:**
```bash
time papermill notebook.ipynb /tmp/output.ipynb
```

---

### 5. Working directory issues

**Symptom:** Relative paths don't resolve correctly

**Cause:** Papermill runs from wrong directory

**Solution:**
```yaml
- name: Execute notebook
  run: |
    papermill notebook.ipynb output.ipynb \
      --cwd "$PWD"  # Set working directory explicitly
```

**Verify locally:**
```bash
pwd
papermill notebook.ipynb /tmp/output.ipynb --cwd "$PWD"
```

---

### 6. Artifact upload fails

**Symptom:** Artifacts not available after run

**Cause:** File path doesn't exist or glob pattern wrong

**Solution:**
```yaml
- name: Upload artifacts
  if: always()  # Run even if previous steps failed
  uses: actions/upload-artifact@v4
  with:
    name: results
    path: |
      output.ipynb
      Build/model_results.log
```

**Verify files exist:**
```bash
ls -lh output.ipynb Build/model_results.log
```

---

## Debugging Workflow Runs

### 1. Check Workflow Logs
```bash
gh run list --limit 5
gh run view <run_id> --log
```

### 2. Download Artifacts
```bash
gh run download <run_id>
cd <artifact_name>
jupyter notebook *.ipynb  # View executed notebook
```

### 3. Test Locally First
```bash
# Set environment exactly as workflow
export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"

# Run same papermill command
papermill \
  3_OUTPUT/3_xg_boost/XGBoost_Testing.ipynb \
  /tmp/test_output.ipynb \
  --kernel python3 \
  --execution-timeout 1800 \
  --cwd "$PWD"

# Check output
jupyter notebook /tmp/test_output.ipynb
```

### 4. Verify Environment
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(pandas|numpy|scikit-learn|xgboost|papermill)"

# Check path_utils
python -c "import sys; sys.path.insert(0, 'Build'); from path_utils import get_project_root; print(get_project_root())"

# Check data file
ls -lh 1_LIB/master/master.parquet
```

---

## Workflow Best Practices

### ✅ Do:
- Always export `PYTHONPATH` before running Python scripts
- Use `--cwd "$PWD"` with papermill
- Enable Git LFS checkout (`lfs: true`)
- Use separate output files (don't overwrite source notebooks)
- Add `if: always()` to artifact upload steps
- Test locally before pushing

### ❌ Don't:
- Hardcode absolute paths in notebooks or workflows
- Forget to install papermill in dependencies
- Run multiple notebooks without PYTHONPATH reset
- Skip error handling in critical steps
- Use outdated action versions

---

## Testing Checklist

Before pushing workflow changes:

- [ ] Test papermill command locally with PYTHONPATH set
- [ ] Verify path_utils imports work
- [ ] Check master.parquet exists and Git LFS is configured
- [ ] Run extract_results.py to ensure it works
- [ ] Check all file paths are relative (no hardcoded paths)
- [ ] Verify Dependencies/requirements.txt path is correct
- [ ] Test with clean environment (fresh venv)
- [ ] Check notebook outputs for errors
- [ ] Verify artifacts can be downloaded

---

## Quick Reference Commands

```bash
# Test path_utils
export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"
python -c "from path_utils import get_project_root; print(get_project_root())"

# Test notebook execution
papermill notebook.ipynb /tmp/output.ipynb --kernel python3 --cwd "$PWD"

# Test dependencies
pip install -r Dependencies/requirements.txt

# Test extraction
python Build/extract_results.py

# View workflow runs
gh run list --limit 5

# Trigger workflow manually
gh workflow run "XGBoost Testing"
```

---

**For additional help:**
- See [README.md](README.md) for full documentation
- Check [Build/Build_README.md](../../Build/Build_README.md) for local execution
- Review [Dependencies/Dependencies_README.md](../../Dependencies/Dependencies_README.md) for setup
