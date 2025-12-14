# GitHub Workflows

**Status:** ✅ All 3 workflows passing  
**View:** https://github.com/mfmanberg/CS506_Project/actions

---

## Quick Reference

### Check Status
```bash
# Windows
.\.github\workflows\Check-WorkflowStatus.ps1

# Linux/WSL
bash .github/workflows/check_workflow_status.sh
```

### Test Before Push
```bash
bash .github/workflows/test_reproducibility.sh  # 30s validation
bash .github/workflows/test_full_workflow.sh    # Full simulation
```

### Trigger Manually
```bash
gh workflow run "Linear Regression Test"
gh workflow run "SVM Daily Test"
gh workflow run "XGBoost Testing"
```

---

## Active Workflows

| Workflow | Python | Runtime | Status |
|----------|--------|---------|--------|
| Linear Regression | 3.12 | ~4-5 min | ✅ |
| SVM Daily | 3.13 | ~1-2 min | ✅ |
| XGBoost Testing | 3.12 | ~1-2 min | ✅ |

**Triggers:** Auto on push to `main`, or manual dispatch  
**Outputs:** Executed notebook + metrics log (artifacts)

---

## Key Requirements

All workflows require:
1. **PYTHONPATH export:** `export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"`
2. **Papermill --cwd flag:** `papermill notebook.ipynb output.ipynb --cwd "$PWD"`
3. **Git LFS enabled:** Downloads master.parquet (38MB)
4. **Dependencies path:** `pip install -r Dependencies/requirements.txt`

---

## Testing Scripts

| Script | Purpose |
|--------|---------|
| `test_reproducibility.sh` | Validates workflow config, auto-installs deps |
| `test_full_workflow.sh` | Simulates GitHub Actions locally |
| `Check-WorkflowStatus.ps1` | Monitors runs, auto-waits for completion |
| `trigger_workflows.ps1` | Triggers all workflows manually |

---

## Common Tasks

**Setup new machine:**
```bash
git clone https://github.com/mfmanberg/CS506_Project.git
cd CS506_Project
bash .github/workflows/test_reproducibility.sh
```

**Before pushing:**
```bash
bash .github/workflows/test_reproducibility.sh
git add . && git commit -m "Update" && git push
```

**Debug failure:**
```bash
gh run view <run_id> --log
# or
bash .github/workflows/test_full_workflow.sh
```

---

## Troubleshooting

- **ModuleNotFoundError:** Check PYTHONPATH in workflow
- **File not found:** Verify --cwd flag and Git LFS
- **Dependencies fail:** Check `Dependencies/requirements.txt` path

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

---
