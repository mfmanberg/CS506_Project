# Makefile Directory

This directory contains all build automation and execution scripts for the CS506 NYISO Load Forecasting project.

## Files

### Core Execution
- **Makefile.wsl** - Linux makefile for automated notebook execution
- **RunNotebooks.ps1** - PowerShell wrapper for Windows
- **path_utils.py** - Cross-platform path utilities
- **extract_results.py** - Result extraction from executed notebooks
- **test_execution.py** - Verify notebook execution completion

## Usage

### From Project Root (Recommended)

**Windows (PowerShell):**
```powershell
.\Makefile\RunNotebooks.ps1
```

**WSL/Linux:**
```bash
make -f Makefile/Makefile.wsl
```

### Extract Results
```bash
python Makefile/extract_results.py
```

### Test Execution
```bash
python Makefile/test_execution.py <notebook_path>
```

## Configuration

All scripts automatically detect the project root and use absolute paths for reproducibility.

**Key Features:**
- ✅ Dynamic path resolution
- ✅ Cross-platform compatibility (Windows/WSL/Linux)
- ✅ Automated error handling
- ✅ Result logging to `model_results.log`
- ✅ Works from any directory

## Requirements

- Python 3.12+ with virtual environment at `.venv_wsl/`
- All dependencies from `Dependencies/requirements.txt` installed
- Data file: `1_LIB/master/master.parquet` (38MB via Git LFS)

## Execution Flow

1. **Activate virtual environment** → `.venv_wsl/bin/activate`
2. **Execute 7 notebooks** → via papermill
3. **Extract results** → `extract_results.py`
4. **Log output** → `model_results.log`

## Performance

- **Single notebook:** 30-240 seconds
- **Full makefile (7 notebooks):** 30-60 minutes
- **Resource usage:** ~8GB RAM, 4 CPU cores
