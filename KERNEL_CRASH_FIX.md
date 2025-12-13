# Kernel Crash Fix

The notebooks are crashing with `DeadKernelError: Kernel died`. This is due to:

1. **Windows asyncio issues** - ProactorEventLoop incompatibility with zmq
2. **Memory overload** - Loading 38 MB parquet file multiple times
3. **Notebook code errors** - Errors in the notebook cells

## Quick Fix

**Add this to the FIRST cell of each failing notebook:**

```python
# Windows asyncio fix
import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Memory optimization
import gc
import pandas as pd

pd.options.mode.copy_on_write = True

def clear_mem():
    gc.collect()

print("Kernel fixes applied")
```

## Which Notebooks Failed?

Based on the output, these notebooks are crashing:
- `linear_regression.ipynb`
- `SVM_Trunc.ipynb`
- `SVMDaily.ipynb`
- `SVMDailywoutMeso.ipynb`

## How to Fix Them

### Option 1: Add Fix to Each Notebook (Recommended)

```bash
# Open each notebook
jupyter notebook 3_OUTPUT/3_linear_regression/linear_regression.ipynb

# Add the fix code to the FIRST cell (create new cell at top if needed)
# Run the notebook manually to test
```

### Option 2: Run Notebooks Individually

```bash
# Test one by one to see which cell fails
jupyter notebook 3_OUTPUT/3_linear_regression/linear_regression.ipynb
# Execute cells one by one
# Fix the failing cell
```

### Option 3: Reduce Data Size in Notebooks

If the notebooks load data, modify them to use less data:

```python
# Instead of:
# df = pd.read_parquet('../../1_LIB/master/master.parquet')

# Use:
df = pd.read_parquet('../../1_LIB/master/master.parquet')
df = df.sample(frac=0.1, random_state=42)  # Use 10% of data
print(f"Using {len(df):,} rows for analysis")
```

## Check Specific Error

To see the ACTUAL error (not just "kernel died"), run manually:

```bash
cd 3_OUTPUT/3_linear_regression
jupyter notebook linear_regression.ipynb
# Run cells one by one
# The error message will show which cell and why it failed
```

## Common Issues

1. **Out of Memory**: Add `df.sample(frac=0.1)` to reduce dataset size
2. **Missing imports**: Add `pip install <package>` to notebook
3. **File paths wrong**: Use `../../1_LIB/master/master.parquet` for relative paths
4. **Model too complex**: Reduce model complexity or sample data

## Next Steps

1. Open the first failing notebook manually
2. Add the asyncio fix to the first cell
3. Run cells one at a time
4. Fix any errors you encounter
5. Save and try `run_makefile.py` again
