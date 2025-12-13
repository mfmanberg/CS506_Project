# Memory-Efficient Notebook Execution

## Using the Full Dataset Without Crashes

The project now uses the **full master.parquet** dataset with built-in memory management.

### 🔧 How It Works

1. **Memory-optimized subprocess execution** - Uses environment variables to enable aggressive memory cleanup
2. **Helper library available** - Import `notebook_memory_helper.py` in notebooks for memory utilities

### 📝 Using Memory Helper in Notebooks

Add this to the top of any notebook:

```python
# Memory-efficient data loading
from notebook_memory_helper import load_data_chunked, clear_memory, MemoryTracker

# Load data with optimized dtypes
with MemoryTracker("Loading master data"):
    df = load_data_chunked('../../1_LIB/master/master.parquet')

# Your analysis here...

# Clean up memory after heavy operations
del large_variable
clear_memory()
```

### 🚀 Key Functions

**`load_data_chunked(path, columns=None)`**
- Loads parquet file with memory optimization
- Automatically downcasts numeric types
- Converts low-cardinality columns to category dtype
- Optionally load specific columns only

**`clear_memory()`**
- Forces garbage collection
- Returns memory to OS (where supported)

**`MemoryTracker(label)`**
- Context manager to track memory usage
- Shows memory before/after operations

**`optimize_dtypes(df)`**
- Reduces DataFrame memory footprint
- Safe for existing data

### 💡 Best Practices

1. **Load only needed columns:**
   ```python
   df = load_data_chunked('master.parquet', columns=['datetime', 'temp_2m', 'Load'])
   ```

2. **Delete unused variables:**
   ```python
   del df_intermediate
   clear_memory()
   ```

3. **Use chunking for very large operations:**
   ```python
   for chunk in pd.read_parquet('master.parquet', chunksize=10000):
       process(chunk)
   ```

4. **Track memory in critical sections:**
   ```python
   with MemoryTracker("Training model"):
       model.fit(X_train, y_train)
   ```

### ⚙️ System Requirements

- **RAM:** 8+ GB recommended for full dataset
- **Packages:** Install `psutil` for memory monitoring
  ```bash
  pip install psutil
  ```

### 🎯 Running Notebooks

The makefile script now automatically:
- ✅ Clears outputs before execution
- ✅ Uses memory-optimized environment
- ✅ Supports full dataset processing
- ✅ Provides detailed output summary

```bash
python run_makefile.py run-analysis
```
