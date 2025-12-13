"""
Memory management helper for Jupyter notebooks.
Import this at the top of notebooks to enable efficient memory usage.

Usage:
    from notebook_memory_helper import load_data_chunked, clear_memory
    
    # Load data efficiently
    df = load_data_chunked('1_LIB/master/master.parquet', chunksize=10000)
    
    # Clear memory after heavy operations
    clear_memory()
"""

import pandas as pd
import gc
import os


def load_data_chunked(parquet_path, chunksize=None, columns=None, filters=None):
    """
    Load parquet file with memory-efficient options.
    
    Args:
        parquet_path: Path to parquet file
        chunksize: If provided, load in chunks and return iterator
        columns: List of columns to load (None = all)
        filters: PyArrow filters to apply
    
    Returns:
        DataFrame or iterator of DataFrames
    """
    if chunksize:
        # Return iterator for chunk processing
        return pd.read_parquet(
            parquet_path,
            columns=columns,
            filters=filters,
            engine='pyarrow'
        )
    else:
        # Load full file with optimizations
        df = pd.read_parquet(
            parquet_path,
            columns=columns,
            filters=filters,
            engine='pyarrow'
        )
        
        # Optimize dtypes to reduce memory
        df = optimize_dtypes(df)
        return df


def optimize_dtypes(df):
    """
    Optimize DataFrame dtypes to reduce memory usage.
    
    - Downcast numeric columns to smallest possible type
    - Convert object columns to category if beneficial
    """
    # Downcast integers
    int_cols = df.select_dtypes(include=['int']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Downcast floats
    float_cols = df.select_dtypes(include=['float']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert low-cardinality object columns to category
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    return df


def clear_memory():
    """
    Force garbage collection and clear memory.
    Call this after deleting large objects.
    """
    gc.collect()
    
    # Also try to return memory to OS (Unix-like systems)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass  # Skip on Windows or if not available


def get_memory_usage():
    """
    Get current memory usage of the process.
    
    Returns:
        dict with memory info in MB
    """
    import psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
    }


def print_memory_usage(label=""):
    """Print current memory usage."""
    try:
        mem = get_memory_usage()
        print(f"Memory usage {label}: {mem['rss_mb']:.1f} MB (RSS), {mem['vms_mb']:.1f} MB (VMS)")
    except ImportError:
        print("Install psutil to monitor memory: pip install psutil")


# Context manager for memory tracking
class MemoryTracker:
    """
    Context manager to track memory usage of a code block.
    
    Usage:
        with MemoryTracker("Loading data"):
            df = pd.read_parquet('data.parquet')
    """
    def __init__(self, label=""):
        self.label = label
        self.start_mem = None
    
    def __enter__(self):
        clear_memory()
        try:
            self.start_mem = get_memory_usage()
            print(f"[{self.label}] Starting - Memory: {self.start_mem['rss_mb']:.1f} MB")
        except:
            pass
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            end_mem = get_memory_usage()
            diff = end_mem['rss_mb'] - self.start_mem['rss_mb']
            print(f"[{self.label}] Completed - Memory: {end_mem['rss_mb']:.1f} MB (Δ {diff:+.1f} MB)")
        except:
            pass
        clear_memory()
