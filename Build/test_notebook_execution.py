#!/usr/bin/env python3
"""Test script to diagnose notebook execution issues"""

import sys
import pandas as pd
from pathlib import Path

# Add Build directory to path
sys.path.insert(0, str(Path(__file__).parent))

from path_utils import get_project_root

def main():
    print("="*60)
    print("Testing Notebook Execution Environment")
    print("="*60)
    
    # Test 1: Path detection
    print("\n1. Testing path_utils...")
    try:
        project_root = get_project_root()
        print(f"   ✓ Project root: {project_root}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 2: Master parquet file access
    print("\n2. Testing master.parquet access...")
    try:
        master_path = project_root / "1_LIB" / "master" / "master.parquet"
        print(f"   File path: {master_path}")
        print(f"   File exists: {master_path.exists()}")
        if master_path.exists():
            file_size_mb = master_path.stat().st_size / (1024 * 1024)
            print(f"   File size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 3: Load master parquet
    print("\n3. Testing parquet loading...")
    try:
        df = pd.read_parquet(master_path)
        print(f"   ✓ Loaded successfully")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {list(df.columns)[:5]}...")  # First 5 columns
    except Exception as e:
        print(f"   ✗ Error loading parquet: {e}")
        return False
    
    # Test 4: Memory usage
    print("\n4. Testing memory usage...")
    try:
        mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"   DataFrame memory: {mem_usage_mb:.2f} MB")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
