#!/usr/bin/env python3
"""Test path_utils import in notebook context"""
import sys
from pathlib import Path

# Simulate notebook import
sys.path.insert(0, str(Path.cwd() / "Build"))

try:
    from path_utils import get_project_root, MASTER_PARQUET
    print("✓ Import successful")
    print(f"✓ Project root: {get_project_root()}")
    print(f"✓ Master parquet: {MASTER_PARQUET}")
    print(f"✓ File exists: {MASTER_PARQUET.exists()}")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All checks passed")
