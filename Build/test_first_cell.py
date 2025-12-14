#!/usr/bin/env python3
"""Test if first cell of nyiso notebook works"""
import sys
from pathlib import Path

# Add Build directory to path for path_utils
sys.path.insert(0, str(Path.cwd() / "Build"))
from path_utils import get_project_root, MASTER_PARQUET

# Get project root and master path
project_root = get_project_root()
MASTER_PATH = MASTER_PARQUET

print(f"Project root: {project_root}")
print(f"Loading data from: {MASTER_PATH}")
print("SUCCESS!")
