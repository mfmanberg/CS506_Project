"""Test if cell 2 from notebook will work"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Simulate notebook execution
import os
os.chdir(r"c:\Users\Matt\Desktop\CS506\CS506_Project\2_FIGURES\2_data_exploration")

# Find project root
notebook_dir = Path.cwd()
project_root = notebook_dir
print(f"Starting from: {notebook_dir}")

while not (project_root / "1_LIB").exists() and project_root.parent != project_root:
    print(f"  Checking: {project_root}")
    project_root = project_root.parent

if not (project_root / "1_LIB").exists():
    raise FileNotFoundError("Could not find project root with 1_LIB folder")

MASTER_PATH = project_root / "1_LIB" / "master" / "master.parquet"
print(f"\nProject root: {project_root}")
print(f"Master path: {MASTER_PATH}")
print(f"File exists: {MASTER_PATH.exists()}")

if MASTER_PATH.exists():
    print("\n✓ All path finding works correctly!")
    print("\nThe notebook should run if:")
    print("  1. You've selected the 'Python (CS506 venv)' kernel")
    print("  2. The kernel has started (look for kernel indicator in top-right)")
    print("  3. You click the ▶ button to run cell 2")
else:
    print("\n✗ Problem: master.parquet not found")
