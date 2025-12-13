#!/usr/bin/env python3
r"""
Export all figures from notebooks to 2_FIGURES\FIGURES directory.

This script executes all notebooks that generate figures and exports them
to a central location with standardized naming.

USAGE:
------
From project root:
    python Build/export_all_figures.py

REQUIREMENTS:
-------------
- jupyter
- nbconvert
- nbformat
- matplotlib
- seaborn
- pandas
- numpy
- scikit-learn
- xgboost
- joblib

Install via: pip install -r Dependencies/requirements.txt
"""

import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Define absolute paths
FIGURES_OUTPUT_DIR = PROJECT_ROOT / "2_FIGURES" / "FIGURES"
BUILD_DIR = PROJECT_ROOT / "Build"

# Ensure output directory exists
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def check_dependencies():
    """Check if required packages are installed."""
    required = {
        'jupyter': 'jupyter',
        'nbformat': 'nbformat',
        'nbconvert': 'nbconvert',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'joblib': 'joblib'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("ERROR: Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall dependencies:")
        print("  pip install -r Dependencies/requirements.txt")
        sys.exit(1)

check_dependencies()

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for saving
matplotlib.use('Agg')

def execute_notebook_and_export(notebook_path, output_prefix):
    """
    Execute a notebook and save generated figures.
    
    Args:
        notebook_path: Path to notebook file
        output_prefix: Prefix for output filenames
    
    Returns:
        List of exported figure paths
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"  ⚠ Notebook not found: {notebook_path}")
        return []
    
    print(f"  📓 Processing: {notebook_path.name}")
    
    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Configure execution
        ep = ExecutePreprocessor(
            timeout=600,  # 10 minutes max per cell
            kernel_name='python3',
            allow_errors=False
        )
        
        # Execute notebook in its directory
        notebook_dir = notebook_path.parent
        original_dir = os.getcwd()
        
        try:
            os.chdir(notebook_dir)
            ep.preprocess(nb, {'metadata': {'path': str(notebook_dir)}})
        finally:
            os.chdir(original_dir)
        
        # Find and copy generated images
        exported_files = []
        
        # Look for saved images in notebook directory
        for img_file in notebook_dir.glob('*.png'):
            if img_file.is_file():
                # Create new filename with prefix
                new_name = f"{output_prefix}_{img_file.name}"
                dest_path = FIGURES_OUTPUT_DIR / new_name
                shutil.copy2(img_file, dest_path)
                exported_files.append(str(dest_path.relative_to(PROJECT_ROOT)))
                print(f"    ✓ Exported: {new_name}")
        
        # Also check for any .jpg or .jpeg files
        for img_file in notebook_dir.glob('*.jpg'):
            if img_file.is_file():
                new_name = f"{output_prefix}_{img_file.name}"
                dest_path = FIGURES_OUTPUT_DIR / new_name
                shutil.copy2(img_file, dest_path)
                exported_files.append(str(dest_path.relative_to(PROJECT_ROOT)))
                print(f"    ✓ Exported: {new_name}")
        
        return exported_files
        
    except Exception as e:
        print(f"    ✗ Error executing notebook: {e}")
        return []

def copy_existing_images():
    """Copy existing images from various directories."""
    print("\n📁 Copying existing images...")
    
    existing_images = [
        # Images in 4_VAULT
        (PROJECT_ROOT / "4_VAULT" / "NY_Zones.png", "nyiso_zones.png"),
        (PROJECT_ROOT / "4_VAULT" / "TotalLoad2023Day15Min.png", "total_load_2023_15min.png"),
        (PROJECT_ROOT / "4_VAULT" / "DayByDayJan2023.png", "day_by_day_jan_2023.png"),
        (PROJECT_ROOT / "4_VAULT" / "Load_With_Losses.png", "load_with_losses.png"),
    ]
    
    copied = []
    for src, dest_name in existing_images:
        if src.exists():
            dest = FIGURES_OUTPUT_DIR / dest_name
            shutil.copy2(src, dest)
            copied.append(str(dest.relative_to(PROJECT_ROOT)))
            print(f"  ✓ Copied: {dest_name}")
        else:
            print(f"  ⚠ Not found: {src}")
    
    return copied

def main():
    """Main execution function."""
    print("=" * 70)
    print("EXPORTING ALL FIGURES FROM NOTEBOOKS")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Output Directory: {FIGURES_OUTPUT_DIR}")
    print("=" * 70)
    
    all_exported = []
    
    # Copy existing static images first
    all_exported.extend(copy_existing_images())
    
    # Define notebooks to process
    notebooks_to_process = [
        # Data exploration
        {
            'path': PROJECT_ROOT / "2_FIGURES" / "2_data_exploration" / "nyiso_data_exploration.ipynb",
            'prefix': "data_exploration"
        },
        
        # XGBoost models
        {
            'path': PROJECT_ROOT / "3_OUTPUT" / "3_xg_boost" / "XGBoost_Testing_output.ipynb",
            'prefix': "xgboost_testing"
        },
        {
            'path': PROJECT_ROOT / "3_OUTPUT" / "3_xg_boost" / "ComparisonMetrics.ipynb",
            'prefix': "model_comparison"
        },
        
        # SVR models
        {
            'path': PROJECT_ROOT / "3_OUTPUT" / "3_svr" / "SVMDaily.ipynb",
            'prefix': "svr_daily"
        },
        {
            'path': PROJECT_ROOT / "3_OUTPUT" / "3_svr" / "SVMHourly.ipynb",
            'prefix': "svr_hourly"
        },
        {
            'path': PROJECT_ROOT / "3_OUTPUT" / "3_svr" / "SVM15Min.ipynb",
            'prefix': "svr_15min"
        },
        
        # Linear Regression
        {
            'path': PROJECT_ROOT / "3_OUTPUT" / "3_linear_regression" / "linear_regression_output.ipynb",
            'prefix': "linear_regression"
        },
    ]
    
    print("\n📊 Executing notebooks and exporting figures...")
    
    for nb_info in notebooks_to_process:
        print(f"\n{nb_info['prefix'].upper()}")
        exported = execute_notebook_and_export(
            nb_info['path'],
            nb_info['prefix']
        )
        all_exported.extend(exported)
    
    # Copy SVR animations if they exist
    print("\n🎬 Copying SVR animations...")
    svr_anim_src = FIGURES_OUTPUT_DIR / "svr_animations"
    if svr_anim_src.exists():
        for gif_file in svr_anim_src.glob("*.gif"):
            # They're already in the right place
            all_exported.append(str(gif_file.relative_to(PROJECT_ROOT)))
            print(f"  ✓ Found: {gif_file.name}")
    
    # Create summary report
    print("\n" + "=" * 70)
    print(f"✅ EXPORT COMPLETE - {len(all_exported)} files")
    print("=" * 70)
    
    # Save manifest
    manifest = {
        'export_date': datetime.now().isoformat(),
        'project_root': str(PROJECT_ROOT),
        'output_directory': str(FIGURES_OUTPUT_DIR),
        'total_files': len(all_exported),
        'files': all_exported
    }
    
    manifest_path = FIGURES_OUTPUT_DIR / "export_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📄 Manifest saved to: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"\n📂 All figures available in: {FIGURES_OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    
    return all_exported

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Export interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
