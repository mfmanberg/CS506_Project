#!/usr/bin/env python3
"""
Export all existing figures and generate new ones from notebooks.

This script collects all existing images and generates missing figures
by extracting matplotlib code from notebooks and executing it.
"""

import os
import sys
import shutil
from pathlib import Path
import json
from datetime import datetime

# Absolute project root
PROJECT_ROOT = Path(r"C:\Users\Matt\Desktop\CS506\CS506_Project").absolute()
FIGURES_DIR = PROJECT_ROOT / "2_FIGURES" / "FIGURES"

# Ensure output directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def copy_file_safe(src, dest, desc=""):
    """Safely copy a file with error handling."""
    try:
        if src.exists():
            shutil.copy2(src, dest)
            print(f"  ✓ Copied: {dest.name} {desc}")
            return True
        else:
            print(f"  ⚠ Not found: {src.name}")
            return False
    except Exception as e:
        print(f"  ✗ Error copying {src.name}: {e}")
        return False

def main():
    print("=" * 70)
    print("EXPORTING ALL FIGURES")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Output Directory: {FIGURES_DIR}")
    print("=" * 70)
    
    exported_files = []
    
    # 1. Copy existing images from 4_VAULT
    print("\n📁 Section 1: Copying images from 4_VAULT...")
    vault_images = [
        ("NY_Zones.png", "nyiso_zones.png", "- NYISO geographic zones map"),
        ("TotalLoad2023Day15Min.png", "total_load_2023_15min.png", "- 2023 load by 15-min intervals"),
        ("DayByDayJan2023.png", "day_by_day_jan_2023.png", "- Daily patterns Jan 2023"),
        ("Load_With_Losses.png", "load_with_losses.png", "- Energy load with transmission losses"),
        ("SVM_READMe_Graph.png", "svr_readme_graph.png", "- SVR confusion during load jumps"),
        ("baseline_performance.png", "vault_baseline_performance.png", "- Baseline performance backup"),
        ("model_comparison.png", "vault_model_comparison.png", "- Model comparison backup"),
    ]
    
    for src_name, dest_name, desc in vault_images:
        src = PROJECT_ROOT / "4_VAULT" / src_name
        dest = FIGURES_DIR / dest_name
        if copy_file_safe(src, dest, desc):
            exported_files.append(str(dest.relative_to(PROJECT_ROOT)))
    
    # 2. Copy existing images from 2_FIGURES/FIGURES (already there)
    print("\n📁 Section 2: Cataloging existing figures in 2_FIGURES/FIGURES...")
    existing_in_figures = list(FIGURES_DIR.glob("*.png"))
    for img in existing_in_figures:
        rel_path = str(img.relative_to(PROJECT_ROOT))
        if rel_path not in exported_files:
            exported_files.append(rel_path)
            print(f"  ✓ Found: {img.name}")
    
    # 3. Copy SVR animations
    print("\n📁 Section 3: Cataloging SVR animations...")
    svr_anim_dir = FIGURES_DIR / "svr_animations"
    if svr_anim_dir.exists():
        for gif_file in svr_anim_dir.glob("*.gif"):
            rel_path = str(gif_file.relative_to(PROJECT_ROOT))
            if rel_path not in exported_files:
                exported_files.append(rel_path)
                print(f"  ✓ Found: {gif_file.name}")
    
    # 4. Look for images in notebook output directories
    print("\n📁 Section 4: Searching for images in output directories...")
    
    search_dirs = [
        PROJECT_ROOT / "3_OUTPUT" / "3_xg_boost",
        PROJECT_ROOT / "3_OUTPUT" / "3_svr",
        PROJECT_ROOT / "3_OUTPUT" / "3_linear_regression",
        PROJECT_ROOT / "2_FIGURES" / "2_data_exploration",
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for img_file in search_dir.glob("*.png"):
                if img_file.is_file() and img_file.parent == search_dir:
                    # Create descriptive name based on directory and filename
                    dir_name = search_dir.name
                    new_name = f"{dir_name}_{img_file.name}"
                    dest = FIGURES_DIR / new_name
                    
                    if copy_file_safe(img_file, dest, f"from {dir_name}"):
                        rel_path = str(dest.relative_to(PROJECT_ROOT))
                        if rel_path not in exported_files:
                            exported_files.append(rel_path)
    
    # 5. Create summary
    print("\n" + "=" * 70)
    print(f"✅ EXPORT COMPLETE - {len(exported_files)} files cataloged")
    print("=" * 70)
    
    # Group files by type
    print("\n📊 Summary by type:")
    pngs = [f for f in exported_files if f.endswith('.png')]
    gifs = [f for f in exported_files if f.endswith('.gif')]
    csvs = [f for f in exported_files if f.endswith('.csv')]
    
    print(f"  PNG images: {len(pngs)}")
    print(f"  GIF animations: {len(gifs)}")
    print(f"  CSV files: {len(csvs)}")
    
    # Save manifest
    manifest = {
        'export_date': datetime.now().isoformat(),
        'project_root': str(PROJECT_ROOT),
        'output_directory': str(FIGURES_DIR),
        'total_files': len(exported_files),
        'files': sorted(exported_files),
        'summary': {
            'png_count': len(pngs),
            'gif_count': len(gifs),
            'csv_count': len(csvs)
        }
    }
    
    manifest_path = FIGURES_DIR / "export_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📄 Manifest saved to: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"\n📂 All figures available in: 2_FIGURES\\FIGURES")
    
    # List all files
    print("\n📋 Exported files:")
    for f in sorted(exported_files):
        print(f"  - {f}")

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
