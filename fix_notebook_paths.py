#!/usr/bin/env python
"""
Fix relative paths in notebooks to use absolute paths.
This ensures notebooks can run from any working directory.
"""
import json
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.absolute()
MASTER_PARQUET = PROJECT_ROOT / "1_LIB" / "master" / "master.parquet"

# Notebooks to fix
NOTEBOOKS = [
    "3_OUTPUT/3_linear_regression/linear_regression.ipynb",
    "3_OUTPUT/3_svr/SVM_Trunc.ipynb",
    "3_OUTPUT/3_svr/SVMDaily.ipynb",
    "3_OUTPUT/3_svr/SVMDailywoutMeso.ipynb",
    "3_OUTPUT/3_xg_boost/XGBoost_Testing.ipynb",
]

def fix_notebook(notebook_path):
    """Fix paths in a single notebook."""
    nb_path = PROJECT_ROOT / notebook_path
    
    if not nb_path.exists():
        print(f"⚠ {notebook_path} not found - skipping")
        return False
    
    print(f"→ Fixing {notebook_path}...")
    
    # Read notebook
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    # Process each cell
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        
        source = cell.get('source', [])
        if isinstance(source, str):
            source = [source]
        
        new_source = []
        for line in source:
            original_line = line
            
            # Fix common relative path patterns
            replacements = [
                ('Path("1_LIB/master/master.parquet")', f'Path(r"{MASTER_PARQUET}")'),
                ('Path("../../../1_LIB/master/master.parquet")', f'Path(r"{MASTER_PARQUET}")'),
                ('Path("../../1_LIB/master/master.parquet")', f'Path(r"{MASTER_PARQUET}")'),
                ('"1_LIB/master/master.parquet"', f'r"{MASTER_PARQUET}"'),
                ('"../../../1_LIB/master/master.parquet"', f'r"{MASTER_PARQUET}"'),
                ('"../../1_LIB/master/master.parquet"', f'r"{MASTER_PARQUET}"'),
                ("'1_LIB/master/master.parquet'", f'r"{MASTER_PARQUET}"'),
                ("'../../../1_LIB/master/master.parquet'", f'r"{MASTER_PARQUET}"'),
                ("'../../1_LIB/master/master.parquet'", f'r"{MASTER_PARQUET}"'),
            ]
            
            for old, new in replacements:
                if old in line:
                    line = line.replace(old, new)
                    if line != original_line:
                        modified = True
            
            new_source.append(line)
        
        cell['source'] = new_source
    
    if modified:
        # Write back
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  ✓ Fixed paths in {notebook_path}")
        return True
    else:
        print(f"  ℹ No changes needed for {notebook_path}")
        return False

def main():
    print("=" * 70)
    print("FIXING NOTEBOOK PATHS")
    print("=" * 70)
    print()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Master file: {MASTER_PARQUET}")
    print()
    
    if not MASTER_PARQUET.exists():
        print(f"⚠ WARNING: {MASTER_PARQUET} does not exist!")
        print("  Notebooks will still be updated but may fail to run.")
        print()
    
    fixed_count = 0
    for notebook in NOTEBOOKS:
        if fix_notebook(notebook):
            fixed_count += 1
    
    print()
    print("=" * 70)
    print(f"✓ Processed {len(NOTEBOOKS)} notebooks")
    print(f"✓ Fixed {fixed_count} notebooks")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Review the changes: git diff")
    print("  2. Run notebooks: python run_makefile.py run-analysis")
    print()

if __name__ == "__main__":
    main()
