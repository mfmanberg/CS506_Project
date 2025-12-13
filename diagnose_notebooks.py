#!/usr/bin/env python
"""
Diagnose notebook execution issues.
Checks for common problems that cause kernel crashes.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MASTER_PARQUET = PROJECT_ROOT / "1_LIB" / "master" / "master.parquet"

FAILING_NOTEBOOKS = [
    "3_OUTPUT/3_linear_regression/linear_regression.ipynb",
    "3_OUTPUT/3_svr/SVM_Trunc.ipynb",
    "3_OUTPUT/3_svr/SVMDaily.ipynb",
    "3_OUTPUT/3_svr/SVMDailywoutMeso.ipynb",
]

def check_notebook(notebook_path):
    """Check a single notebook for issues."""
    nb_path = PROJECT_ROOT / notebook_path
    
    if not nb_path.exists():
        print(f"❌ {notebook_path} - NOT FOUND")
        return
    
    print(f"\n{'='*70}")
    print(f"📓 {notebook_path}")
    print(f"{'='*70}")
    
    # Read notebook
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    
    print(f"Total cells: {len(cells)}")
    print(f"Code cells: {len(code_cells)}")
    
    # Check for problematic patterns
    issues = []
    
    for i, cell in enumerate(code_cells, 1):
        source = ''.join(cell.get('source', []))
        
        # Check for absolute path issues
        if 'master.parquet' in source:
            if str(MASTER_PARQUET) not in source:
                issues.append(f"Cell {i}: Uses relative path to master.parquet")
        
        # Check for memory-intensive operations
        if 'read_parquet' in source and 'head(' not in source and 'nrows' not in source:
            issues.append(f"Cell {i}: Loads full parquet file (may cause memory issues)")
        
        # Check for large data operations
        if any(keyword in source for keyword in ['groupby', 'pivot', 'merge']):
            if 'master' in source.lower() or 'parquet' in source.lower():
                issues.append(f"Cell {i}: Heavy data operation on full dataset")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} potential issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✓ No obvious issues found")
    
    # Check file size
    size_mb = nb_path.stat().st_size / (1024 * 1024)
    print(f"\nFile size: {size_mb:.2f} MB")
    if size_mb > 50:
        print("   ⚠️  Large file size may indicate embedded outputs")

def main():
    print("="*70)
    print("NOTEBOOK DIAGNOSTICS - Kernel Death Investigation")
    print("="*70)
    
    # Check master.parquet
    print(f"\n📊 Master Data File:")
    if MASTER_PARQUET.exists():
        size_mb = MASTER_PARQUET.stat().st_size / (1024 * 1024)
        print(f"   ✓ Found: {MASTER_PARQUET}")
        print(f"   Size: {size_mb:.2f} MB")
        
        if size_mb > 500:
            print(f"   ⚠️  Very large file - may cause memory issues")
            print(f"   💡 Recommendation: Use data subsampling")
    else:
        print(f"   ❌ Not found: {MASTER_PARQUET}")
        print(f"   💡 Run: python run_makefile.py process")
    
    # Check each failing notebook
    for notebook in FAILING_NOTEBOOKS:
        check_notebook(notebook)
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS:")
    print("="*70)
    print("""
1. Run the path fixer first:
   python fix_notebook_paths.py

2. If master.parquet is very large (>500MB):
   - Consider using data subsampling
   - Add memory limits to notebooks
   
3. Clear notebook outputs to reduce file size:
   jupyter nbconvert --clear-output --inplace <notebook.ipynb>

4. Try running notebooks individually in Jupyter:
   jupyter notebook
   (Then run cells one by one to identify the problematic cell)

5. Check available RAM:
   - Notebooks may need 8GB+ for large datasets
   - Close other applications
   
6. Update the kernel spec to allow more memory:
   Add to first cell: import os; os.environ['JUPYTER_MEMORY_LIMIT']='8G'
""")

if __name__ == "__main__":
    main()
