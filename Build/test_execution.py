"""Test script to verify all notebook cells are executed.

Usage:
    python test_execution.py           # Check execution status
    python test_execution.py --clean   # Remove error cells
"""
import json
from pathlib import Path

notebooks = [
    '3_OUTPUT/3_linear_regression/linear_regression.ipynb',
    '3_OUTPUT/3_svr/SVM_Trunc.ipynb',
    '3_OUTPUT/3_svr/SVMDaily.ipynb',
    '3_OUTPUT/3_svr/SVMDailywoutMeso.ipynb',
    '3_OUTPUT/3_xg_boost/ComparisonMetrics.ipynb',
    '3_OUTPUT/3_xg_boost/XGBoost_PostMid.ipynb',
    '3_OUTPUT/3_xg_boost/XGBoost_Testing.ipynb',
]

print("=" * 80)
print("NOTEBOOK EXECUTION STATUS")
print("=" * 80)
print()

all_complete = True

for nb_path in notebooks:
    nb = json.load(open(nb_path))
    name = Path(nb_path).name
    
    code_cells = [c for c in nb['cells'] if c.get('cell_type') == 'code']
    cells_with_outputs = [c for c in code_cells if c.get('outputs')]
    
    total = len(code_cells)
    executed = len(cells_with_outputs)
    
    status = "✓ COMPLETE" if executed == total else "✗ INCOMPLETE"
    if executed != total:
        all_complete = False
    
    print(f"{status:15} {name:30} {executed:2}/{total:2} cells")
    
    # Show which cells are missing outputs
    if executed != total:
        empty_indices = [i+1 for i, c in enumerate(code_cells) if not c.get('outputs')]
        print(f"               Missing outputs in cells: {empty_indices}")

print()
print("=" * 80)
if all_complete:
    print("✓ ALL NOTEBOOKS FULLY EXECUTED")
else:
    print("✗ Some notebooks have cells without outputs")
print("=" * 80)
