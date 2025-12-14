#!/usr/bin/env python3
"""Check notebook for path_utils usage"""
import json
import sys
from pathlib import Path

nb_path = sys.argv[1] if len(sys.argv) > 1 else '2_FIGURES/2_data_exploration/nyiso_data_exploration.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

print(f"Notebook: {nb_path}")
print(f"Total cells: {len(nb['cells'])}\n")

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    
    # Check for path-related code
    if 'project_root' in source or 'path_utils' in source or 'MASTER_PATH' in source:
        print(f"Cell {i} ({cell['cell_type']}):")
        print(source[:500])
        print("---\n")
