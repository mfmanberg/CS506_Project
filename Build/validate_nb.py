#!/usr/bin/env python3
import json
import sys

try:
    with open("2_FIGURES/2_data_exploration/nyiso_data_exploration.ipynb", encoding='utf-8') as f:
        nb = json.load(f)
    print("JSON is valid")
    print(f"Cells: {len(nb['cells'])}")
    print(f"Cell 1 type: {nb['cells'][1]['cell_type']}")
    print(f"Cell 1 source lines: {len(nb['cells'][1]['source'])}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
