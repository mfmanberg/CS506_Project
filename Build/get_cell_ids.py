#!/usr/bin/env python3
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells'][:20]):
    print(f"Cell {i}: ID={cell.get('id', 'NO_ID')}, type={cell['cell_type']}, lines={len(cell.get('source', []))}")
