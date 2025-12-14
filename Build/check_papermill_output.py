#!/usr/bin/env python3
import json
import os

nb_path = '/tmp/test_nyiso2.ipynb'
if not os.path.exists(nb_path):
    print(f"Notebook not found at {nb_path}")
    exit(1)

with open(nb_path) as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}\n")

for i, c in enumerate(nb['cells'][:10]):
    exec_count = c.get('execution_count')
    has_output = len(c.get('outputs', [])) > 0
    cell_type = c['cell_type']
    
    print(f"Cell {i}: type={cell_type}, exec_count={exec_count}, has_output={has_output}")
    
    if has_output and cell_type == 'code':
        for o in c['outputs']:
            if 'ename' in o:
                print(f"  ERROR: {o['ename']}")
                print(f"  VALUE: {o.get('evalue', 'N/A')[:200]}")
                if 'traceback' in o and len(o['traceback']) > 0:
                    print("  TRACEBACK (first 5 lines):")
                    for line in o['traceback'][:5]:
                        # Strip ANSI codes
                        import re
                        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                        print(f"    {clean}")
