#!/usr/bin/env python3
import json

nb = json.load(open('/tmp/test_nyiso.ipynb'))
print(f"Total cells: {len(nb['cells'])}")
executed = sum(1 for c in nb['cells'] if c.get('execution_count'))
print(f"Executed: {executed}")
errors = sum(1 for c in nb['cells'] if 'outputs' in c and any('ename' in o for o in c.get('outputs', [])))
print(f"Errors: {errors}")

# Check last few cells
for i in range(max(0, len(nb['cells'])-5), len(nb['cells'])):
    c = nb['cells'][i]
    print(f"\nCell {i}: type={c['cell_type']}, exec_count={c.get('execution_count', 'N/A')}")
    if 'outputs' in c and c['outputs']:
        for o in c['outputs']:
            if 'ename' in o:
                print(f"  ERROR: {o['ename']}: {o.get('evalue', '')[:150]}")
