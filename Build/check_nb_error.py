#!/usr/bin/env python3
import json
import sys

nb = json.load(open('/tmp/test_nyiso.ipynb'))
error_cells = [c for c in nb['cells'] if 'outputs' in c and any('traceback' in o or 'ename' in o for o in c['outputs'])]
print(f'Error cells: {len(error_cells)}')
for i, c in enumerate(error_cells[:3]):
    for o in c['outputs']:
        if 'ename' in o:
            print(f"\nCell error {i}:")
            print(f"  Type: {o.get('ename', 'UNKNOWN')}")
            print(f"  Value: {o.get('evalue', '')[:300]}")
            if 'traceback' in o:
                print(f"  Traceback:")
                for line in o['traceback'][:10]:
                    print(f"    {line}")
