#!/usr/bin/env python3
import json

try:
    with open('/tmp/papermill_output/nyiso_data_exploration.ipynb', 'r') as f:
        nb = json.load(f)
    
    print(f"Total cells: {len(nb['cells'])}")
    
    for i, c in enumerate(nb['cells']):
        if 'outputs' in c and c.get('outputs'):
            for o in c['outputs']:
                if 'ename' in o:
                    print(f"\n=== ERROR IN CELL {i} ===")
                    print(f"Error type: {o['ename']}")
                    print(f"Error value: {o.get('evalue', 'N/A')}")
                    if 'traceback' in o:
                        print("Traceback:")
                        for line in o['traceback'][:15]:
                            print(line)
                    break
except FileNotFoundError:
    print("Notebook not found - may not have been created yet")
except Exception as e:
    print(f"Error reading notebook: {e}")
