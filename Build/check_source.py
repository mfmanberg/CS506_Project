#!/usr/bin/env python3
import json

nb = json.load(open('2_FIGURES/2_data_exploration/nyiso_data_exploration.ipynb', encoding='utf-8'))
src = nb['cells'][1]['source']
print(f'Source type: {type(src)}')
print(f'Source length: {len(src)}')
print(f'First item type: {type(src[0]) if src else "N/A"}')
print('First 5 lines:')
for i, line in enumerate(src[:5]):
    print(f'  {i}: {repr(line)}')
