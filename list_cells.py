;import json

with open('3_OUTPUT/3_svr/SVMDaily.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source'])[:80]
    print(f"Cell {i}: {source.replace(chr(10), ' ')}")
