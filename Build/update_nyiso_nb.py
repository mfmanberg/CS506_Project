#!/usr/bin/env python3
"""Update nyiso_data_exploration.ipynb to use path_utils"""
import json
import sys
from pathlib import Path

nb_path = '2_FIGURES/2_data_exploration/nyiso_data_exploration.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1 - main data loading cell
old_imports = """# Read and aggregate master parquet file
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Find project root by going up from current notebook location
notebook_dir = Path.cwd()
project_root = notebook_dir
while not (project_root / "1_LIB").exists() and project_root.parent != project_root:
    project_root = project_root.parent

# Verify we found the right location
if not (project_root / "1_LIB").exists():
    raise FileNotFoundError("Could not find project root with 1_LIB folder")

MASTER_PATH = project_root / "1_LIB" / "master" / "master.parquet"
print(f"Project root: {project_root}")
print(f"Loading data from: {MASTER_PATH}")"""

new_imports = """# Read and aggregate master parquet file
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Add Build directory to path for path_utils
sys.path.insert(0, str(Path.cwd() / "Build"))
from path_utils import get_project_root, MASTER_PARQUET

# Get project root and master path
project_root = get_project_root()
MASTER_PATH = MASTER_PARQUET

print(f"Project root: {project_root}")
print(f"Loading data from: {MASTER_PATH}")"""

# Cell 14 - another cell with manual path detection
old_path_14 = """from pathlib import Path
project_root = Path(__file__).parent.parent.parent if '__file__' in dir() else Path.cwd().parent.parent
MASTER_PATH = project_root / "1_LIB" / "master" / "master.parquet"""

new_path_14 = """from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "Build"))
from path_utils import get_project_root, MASTER_PARQUET

project_root = get_project_root()
MASTER_PATH = MASTER_PARQUET"""

# Update cells
updated = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Update cell 1
        if old_imports in source:
            new_source = source.replace(old_imports, new_imports)
            cell['source'] = new_source.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                            for i, line in enumerate(cell['source'])]
            print(f"Updated cell {i} (main imports)")
            updated = True
        
        # Update cell 14
        if old_path_14 in source:
            new_source = source.replace(old_path_14, new_path_14)
            cell['source'] = new_source.split('\n')
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                            for i, line in enumerate(cell['source'])]
            print(f"Updated cell {i} (path detection)")
            updated = True

if updated:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"\n✓ Updated {nb_path}")
else:
    print("No updates needed")
