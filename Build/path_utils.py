"""
Path utilities for cross-platform compatibility (Windows/WSL/Linux).
Automatically detects environment and converts paths appropriately.
"""
import os
import sys
from pathlib import Path


def get_project_root():
    """
    Get the project root directory, works in both Windows and WSL.
    Returns absolute path in the format appropriate for the current environment.
    """
    # Try to get the current file's directory
    if '__file__' in globals():
        current = Path(__file__).parent.absolute()
        # If we're in the Build subdirectory, go up one level
        if current.name == "Build" or current.name == "Makefile":
            return current.parent
        return current
    
    # Fallback: use current working directory
    cwd = Path.cwd()
    
    # If we're in a subdirectory, try to find project root
    # (look for characteristic files/folders)
    current = cwd
    while current != current.parent:
        if (current / 'Dependencies').exists() or (current / '1_LIB').exists():
            return current
        current = current.parent
    
    # If not found, return cwd
    return cwd


def is_wsl():
    """Check if running in WSL environment."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False


def normalize_path(path_str):
    """
    Convert any path format to the appropriate format for current environment.
    
    Args:
        path_str: Path as string (can be Windows or Unix format)
    
    Returns:
        Path object in the correct format for current environment
    """
    # Convert to string if Path object
    if isinstance(path_str, Path):
        path_str = str(path_str)
    
    # Replace backslashes with forward slashes for consistency
    path_str = path_str.replace('\\', '/')
    
    # If in WSL and path starts with Windows drive letter
    if is_wsl():
        # Convert c:/ or c:\ to /mnt/c/
        if len(path_str) >= 2 and path_str[1] == ':':
            drive = path_str[0].lower()
            rest = path_str[2:].lstrip('/')
            return Path(f'/mnt/{drive}/{rest}')
    
    # Return as Path object
    return Path(path_str)


def get_master_parquet_path():
    """Get the path to master.parquet file."""
    root = get_project_root()
    master_path = root / '1_LIB' / 'master' / 'master.parquet'
    return master_path


def get_data_path(*subdirs):
    """
    Get path to data directory with optional subdirectories.
    
    Args:
        *subdirs: Optional subdirectory names
    
    Returns:
        Path object
    """
    root = get_project_root()
    path = root / '1_LIB'
    for subdir in subdirs:
        path = path / subdir
    return path


def get_output_path(*subdirs):
    """
    Get path to output directory with optional subdirectories.
    
    Args:
        *subdirs: Optional subdirectory names
    
    Returns:
        Path object
    """
    root = get_project_root()
    path = root / '3_OUTPUT'
    for subdir in subdirs:
        path = path / subdir
    return path


# Convenience constants
PROJECT_ROOT = get_project_root()
MASTER_PARQUET = get_master_parquet_path()
DATA_DIR = PROJECT_ROOT / '1_LIB'
OUTPUT_DIR = PROJECT_ROOT / '3_OUTPUT'


def clean_notebooks():
    """Remove papermill error cells from notebooks."""
    import json
    
    notebooks = [
        PROJECT_ROOT / '3_OUTPUT' / '3_linear_regression' / 'linear_regression.ipynb',
        PROJECT_ROOT / '3_OUTPUT' / '3_svr' / 'SVM_Trunc.ipynb',
        PROJECT_ROOT / '3_OUTPUT' / '3_svr' / 'SVMDaily.ipynb',
        PROJECT_ROOT / '3_OUTPUT' / '3_svr' / 'SVMDailywoutMeso.ipynb',
        PROJECT_ROOT / '3_OUTPUT' / '3_xg_boost' / 'ComparisonMetrics.ipynb',
        PROJECT_ROOT / '3_OUTPUT' / '3_xg_boost' / 'XGBoost_PostMid.ipynb',
        PROJECT_ROOT / '3_OUTPUT' / '3_xg_boost' / 'XGBoost_Testing.ipynb',
    ]
    
    for nb_path in notebooks:
        if nb_path.exists():
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            original_count = len(nb['cells'])
            nb['cells'] = [cell for cell in nb['cells'] 
                           if not (cell.get('cell_type') == 'markdown' and 
                                   ('papermill-error-cell' in ''.join(cell.get('source', [])) or
                                    'An Exception was encountered' in ''.join(cell.get('source', []))))]
            
            if len(nb['cells']) < original_count:
                with open(nb_path, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=1)
                print(f"Cleaned: {nb_path.name}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        clean_notebooks()
    else:
        print(f"Running in WSL: {is_wsl()}")
        print(f"Project Root: {PROJECT_ROOT}")
        print(f"Master Parquet: {MASTER_PARQUET}")
        print(f"Master Parquet exists: {MASTER_PARQUET.exists()}")
        print(f"Data Directory: {DATA_DIR}")
        print(f"Output Directory: {OUTPUT_DIR}")
