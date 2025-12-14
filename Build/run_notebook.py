"""
Run a single notebook using papermill
Usage: python run_notebook.py <notebook_path>
"""
import sys
import os
from pathlib import Path
import subprocess

def get_project_root():
    """Get project root from script location."""
    return Path(__file__).parent.parent.resolve()

def run_notebook(notebook_path, project_root):
    """Execute notebook with papermill."""
    nb_path = project_root / notebook_path
    
    if not nb_path.exists():
        print(f"ERROR: Notebook not found: {nb_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Executing: {notebook_path}")
    print(f"{'='*60}\n")
    
    # Use temporary file for output
    temp_output = nb_path.parent / f"{nb_path.stem}_temp.ipynb"
    
    try:
        # Use conda environment or system python
        import shutil
        
        # Find python and papermill in current environment
        venv_python = shutil.which("python")
        venv_papermill = shutil.which("papermill")
        
        if not venv_python:
            print(f"ERROR: Python not found in PATH. Please activate anaconda3 environment.")
            return False
        
        venv_python = Path(venv_python)
        
        if not venv_papermill:
            print(f"Papermill not found. Installing...")
            subprocess.run([str(venv_python), "-m", "pip", "install", "papermill"], check=True)
            venv_papermill = shutil.which("papermill")
            if not venv_papermill:
                print(f"ERROR: Papermill installation failed")
                return False
        
        venv_papermill = Path(venv_papermill)
            
        # Set PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{project_root};{project_root / 'Build'}"
        
        # Run papermill
        cmd = [
            str(venv_papermill),
            str(nb_path),
            str(temp_output),
            "--kernel", "python3",
            "--execution-timeout", "3600",
            "--cwd", str(project_root)
        ]
        
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        
        if result.returncode == 0:
            # Replace original with executed version
            temp_output.replace(nb_path)
            print(f"\n{'='*60}")
            print(f"SUCCESS: {notebook_path}")
            print(f"{'='*60}\n")
            return True
        else:
            print(f"\n{'='*60}")
            print(f"FAILED: {notebook_path}")
            print(f"{'='*60}\n")
            if temp_output.exists():
                temp_output.unlink()
            return False
            
    except Exception as e:
        print(f"\nERROR: {e}")
        if temp_output.exists():
            temp_output.unlink()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <notebook_path>")
        print("Example: python run_notebook.py 2_FIGURES/2_data_exploration/nyiso_data_exploration.ipynb")
        sys.exit(1)
    
    project_root = get_project_root()
    notebook_path = sys.argv[1]
    
    success = run_notebook(notebook_path, project_root)
    sys.exit(0 if success else 1)
