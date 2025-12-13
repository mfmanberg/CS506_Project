#!/usr/bin/env python
"""
Run Makefile targets in the current Python environment.
This script parses the Makefile and executes targets directly using Python.
"""
import os
import sys
import subprocess
import glob

# Makefile configuration (from your Makefile)
ENABLE_TIMEOUT = True
TIMEOUT_SECONDS = 1200

MASTER_PARQUET = r"1_LIB\master\master.parquet"
FIRST_PASS_NOTEBOOK = r"2_FIGURES\1_data_wrangling\1st_pass.ipynb"
ANALYSIS_NOTEBOOKS = [
    r"3_OUTPUT\3_linear_regression\linear_regression.ipynb",
    r"3_OUTPUT\3_svr\SVM_Trunc.ipynb",
    r"3_OUTPUT\3_svr\SVMDaily.ipynb",
    r"3_OUTPUT\3_svr\SVMDailywoutMeso.ipynb",
    r"3_OUTPUT\3_xg_boost\ComparisonMetrics.ipynb",
    r"3_OUTPUT\3_xg_boost\XGBoost_PostMid.ipynb",
    r"3_OUTPUT\3_xg_boost\XGBoost_Testing.ipynb",
]
COMPLETION_DIR = ".make_completion"


def get_system_memory():
    """Get total system memory in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        return total_gb, available_gb
    except ImportError:
        return None, None


def check_master():
    """Check if master.parquet exists."""
    print()
    if os.path.exists(MASTER_PARQUET):
        print(f"✓ master.parquet already exists at {MASTER_PARQUET}")
        print("✓ Data wrangling complete - skipping 1st_pass.ipynb")
        print("  To reprocess, run: python run_makefile.py clean-master")
    else:
        print("✗ master.parquet not found")
        print("→ Run 'python run_makefile.py process' to execute data wrangling")
    print()


def process():
    """Run data wrangling notebook to create master.parquet."""
    print()
    if os.path.exists(MASTER_PARQUET):
        print("⚠ master.parquet already exists")
        print("  Run 'python run_makefile.py clean-master' first to reprocess")
        return 1
    
    print("→ Running data wrangling notebook...")
    print("⚠ This will execute heavy computations")
    
    os.makedirs(COMPLETION_DIR, exist_ok=True)
    
    try:
        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            FIRST_PASS_NOTEBOOK
        ], check=True)
        
        print("✓ Data wrangling complete")
        
        # Create completion marker
        with open(os.path.join(COMPLETION_DIR, "1st_pass.done"), 'w') as f:
            f.write("")
        
        return 0
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running notebook: {e}")
        return 1


def clear_notebook_outputs():
    """Clear outputs from all analysis notebooks to reduce file size."""
    print()
    print("=== Clearing Notebook Outputs ===")
    
    for notebook in ANALYSIS_NOTEBOOKS:
        if not os.path.exists(notebook):
            continue
        
        print(f"→ Clearing {os.path.basename(notebook)}...")
        try:
            subprocess.run([
                "jupyter", "nbconvert",
                "--clear-output",
                "--inplace",
                notebook
            ], check=True, capture_output=True)
            print(f"✓ Cleared {os.path.basename(notebook)}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to clear {os.path.basename(notebook)}: {e}")
    
    print("✓ All outputs cleared")
    print()


def create_sample_dataset():
    """DEPRECATED - Skipped to use full dataset with system memory."""
    # This function is no longer called but kept for compatibility
    pass


def run_analysis():
    """Run all analysis notebooks."""
    print()
    if not os.path.exists(MASTER_PARQUET):
        print("⚠ master.parquet not found. Run 'python run_makefile.py process' first.")
        return 1
    
    # Clear outputs first
    clear_notebook_outputs()
    
    # Show system memory info
    total_mem, available_mem = get_system_memory()
    if total_mem:
        print(f"ℹ️  System RAM: {total_mem:.1f} GB total, {available_mem:.1f} GB available")
        if available_mem < 4:
            print(f"   ⚠️  WARNING: Low memory! Consider closing other applications")
    
    # Skip sample creation - use full dataset
    print("ℹ️  Using full master.parquet dataset")
    print("   Notebooks will access system memory directly")
    print()
    
    print("=== Running Analysis Notebooks ===")
    print("ℹ️  Notebooks will run sequentially (one at a time)")
    print()
    os.makedirs(COMPLETION_DIR, exist_ok=True)
    
    if not ANALYSIS_NOTEBOOKS:
        print("No analysis notebooks configured.")
        return 0
    
    total_notebooks = len(ANALYSIS_NOTEBOOKS)
    completed_count = 0
    failed_notebooks = []
    
    for idx, notebook in enumerate(ANALYSIS_NOTEBOOKS, 1):
        nb_name = os.path.splitext(os.path.basename(notebook))[0]
        done_marker = os.path.join(COMPLETION_DIR, f"{nb_name}.done")
        
        print(f"\n[{idx}/{total_notebooks}] Processing: {os.path.basename(notebook)}")
        print("-" * 70)
        
        if os.path.exists(done_marker):
            print(f"✓ Already complete - skipping")
            completed_count += 1
            continue
        
        if not os.path.exists(notebook):
            print(f"✗ WARNING: File not found - skipping")
            failed_notebooks.append((notebook, "File not found"))
            continue
        
        print(f"→ Executing notebook (this may take several minutes)...")
        print(f"  Timeout: {TIMEOUT_SECONDS}s ({TIMEOUT_SECONDS//60} minutes)")
        print(f"  Progress updates will appear as cells execute...")
        print()
        
        cmd = [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            notebook,
            "--ExecutePreprocessor.kernel_name=python3",
            "--log-level=INFO"  # Show execution progress
        ]
        
        if ENABLE_TIMEOUT:
            cmd.extend(["--ExecutePreprocessor.timeout", str(TIMEOUT_SECONDS)])
        
        # Add memory management options
        env = os.environ.copy()
        
        # System memory allocation - use local machine RAM directly
        env['PYTHONHASHSEED'] = '0'  # Reproducible results
        env['MALLOC_TRIM_THRESHOLD_'] = '65536'  # Aggressive memory cleanup
        
        # Allow Jupyter to use system memory directly (no limits)
        env['PYTHONMALLOC'] = 'malloc'  # Use system allocator directly
        env['PYTHONUNBUFFERED'] = '1'  # Unbuffered output for live updates
        
        # Fix Windows asyncio issues that cause kernel crashes
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        
        # Remove any memory limits to allow full system RAM access
        for key in list(env.keys()):
            if 'MEMORY' in key.upper() or 'LIMIT' in key.upper():
                if key not in ['MALLOC_TRIM_THRESHOLD_']:
                    del env[key]
        
        # Platform-specific optimizations
        if sys.platform == 'win32':
            env['_PYTHON_HOST_PLATFORM'] = 'win-amd64'
            # Use Windows Selector Event Loop Policy to avoid asyncio warnings/crashes
            env['JUPYTER_PLATFORM_DIRS'] = '1'
        
        print(f"  Memory: Using system RAM directly (unrestricted)")
        
        # Show live output from notebook execution
        sys.stdout.flush()
        
        try:
            # Run without capture_output so we see progress
            result = subprocess.run(cmd, check=True, env=env, 
                                   stdout=sys.stdout, stderr=sys.stderr)
            print()
            print(f"✓ Completed successfully")
            completed_count += 1
            
            # Create completion marker
            with open(done_marker, 'w') as f:
                f.write("")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Execution failed (exit code {e.returncode})"
            if ENABLE_TIMEOUT:
                error_msg += f" - may have timed out after {TIMEOUT_SECONDS}s"
            print(f"✗ {error_msg}")
            failed_notebooks.append((notebook, error_msg))
        except KeyboardInterrupt:
            print(f"\n✗ Interrupted by user")
            print(f"\nProgress: {completed_count}/{total_notebooks} completed")
            return 1
    
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total notebooks: {total_notebooks}")
    print(f"Completed: {completed_count}")
    print(f"Failed: {len(failed_notebooks)}")
    
    if failed_notebooks:
        print(f"\nFailed notebooks:")
        for notebook, reason in failed_notebooks:
            print(f"  ✗ {os.path.basename(notebook)}: {reason}")
    
    print("=" * 70)
    
    print("✓ All analysis notebooks processed")
    print()
    print_output_summary()
    return 0


def mark_complete(notebook_path):
    """Mark a notebook as complete without running it."""
    if not notebook_path:
        print("Usage: python run_makefile.py mark-complete <path/to/notebook.ipynb>")
        return 1
    
    os.makedirs(COMPLETION_DIR, exist_ok=True)
    nb_name = os.path.splitext(os.path.basename(notebook_path))[0]
    done_marker = os.path.join(COMPLETION_DIR, f"{nb_name}.done")
    
    with open(done_marker, 'w') as f:
        f.write("")
    
    print(f"✓ Marked {os.path.basename(notebook_path)} as complete")
    return 0


def list_status():
    """List completion status of all notebooks."""
    print("=== Completion Status ===")
    print()
    print("Data Wrangling:")
    
    first_pass_done = os.path.join(COMPLETION_DIR, "1st_pass.done")
    if os.path.exists(first_pass_done):
        print("  ✓ 1st_pass.ipynb - COMPLETE")
    elif os.path.exists(MASTER_PARQUET):
        print("  ✓ 1st_pass.ipynb - COMPLETE (master.parquet exists)")
    else:
        print("  ✗ 1st_pass.ipynb - NOT RUN")
    
    print()
    print("Analysis Notebooks:")
    
    if not ANALYSIS_NOTEBOOKS:
        print("  (No analysis notebooks configured)")
    else:
        for notebook in ANALYSIS_NOTEBOOKS:
            nb_name = os.path.splitext(os.path.basename(notebook))[0]
            done_marker = os.path.join(COMPLETION_DIR, f"{nb_name}.done")
            
            if os.path.exists(done_marker):
                print(f"  ✓ {os.path.basename(notebook)} - COMPLETE")
            else:
                print(f"  ✗ {os.path.basename(notebook)} - NOT RUN")


def status():
    """Show project status."""
    print("=== CS506 Project Status ===")
    print()
    print("Master Data:")
    if os.path.exists(MASTER_PARQUET):
        print(f"  ✓ master.parquet exists")
    else:
        print(f"  ✗ master.parquet missing")
    
    print()
    print("Notebooks:")
    if os.path.exists(FIRST_PASS_NOTEBOOK):
        print(f"  ✓ 1st_pass.ipynb found")
    else:
        print(f"  ✗ 1st_pass.ipynb missing")


def clean_master():
    """Remove master.parquet to force reprocessing."""
    if os.path.exists(MASTER_PARQUET):
        print(f"Removing {MASTER_PARQUET}...")
        os.remove(MASTER_PARQUET)
        print("✓ Removed. Run 'python run_makefile.py process' to regenerate")
    else:
        print("master.parquet does not exist")


def clean_all():
    """Remove all completion markers."""
    if os.path.exists(COMPLETION_DIR):
        print("Removing all completion markers...")
        import shutil
        shutil.rmtree(COMPLETION_DIR)
        print("✓ All completion markers removed")
    else:
        print("No completion markers to remove")


def print_output_summary():
    """Print summary of all outputs and results."""
    print()
    print("=" * 70)
    print("OUTPUT SUMMARY")
    print("=" * 70)
    print()
    
    # Check for master dataset
    print("Dataset:")
    print(f"  Master: {MASTER_PARQUET}")
    if os.path.exists(MASTER_PARQUET):
        size_mb = os.path.getsize(MASTER_PARQUET) / 1024 / 1024
        print(f"    Size: {size_mb:.2f} MB")
    
    master_sample = r"1_LIB\master\master_sample.parquet"
    if os.path.exists(master_sample):
        size_mb = os.path.getsize(master_sample) / 1024 / 1024
        print(f"  Sample: {master_sample} ({size_mb:.2f} MB) - not used")
    print()
    
    # Check completion markers
    print("Completion Status:")
    if os.path.exists(COMPLETION_DIR):
        done_files = [f for f in os.listdir(COMPLETION_DIR) if f.endswith('.done')]
        if done_files:
            for done_file in sorted(done_files):
                nb_name = done_file.replace('.done', '')
                print(f"  DONE: {nb_name}")
        else:
            print("  (No notebooks completed yet)")
    else:
        print("  (No completion markers)")
    print()
    
    # List output notebooks
    print("Analysis Notebooks (contain embedded outputs):")
    for notebook in ANALYSIS_NOTEBOOKS:
        if os.path.exists(notebook):
            size_mb = os.path.getsize(notebook) / 1024 / 1024
            status = "DONE" if os.path.exists(os.path.join(COMPLETION_DIR, f"{os.path.splitext(os.path.basename(notebook))[0]}.done")) else "PEND"
            print(f"  [{status}] {notebook}")
            print(f"         Size: {size_mb:.2f} MB")
        else:
            print(f"  [MISS] {notebook} (not found)")
    print()
    
    print("=" * 70)
    print("To view results:")
    print("  jupyter notebook <notebook_path>")
    print()
    print("To check individual notebook:")
    print("  jupyter notebook 3_OUTPUT/3_linear_regression/linear_regression.ipynb")
    print("=" * 70)
    print()

def clear_outputs_only():
    """Clear outputs from all analysis notebooks."""
    clear_notebook_outputs()
    return 0


def show_help():
    """Show help information."""
    print("CS506 Project Makefile Runner")
    print()
    print("Usage: python run_makefile.py [target]")
    print()
    print("Main Targets:")
    print("  all               - Check master.parquet and run analysis notebooks")
    print("  process           - Run data wrangling (only if master.parquet missing)")
    print("  run-analysis      - Run all configured analysis notebooks")
    print()
    print("Status and Information:")
    print("  status            - Show project status")
    print("  list-status       - Show detailed completion status")
    print("  check-master      - Check if master.parquet exists")
    print()
    print("Utilities:")
    print("  clear-outputs     - Clear all notebook outputs (reduces file size)")
    print("  mark-complete <notebook.ipynb>")
    print("                    - Mark a notebook as complete without running it")
    print("  clean-master      - Remove master.parquet to force reprocessing")
    print("  clean-all         - Remove all completion markers")
    print()
    print("Examples:")
    print("  python run_makefile.py status")
    print("  python run_makefile.py process")
    print("  python run_makefile.py clear-outputs")
    print("  python run_makefile.py run-analysis")
    print("  python run_makefile.py mark-complete 2_FIGURES\\heavy_computation.ipynb")


def main():
    if len(sys.argv) < 2:
        target = "all"
    else:
        target = sys.argv[1]
    
    targets = {
        "all": lambda: check_master() or run_analysis(),
        "check-master": lambda: check_master() or 0,
        "process": process,
        "run-analysis": run_analysis,
        "clear-outputs": clear_outputs_only,
        "mark-complete": lambda: mark_complete(sys.argv[2] if len(sys.argv) > 2 else None),
        "list-status": lambda: list_status() or 0,
        "status": lambda: status() or 0,
        "clean-master": lambda: clean_master() or 0,
        "clean-all": lambda: clean_all() or 0,
        "help": lambda: show_help() or 0,
    }
    
    if target not in targets:
        print(f"Unknown target: {target}")
        print("Run 'python run_makefile.py help' for available targets")
        return 1
    
    return targets[target]()


if __name__ == "__main__":
    sys.exit(main())
