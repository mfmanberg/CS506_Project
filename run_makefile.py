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


def run_analysis():
    """Run all analysis notebooks."""
    print()
    if not os.path.exists(MASTER_PARQUET):
        print("⚠ master.parquet not found. Run 'python run_makefile.py process' first.")
        return 1
    
    print("=== Running Analysis Notebooks ===")
    os.makedirs(COMPLETION_DIR, exist_ok=True)
    
    if not ANALYSIS_NOTEBOOKS:
        print("No analysis notebooks configured.")
        return 0
    
    for notebook in ANALYSIS_NOTEBOOKS:
        nb_name = os.path.splitext(os.path.basename(notebook))[0]
        done_marker = os.path.join(COMPLETION_DIR, f"{nb_name}.done")
        
        if os.path.exists(done_marker):
            print(f"✓ {os.path.basename(notebook)} already complete - skipping")
            continue
        
        if not os.path.exists(notebook):
            print(f"⚠ WARNING: {os.path.basename(notebook)} not found - skipping")
            continue
        
        print(f"→ Running {os.path.basename(notebook)}...")
        
        cmd = [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            notebook
        ]
        
        if ENABLE_TIMEOUT:
            cmd.extend(["--ExecutePreprocessor.timeout", str(TIMEOUT_SECONDS)])
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ {os.path.basename(notebook)} complete")
            
            # Create completion marker
            with open(done_marker, 'w') as f:
                f.write("")
        except subprocess.CalledProcessError:
            if ENABLE_TIMEOUT:
                print(f"✗ {os.path.basename(notebook)} failed or timed out (timeout={TIMEOUT_SECONDS}s)")
            else:
                print(f"✗ {os.path.basename(notebook)} failed")
    
    print("✓ All analysis notebooks processed")
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
    print("  mark-complete <notebook.ipynb>")
    print("                    - Mark a notebook as complete without running it")
    print("  clean-master      - Remove master.parquet to force reprocessing")
    print("  clean-all         - Remove all completion markers")
    print()
    print("Examples:")
    print("  python run_makefile.py status")
    print("  python run_makefile.py process")
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
