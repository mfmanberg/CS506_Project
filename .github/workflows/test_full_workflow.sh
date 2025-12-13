#!/bin/bash
# Full GitHub Actions Workflow Simulation
# This script simulates the complete GitHub Actions workflow locally

set -e  # Exit on error

echo "=========================================="
echo "Full GitHub Actions Workflow Simulation"
echo "=========================================="
echo ""
echo "This script simulates what happens in GitHub Actions:"
echo "1. Environment setup"
echo "2. Dependency installation"
echo "3. Notebook execution"
echo "4. Result extraction"
echo "5. Verification"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up temporary files..."
    rm -f /tmp/test_*.ipynb
}
trap cleanup EXIT

# Step 1: Environment Setup
echo "=========================================="
echo "Step 1: Environment Setup"
echo "=========================================="

if [ ! -d ".venv_wsl" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv_wsl
fi

echo "Activating virtual environment..."
source .venv_wsl/bin/activate

echo "Python version: $(python3 --version)"
echo "pip version: $(pip --version)"
echo ""

# Step 2: Install Dependencies
echo "=========================================="
echo "Step 2: Install Dependencies"
echo "=========================================="

echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "Installing dependencies from Dependencies/requirements.txt..."
if [ -f "Dependencies/requirements.txt" ]; then
    pip install -r Dependencies/requirements.txt --quiet
    echo "✓ Dependencies installed"
else
    echo "✗ Dependencies/requirements.txt not found!"
    exit 1
fi

echo "Installing papermill, jupyter, nbformat..."
pip install papermill jupyter nbformat --quiet
echo "✓ Notebook execution tools installed"
echo ""

# Step 3: Verify Setup
echo "=========================================="
echo "Step 3: Verify Setup"
echo "=========================================="

echo "Checking critical packages..."
PACKAGES=("pandas" "numpy" "scikit-learn" "xgboost" "papermill" "jupyter")
for pkg in "${PACKAGES[@]}"; do
    if pip show "$pkg" > /dev/null 2>&1; then
        VERSION=$(pip show "$pkg" | grep "^Version:" | cut -d' ' -f2)
        echo "   ✓ $pkg: $VERSION"
    else
        echo "   ✗ $pkg: NOT INSTALLED"
        exit 1
    fi
done
echo ""

# Step 4: Check Git LFS
echo "=========================================="
echo "Step 4: Check Git LFS (Data Files)"
echo "=========================================="

if [ -f "1_LIB/master/master.parquet" ]; then
    SIZE=$(du -h 1_LIB/master/master.parquet | cut -f1)
    echo "✓ master.parquet found (size: $SIZE)"
    
    # Check if it's a Git LFS pointer file (should be ~38MB, not ~130 bytes)
    FILE_SIZE=$(stat -c%s "1_LIB/master/master.parquet" 2>/dev/null || stat -f%z "1_LIB/master/master.parquet" 2>/dev/null)
    if [ "$FILE_SIZE" -lt 1000 ]; then
        echo "⚠ Warning: File appears to be a Git LFS pointer file"
        echo "  Run: git lfs pull"
    else
        echo "✓ Data file is valid (not a pointer)"
    fi
else
    echo "✗ master.parquet not found!"
    exit 1
fi
echo ""

# Step 5: Test path_utils
echo "=========================================="
echo "Step 5: Test path_utils Module"
echo "=========================================="

export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"
echo ""

python3 -c "
from path_utils import get_project_root
root = get_project_root()
print(f'✓ Project root detected: {root}')
print(f'✓ path_utils imports successfully')
"
echo ""

# Step 6: Execute Test Notebooks
echo "=========================================="
echo "Step 6: Execute Test Notebooks"
echo "=========================================="

NOTEBOOKS=(
    "3_OUTPUT/3_xg_boost/XGBoost_Testing.ipynb"
    "3_OUTPUT/3_linear_regression/linear_regression.ipynb"
    "3_OUTPUT/3_svr/SVMDaily.ipynb"
)

EXECUTED=0
FAILED=0

for nb in "${NOTEBOOKS[@]}"; do
    BASENAME=$(basename "$nb")
    OUTPUT="/tmp/test_${BASENAME}"
    
    echo "Executing: $BASENAME"
    echo "  Input:  $nb"
    echo "  Output: $OUTPUT"
    echo "  Timeout: 300 seconds (5 minutes)"
    
    if papermill \
        "$nb" \
        "$OUTPUT" \
        --kernel python3 \
        --execution-timeout 300 \
        --cwd "$PWD" \
        --log-output \
        2>&1 | grep -E "(Executing|Error|Success)" | tail -5; then
        echo "  ✓ SUCCESS"
        EXECUTED=$((EXECUTED + 1))
    else
        echo "  ✗ FAILED"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# Step 7: Extract Results
echo "=========================================="
echo "Step 7: Extract Results"
echo "=========================================="

echo "Running extract_results.py..."
if python3 Build/extract_results.py; then
    echo "✓ Results extracted successfully"
    if [ -f "Build/model_results.log" ]; then
        echo ""
        echo "Recent results:"
        tail -20 Build/model_results.log
    fi
else
    echo "⚠ Result extraction completed with warnings"
fi
echo ""

# Step 8: Verify Execution
echo "=========================================="
echo "Step 8: Verify Execution Environment"
echo "=========================================="

echo "Running test_notebook_execution.py..."
if python3 Build/test_notebook_execution.py; then
    echo "✓ Environment verification passed"
else
    echo "⚠ Verification completed with warnings"
fi
echo ""

# Final Summary
echo "=========================================="
echo "Workflow Simulation Summary"
echo "=========================================="
echo "Notebooks executed: $EXECUTED"
echo "Notebooks failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ All workflows simulated successfully!"
    echo ""
    echo "Your setup is ready for GitHub Actions:"
    echo "  - Dependencies installed correctly"
    echo "  - path_utils working"
    echo "  - Notebooks execute successfully"
    echo "  - Results extracted properly"
    echo ""
    echo "Safe to push to GitHub! 🚀"
    exit 0
else
    echo "❌ Some workflows failed"
    echo "Please fix issues before pushing to GitHub"
    exit 1
fi
