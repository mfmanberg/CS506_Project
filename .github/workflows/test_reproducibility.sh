#!/bin/bash
# Test GitHub Workflows Reproducibility
# This script verifies that workflows are configured correctly for any machine

echo "=========================================="
echo "GitHub Workflows Reproducibility Test"
echo "=========================================="
echo ""

ERRORS=0

# Pre-check: Verify virtual environment and dependencies
echo "0. Checking Python environment..."

# Check if virtual environment exists
if [ ! -d ".venv_wsl" ]; then
    echo "   ⚠ Virtual environment not found"
    echo "   Creating .venv_wsl..."
    python3 -m venv .venv_wsl
    if [ $? -ne 0 ]; then
        echo "   ✗ Failed to create virtual environment"
        ERRORS=$((ERRORS + 1))
    else
        echo "   ✓ Virtual environment created"
    fi
fi

# Activate virtual environment
if [ -f ".venv_wsl/bin/activate" ]; then
    source .venv_wsl/bin/activate
    echo "   ✓ Virtual environment activated"
else
    echo "   ✗ Cannot activate virtual environment"
    ERRORS=$((ERRORS + 1))
fi

# Check if dependencies are installed
echo ""
echo "   Checking dependencies..."
if [ -f "Dependencies/requirements.txt" ]; then
    # Check if .deps_installed marker exists and is newer than requirements.txt
    if [ ! -f ".venv_wsl/.deps_installed" ] || [ "Dependencies/requirements.txt" -nt ".venv_wsl/.deps_installed" ]; then
        echo "   Installing/updating dependencies from requirements.txt..."
        pip install --upgrade pip
        pip install -r Dependencies/requirements.txt
        if [ $? -eq 0 ]; then
            touch .venv_wsl/.deps_installed
            echo "   ✓ Dependencies installed successfully"
        else
            echo "   ✗ Failed to install dependencies"
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo "   ✓ Dependencies up to date"
    fi
else
    echo "   ✗ Dependencies/requirements.txt not found"
    ERRORS=$((ERRORS + 1))
fi

# Verify critical packages
echo "   Verifying critical packages..."
CRITICAL_PACKAGES=("pandas" "numpy" "scikit-learn" "xgboost" "papermill" "jupyter")
MISSING_PACKAGES=()

for pkg in "${CRITICAL_PACKAGES[@]}"; do
    if ! pip show "$pkg" > /dev/null 2>&1; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo "   ✓ All critical packages installed"
else
    echo "   ✗ Missing packages: ${MISSING_PACKAGES[*]}"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Test 1: Check workflow files exist
echo "1. Checking workflow files..."
WORKFLOWS=(
    ".github/workflows/workflows_linear_regression.yml"
    ".github/workflows/workflows_svmdaily.yml"
    ".github/workflows/workflows_xgboost_testing.yml"
)

for workflow in "${WORKFLOWS[@]}"; do
    if [ -f "$workflow" ]; then
        echo "   ✓ Found: $workflow"
    else
        echo "   ✗ Missing: $workflow"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 2: Check PYTHONPATH configuration
echo ""
echo "2. Checking PYTHONPATH in workflows..."
for workflow in "${WORKFLOWS[@]}"; do
    if grep -q 'export PYTHONPATH="\$PWD:\$PWD/Build:\$PYTHONPATH"' "$workflow"; then
        echo "   ✓ PYTHONPATH configured in: $(basename $workflow)"
    else
        echo "   ✗ PYTHONPATH missing in: $(basename $workflow)"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 3: Check --cwd flag in papermill commands
echo ""
echo "3. Checking papermill --cwd flag..."
for workflow in "${WORKFLOWS[@]}"; do
    if grep -q '\-\-cwd "\$PWD"' "$workflow"; then
        echo "   ✓ --cwd flag found in: $(basename $workflow)"
    else
        echo "   ✗ --cwd flag missing in: $(basename $workflow)"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 4: Check Dependencies/requirements.txt path
echo ""
echo "4. Checking requirements.txt path..."
for workflow in "${WORKFLOWS[@]}"; do
    if grep -q 'Dependencies/requirements.txt' "$workflow"; then
        echo "   ✓ Correct path in: $(basename $workflow)"
    else
        echo "   ✗ Wrong path in: $(basename $workflow)"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 5: Check Git LFS is enabled
echo ""
echo "5. Checking Git LFS configuration..."
for workflow in "${WORKFLOWS[@]}"; do
    if grep -q 'lfs: true' "$workflow"; then
        echo "   ✓ LFS enabled in: $(basename $workflow)"
    else
        echo "   ✗ LFS not enabled in: $(basename $workflow)"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 6: Check for hardcoded paths
echo ""
echo "6. Checking for hardcoded paths..."
HARDCODED=0
for workflow in "${WORKFLOWS[@]}"; do
    # Look for common hardcoded path patterns
    if grep -qE '(/home/|/mnt/c/|C:\\|/Users/)' "$workflow" 2>/dev/null; then
        echo "   ⚠ Warning: Possible hardcoded path in: $(basename $workflow)"
        HARDCODED=$((HARDCODED + 1))
    fi
done

if [ $HARDCODED -eq 0 ]; then
    echo "   ✓ No hardcoded paths detected"
fi

# Test 7: Verify required files exist
echo ""
echo "7. Checking required files..."
REQUIRED_FILES=(
    "Dependencies/requirements.txt"
    "Build/path_utils.py"
    "Build/extract_results.py"
    "Build/test_notebook_execution.py"
    "1_LIB/master/master.parquet"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ Found: $file"
    else
        echo "   ✗ Missing: $file"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 8: Test path_utils import locally
echo ""
echo "8. Testing path_utils import..."
if export PYTHONPATH="$PWD:$PWD/Build:$PYTHONPATH" && python3 -c "from path_utils import get_project_root; print(f'   ✓ Project root: {get_project_root()}')"; then
    :
else
    echo "   ✗ path_utils import failed"
    ERRORS=$((ERRORS + 1))
fi

# Test 9: Verify notebooks use path_utils
echo ""
echo "9. Checking notebooks use path_utils..."
NOTEBOOKS=(
    "3_OUTPUT/3_linear_regression/linear_regression.ipynb"
    "3_OUTPUT/3_svr/SVMDaily.ipynb"
    "3_OUTPUT/3_xg_boost/XGBoost_Testing.ipynb"
)

for notebook in "${NOTEBOOKS[@]}"; do
    if [ -f "$notebook" ]; then
        if grep -q "path_utils" "$notebook"; then
            echo "   ✓ Uses path_utils: $(basename $notebook)"
        else
            echo "   ⚠ Warning: May not use path_utils: $(basename $notebook)"
        fi
    else
        echo "   ✗ Notebook not found: $notebook"
        ERRORS=$((ERRORS + 1))
    fi
done

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ All tests passed!"
    echo "Workflows are configured for reproducibility."
    exit 0
else
    echo "❌ $ERRORS error(s) found"
    echo "Please fix the issues above before pushing to GitHub."
    exit 1
fi
