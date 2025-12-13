#!/bin/bash
# Quick test of organized folder structure

echo "================================"
echo "Testing New Folder Structure"
echo "================================"
echo ""

# Test 1: Check if files exist in new locations
echo "1. Checking file locations..."
if [ -f "Dependencies/requirements.txt" ]; then
    echo "✓ Dependencies/requirements.txt exists"
else
    echo "✗ Dependencies/requirements.txt missing"
fi

if [ -f "Build/Makefile.wsl" ]; then
    echo "✓ Build/Makefile.wsl exists"
else
    echo "✗ Build/Makefile.wsl missing"
fi

if [ -f "Build/path_utils.py" ]; then
    echo "✓ Build/path_utils.py exists"
else
    echo "✗ Build/path_utils.py missing"
fi

echo ""
echo "2. Testing path_utils import..."
cd Build
python3 -c "from path_utils import get_project_root, MASTER_PARQUET; print(f'Project root: {get_project_root()}'); print(f'Master parquet: {MASTER_PARQUET}')"
cd ..

echo ""
echo "3. Testing makefile syntax..."
make -f Build/Makefile.wsl help 2>&1 | head -10

echo ""
echo "================================"
echo "Structure test complete!"
echo "================================"
