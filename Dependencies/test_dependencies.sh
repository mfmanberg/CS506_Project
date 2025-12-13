#!/bin/bash
# Test dependencies installation and makefile execution

echo "================================"
echo "Testing CS506 Project Dependencies"
echo "================================"
echo ""

# Activate virtual environment
source .venv_wsl/bin/activate

echo "1. Testing Python version..."
python --version

echo ""
echo "2. Testing critical package imports..."
python -c "
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import matplotlib
import seaborn as sns
import plotly
import statsmodels
import papermill
print('✓ All critical packages imported successfully')
print(f'  - pandas: {pd.__version__}')
print(f'  - numpy: {np.__version__}')
print(f'  - scikit-learn: {sklearn.__version__}')
print(f'  - xgboost: {xgb.__version__}')
print(f'  - statsmodels: {statsmodels.__version__}')
"

echo ""
echo "3. Testing papermill execution..."
papermill --version

echo ""
echo "4. Running XGBoost_Testing notebook (fast test)..."
papermill 3_OUTPUT/3_xg_boost/XGBoost_Testing.ipynb \
          3_OUTPUT/3_xg_boost/XGBoost_Testing_output.ipynb \
          --kernel python3 \
          --execution-timeout 600 \
          --log-output

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Notebook executed successfully!"
    echo ""
    echo "5. Extracting results..."
    python extract_results.py
    
    echo ""
    echo "================================"
    echo "✓ ALL TESTS PASSED"
    echo "================================"
else
    echo ""
    echo "✗ Notebook execution failed"
    exit 1
fi
