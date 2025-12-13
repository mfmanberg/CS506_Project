@echo off
REM Test Linear Regression workflow locally
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

echo Testing Linear Regression workflow...
echo ========================================
echo.

wsl bash -c "cd /mnt/c/Users/Matt/Desktop/CS506/CS506_Project && source .venv_wsl/bin/activate && papermill 3_OUTPUT/3_linear_regression/linear_regression.ipynb 3_OUTPUT/3_linear_regression/linear_regression_test.ipynb --kernel python3 --execution-timeout 1800"

echo.
echo Linear Regression test complete!
echo Output saved to: 3_OUTPUT/3_linear_regression/linear_regression_test.ipynb
echo.
echo Extracting results...
wsl bash -c "cd /mnt/c/Users/Matt/Desktop/CS506/CS506_Project && source .venv_wsl/bin/activate && python3 extract_results.py"

echo.
pause
