@echo off
REM Test XGBoost_Testing workflow locally
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

echo Testing XGBoost_Testing workflow...
echo ========================================
echo.

wsl bash -c "cd /mnt/c/Users/Matt/Desktop/CS506/CS506_Project && source .venv_wsl/bin/activate && papermill 3_OUTPUT/3_xg_boost/XGBoost_Testing.ipynb 3_OUTPUT/3_xg_boost/XGBoost_Testing_output.ipynb --kernel python3 --execution-timeout 1800"

echo.
echo XGBoost_Testing test complete!
echo Output saved to: 3_OUTPUT/3_xg_boost/XGBoost_Testing_output.ipynb
echo.
echo Extracting results...
wsl bash -c "cd /mnt/c/Users/Matt/Desktop/CS506/CS506_Project && source .venv_wsl/bin/activate && python3 extract_results.py"

echo.
echo Verifying execution...
wsl bash -c "cd /mnt/c/Users/Matt/Desktop/CS506/CS506_Project && source .venv_wsl/bin/activate && python3 test_execution.py"

echo.
echo Test complete! Check output above for results.
pause
