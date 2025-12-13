@echo off
REM Test SVMDaily workflow locally
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

echo Testing SVMDaily workflow...
echo ========================================
echo.

wsl bash -c "cd /mnt/c/Users/Matt/Desktop/CS506/CS506_Project && source .venv_wsl/bin/activate && papermill 3_OUTPUT/3_svr/SVMDaily.ipynb 3_OUTPUT/3_svr/SVMDaily_test.ipynb --kernel python3 --execution-timeout 1800"

echo.
echo SVMDaily test complete!
echo Output saved to: 3_OUTPUT/3_svr/SVMDaily_test.ipynb
pause
