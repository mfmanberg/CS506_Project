@echo off
REM Detect project root (directory containing this script)
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

echo Running notebooks in WSL...
echo This window will close when complete.
echo Project root: %PROJECT_ROOT%
echo.

REM Convert Windows path to WSL path format
set WSL_PATH=%PROJECT_ROOT:\=/%
set WSL_PATH=%WSL_PATH:C:/=/mnt/c/%
set WSL_PATH=%WSL_PATH:D:/=/mnt/d/%
set WSL_PATH=%WSL_PATH:E:/=/mnt/e/%

wsl bash -c "cd '%WSL_PATH%' && source .venv_wsl/bin/activate && make -f Makefile.wsl run && python3 extract_results.py"
echo.
echo Execution complete! Press any key to exit.
pause
