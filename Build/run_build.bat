@echo off
setlocal

REM Get the directory where this batch file is located (Build folder)
set "BUILD_DIR=%~dp0"
REM Remove trailing backslash
set "BUILD_DIR=%BUILD_DIR:~0,-1%"

REM Get the project root (parent of Build folder)
for %%I in ("%BUILD_DIR%\..") do set "PROJECT_ROOT=%%~fI"

REM Convert Windows path to WSL path
set "WSL_PATH=%PROJECT_ROOT:\=/%"
set "WSL_PATH=%WSL_PATH:C:=/mnt/c%"
set "WSL_PATH=%WSL_PATH:D:=/mnt/d%"
set "WSL_PATH=%WSL_PATH:E:=/mnt/e%"

echo.
echo ===================================================================
echo Starting Notebook Build - DO NOT INTERRUPT
echo ===================================================================
echo Project Root: %PROJECT_ROOT%
echo WSL Path: %WSL_PATH%
echo ===================================================================
echo.

wsl bash -c "cd '%WSL_PATH%' && make -f Build/Makefile.simple run"

echo.
echo ===================================================================
echo Build Complete
echo ===================================================================
echo.
pause
endlocal
