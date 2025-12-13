@echo off
REM ============================================================================
REM Complete Dependency Installer for CS506 Project
REM Installs: Chocolatey, Make, Python (if needed), and Python packages
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo CS506 PROJECT - DEPENDENCY INSTALLER
echo ============================================================================
echo.
echo This script will install all required dependencies:
echo   - Chocolatey (package manager for Windows)
echo   - Make (build automation tool)
echo   - Python packages (via pip from requirements.txt)
echo.
echo Press Ctrl+C to cancel, or
pause

REM ============================================================================
REM STEP 1: Check for Administrator Privileges
REM ============================================================================
echo.
echo [STEP 1/6] Checking administrator privileges...

net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Not running as administrator!
    echo.
    echo Some installations may fail without admin rights.
    echo Right-click this script and select "Run as administrator"
    echo.
    echo Press any key to continue anyway, or Ctrl+C to cancel...
    pause >nul
) else (
    echo [OK] Running as administrator
)

REM ============================================================================
REM STEP 2: Check/Install Chocolatey
REM ============================================================================
echo.
echo [STEP 2/6] Checking Chocolatey...

where choco >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Chocolatey already installed
    goto :choco_ready
)

REM Check if Chocolatey is installed but not in PATH
if exist "C:\ProgramData\chocolatey\bin\choco.exe" (
    echo [INFO] Chocolatey found but not in PATH - adding to current session...
    set "PATH=%PATH%;C:\ProgramData\chocolatey\bin"
    echo [OK] Chocolatey is now available
    goto :choco_ready
)

echo [INFO] Chocolatey not found - installing...
echo.

REM Install Chocolatey using PowerShell
powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"

REM Add to PATH for current session
if exist "C:\ProgramData\chocolatey\bin\choco.exe" (
    echo [OK] Chocolatey installed successfully
    echo [INFO] Adding to PATH for current session...
    set "PATH=%PATH%;C:\ProgramData\chocolatey\bin"
) else (
    echo [ERROR] Chocolatey installation failed
    echo Please close this window, open a NEW admin PowerShell, and run this script again
    pause
    exit /b 1
)

:choco_ready

REM ============================================================================
REM STEP 3: Install Make
REM ============================================================================
echo.
echo [STEP 3/6] Checking Make...

where make >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%i in ('make --version 2^>^&1 ^| findstr /C:"GNU Make"') do echo [OK] %%i already installed
    goto :skip_make
)

echo [INFO] Make not found - installing via Chocolatey...

REM Verify choco is available
where choco >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [WARNING] Chocolatey not available in PATH
    echo [INFO] Skipping Make installation
    echo [INFO] You can still use: setup.bat make [command]
    goto :skip_make
)

REM Try installing with refreshenv
echo [INFO] Installing Make (this may take a minute)...
choco install make -y --force

REM Wait for installation
timeout /t 3 /nobreak >nul 2>&1

REM Try to refresh environment using PowerShell
echo [INFO] Refreshing environment variables...
powershell -Command "& {$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User'); Write-Output $env:Path}" > temp_path.txt
for /f "usebackq delims=" %%i in ("temp_path.txt") do set "PATH=%%i"
del temp_path.txt >nul 2>&1

REM Add all possible Make paths
set "PATH=%PATH%;C:\Program Files\GnuWin32\bin"
set "PATH=%PATH%;C:\ProgramData\chocolatey\bin"
set "PATH=%PATH%;C:\ProgramData\chocolatey\lib\make\tools\install\bin"

REM Check if Make is now available
where make >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    for /f "tokens=*" %%i in ('make --version 2^>^&1 ^| findstr /C:"GNU Make"') do echo [OK] %%i installed successfully
    goto :skip_make
)

REM Search for make.exe in all possible locations
echo [INFO] Searching for Make installation...

if exist "C:\Program Files\GnuWin32\bin\make.exe" (
    echo [OK] Make found at C:\Program Files\GnuWin32\bin
    set "PATH=%PATH%;C:\Program Files\GnuWin32\bin"
    goto :skip_make
)

if exist "C:\ProgramData\chocolatey\bin\make.exe" (
    echo [OK] Make found at C:\ProgramData\chocolatey\bin
    set "PATH=%PATH%;C:\ProgramData\chocolatey\bin"
    goto :skip_make
)

if exist "C:\ProgramData\chocolatey\lib\make\tools\install\bin\make.exe" (
    echo [OK] Make found at C:\ProgramData\chocolatey\lib\make\tools\install\bin
    set "PATH=%PATH%;C:\ProgramData\chocolatey\lib\make\tools\install\bin"
    goto :skip_make
)

REM Last resort - search chocolatey directory
for /r "C:\ProgramData\chocolatey" %%f in (make.exe) do (
    if exist "%%f" (
        set "MAKE_DIR=%%~dpf"
        echo [OK] Make found at !MAKE_DIR!
        set "PATH=%PATH%;!MAKE_DIR!"
        goto :skip_make
    )
)

echo [WARNING] Make was installed but cannot be found in PATH
echo [INFO] Make will be available after opening a new terminal
echo [INFO] For now, you can use: setup.bat make [command]

:skip_make

REM ============================================================================
REM STEP 4: Check Python
REM ============================================================================
echo.
echo [STEP 4/6] Checking Python...

set PYTHON_CMD=
set PYTHON_FOUND=0

REM Try py launcher
py --version >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    set PYTHON_CMD=py
    set PYTHON_FOUND=1
    for /f "tokens=*" %%i in ('py --version 2^>^&1') do echo [OK] %%i found via py launcher
    goto :python_ready
)

REM Try python command
if !PYTHON_FOUND! EQU 0 (
    python --version >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set PYTHON_CMD=python
        set PYTHON_FOUND=1
        for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo [OK] %%i found via python command
        goto :python_ready
    )
)

REM Try python3
if !PYTHON_FOUND! EQU 0 (
    python3 --version >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set PYTHON_CMD=python3
        set PYTHON_FOUND=1
        for /f "tokens=*" %%i in ('python3 --version 2^>^&1') do echo [OK] %%i found via python3 command
        goto :python_ready
    )
)

REM Python not found - offer to install
if !PYTHON_FOUND! EQU 0 (
    echo [WARNING] Python not found!
    echo.
    
    REM Check if choco is available
    where choco >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo Would you like to install Python via Chocolatey? (Y/N)
        set /p INSTALL_PYTHON=
        
        if /i "!INSTALL_PYTHON!"=="Y" (
            echo [INFO] Installing Python via Chocolatey...
            choco install python -y
            
            REM Add Python to PATH for current session
            set "PATH=%PATH%;C:\Python312;C:\Python312\Scripts"
            
            REM Check again
            py --version >nul 2>&1
            if !ERRORLEVEL! EQU 0 (
                set PYTHON_CMD=py
                set PYTHON_FOUND=1
                echo [OK] Python installed successfully
            ) else (
                python --version >nul 2>&1
                if !ERRORLEVEL! EQU 0 (
                    set PYTHON_CMD=python
                    set PYTHON_FOUND=1
                    echo [OK] Python installed successfully
                ) else (
                    echo [ERROR] Python installation failed
                    echo Please close this window, open a NEW terminal, and run this script again
                    echo Or install manually from: https://www.python.org/downloads/
                    pause
                    exit /b 1
                )
            )
        ) else (
            echo [ERROR] Python is required to continue
            echo Please install from: https://www.python.org/downloads/
            echo Make sure to check "Add Python to PATH" during installation!
            echo Then run this script again
            pause
            exit /b 1
        )
    ) else (
        echo [ERROR] Python is required but Chocolatey is not available
        echo.
        echo Please install Python manually:
        echo   1. Download from: https://www.python.org/downloads/
        echo   2. Run the installer
        echo   3. CHECK "Add Python to PATH" during installation
        echo   4. After installation, close this window
        echo   5. Open a NEW terminal and run this script again
        echo.
        pause
        exit /b 1
    )
)

:python_ready

REM ============================================================================
REM STEP 5: Create Virtual Environment
REM ============================================================================
echo.
echo [STEP 5/6] Setting up virtual environment...

cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    echo [OK] Virtual environment already exists
) else (
    echo [INFO] Creating virtual environment...
    !PYTHON_CMD! -m venv venv
    
    if !ERRORLEVEL! EQU 0 (
        echo [OK] Virtual environment created
    ) else (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM ============================================================================
REM STEP 6: Install Python Packages
REM ============================================================================
echo.
echo [STEP 6/6] Installing Python packages from requirements.txt...

call venv\Scripts\activate.bat

echo [INFO] Upgrading pip...
python -m pip install --quiet --upgrade pip

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)

echo [INFO] Installing packages (this may take 5-10 minutes)...
python -m pip install -r requirements.txt

if !ERRORLEVEL! EQU 0 (
    echo [OK] All packages installed successfully
) else (
    echo [WARNING] Some packages may have failed to install
)

REM Verify key packages
echo.
echo [INFO] Verifying installations...

REM Check Make first
where make >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    for /f "tokens=*" %%i in ('make --version 2^>^&1 ^| findstr /C:"GNU Make"') do set MAKE_VER=%%i
    echo [OK] !MAKE_VER!
) else (
    echo [INFO] Make - Not installed ^(Python fallback available via setup.bat^)
)

REM Check Python packages
python -c "import notebook; print('[OK] Jupyter Notebook', notebook.__version__)" 2>nul || echo [FAIL] Jupyter Notebook
python -c "import jupyterlab; print('[OK] JupyterLab', jupyterlab.__version__)" 2>nul || echo [FAIL] JupyterLab
python -c "import pandas; print('[OK] Pandas', pandas.__version__)" 2>nul || echo [FAIL] Pandas
python -c "import numpy; print('[OK] NumPy', numpy.__version__)" 2>nul || echo [FAIL] NumPy
python -c "import sklearn; print('[OK] Scikit-learn', sklearn.__version__)" 2>nul || echo [FAIL] Scikit-learn
python -c "import xgboost; print('[OK] XGBoost', xgboost.__version__)" 2>nul || echo [FAIL] XGBoost

call deactivate

REM ============================================================================
REM COMPLETION
REM ============================================================================
echo.
echo ============================================================================
echo INSTALLATION COMPLETE!
echo ============================================================================
echo.
echo All dependencies have been installed.
echo.
echo NEXT STEPS:
echo   1. Close this window
echo   2. Open a NEW PowerShell or Command Prompt
echo   3. Navigate to: %CD%
echo   4. Run: setup.bat (or .\setup.bat in PowerShell)
echo.
echo NOTE: You must open a NEW terminal for PATH changes to take effect!
echo.
pause
