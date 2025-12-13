@echo off
REM ============================================================================
REM Clear All Notebook Outputs
REM Removes embedded outputs to reduce file size and prevent kernel crashes
REM ============================================================================

echo.
echo ============================================================================
echo CLEARING NOTEBOOK OUTPUTS
echo ============================================================================
echo.
echo This will remove all outputs from notebooks to reduce file size.
echo Embedded outputs can cause kernel crashes and make files huge.
echo.
echo Target notebooks:
echo   - linear_regression.ipynb (1.30 MB)
echo   - SVM_Trunc.ipynb (61.74 MB!) 
echo   - SVMDaily.ipynb (9.46 MB)
echo   - SVMDailywoutMeso.ipynb (9.61 MB)
echo.
pause

echo.
echo [1/4] Clearing linear_regression.ipynb...
jupyter nbconvert --clear-output --inplace 3_OUTPUT\3_linear_regression\linear_regression.ipynb
if %ERRORLEVEL% EQU 0 (
    echo ✓ Cleared
) else (
    echo ✗ Failed
)

echo.
echo [2/4] Clearing SVM_Trunc.ipynb (this may take a moment)...
jupyter nbconvert --clear-output --inplace 3_OUTPUT\3_svr\SVM_Trunc.ipynb
if %ERRORLEVEL% EQU 0 (
    echo ✓ Cleared
) else (
    echo ✗ Failed
)

echo.
echo [3/4] Clearing SVMDaily.ipynb...
jupyter nbconvert --clear-output --inplace 3_OUTPUT\3_svr\SVMDaily.ipynb
if %ERRORLEVEL% EQU 0 (
    echo ✓ Cleared
) else (
    echo ✗ Failed
)

echo.
echo [4/4] Clearing SVMDailywoutMeso.ipynb...
jupyter nbconvert --clear-output --inplace 3_OUTPUT\3_svr\SVMDailywoutMeso.ipynb
if %ERRORLEVEL% EQU 0 (
    echo ✓ Cleared
) else (
    echo ✗ Failed
)

echo.
echo ============================================================================
echo DONE!
echo ============================================================================
echo.
echo Notebook outputs have been cleared.
echo File sizes should be much smaller now.
echo.
echo Next steps:
echo   1. Run: python fix_notebook_paths.py
echo   2. Try running again: python run_makefile.py run-analysis
echo.
pause
