# Makefile for CS506 Project
# Manages data pipeline execution
# Windows-compatible version

# === CONFIGURATION ===
# Set to TRUE to enable 10-minute timeout for notebooks, FALSE to disable timeout
ENABLE_TIMEOUT = TRUE
# Timeout in seconds (600 = 10 minutes)
TIMEOUT_SECONDS = 600

# Paths (use backslashes for Windows)
MASTER_PARQUET = 1_LIB\master\master.parquet
FIRST_PASS_NOTEBOOK = 2_FIGURES\1_data_wrangling\1st_pass.ipynb

# Additional notebooks (add your analysis notebooks here)
ANALYSIS_NOTEBOOKS = linear_regression.ipynb, SVM_Trunc.ipynb, SVMDaily.ipynb, SVMDailywoutMeso.ipynb, ComparisionMetrics.ipynb, XGBoost_PostMid_ipynb, XGBoost_Testing.ipynb
# Example: 2_FIGURES\2_analysis\analysis.ipynb 2_FIGURES\3_visualization\plots.ipynb

# Completion markers directory
COMPLETION_DIR = .make_completion
FIRST_PASS_DONE = $(COMPLETION_DIR)\1st_pass.done

# Default target
.PHONY: all
all: check-master run-analysis

# Check if master.parquet exists
.PHONY: check-master
check-master:
	@if exist "$(MASTER_PARQUET)" ( \
		echo ✓ master.parquet already exists at $(MASTER_PARQUET) && \
		echo ✓ Data wrangling complete - skipping 1st_pass.ipynb && \
		echo   To reprocess, run: make clean-master \
	) else ( \
		echo ✗ master.parquet not found && \
		echo → Run 'make process' to execute data wrangling \
	)

# Process data (run notebook) - manual trigger only
.PHONY: process
process:
	@if exist "$(MASTER_PARQUET)" ( \
		echo ⚠ master.parquet already exists && \
		echo   Run 'make clean-master' first to reprocess && \
		exit /b 1 \
	) else ( \
		echo → Running data wrangling notebook... && \
		echo ⚠ This will execute heavy computations && \
		if not exist "$(COMPLETION_DIR)" mkdir "$(COMPLETION_DIR)" && \
		jupyter nbconvert --to notebook --execute --inplace "$(FIRST_PASS_NOTEBOOK)" && \
		echo ✓ Data wrangling complete && \
		echo. > "$(FIRST_PASS_DONE)" \
	)

# Run analysis notebooks (only if master.parquet exists)
.PHONY: run-analysis
run-analysis:
	@if not exist "$(MASTER_PARQUET)" ( \
		echo ⚠ master.parquet not found. Run 'make process' first. && \
		exit /b 1 \
	)
	@echo === Running Analysis Notebooks ===
	@if not exist "$(COMPLETION_DIR)" mkdir "$(COMPLETION_DIR)"
	@for %%f in ($(ANALYSIS_NOTEBOOKS)) do ( \
		set "nb=%%f" && \
		set "done_marker=$(COMPLETION_DIR)\%%~nf.done" && \
		if exist "!done_marker!" ( \
			echo ✓ %%~nxf already complete - skipping \
		) else ( \
			echo → Running %%~nxf... && \
			if "$(ENABLE_TIMEOUT)"=="TRUE" ( \
				jupyter nbconvert --to notebook --execute --inplace "%%f" --ExecutePreprocessor.timeout=$(TIMEOUT_SECONDS) && \
				echo ✓ %%~nxf complete && \
				echo. > "!done_marker!" \
			) else ( \
				jupyter nbconvert --to notebook --execute --inplace "%%f" && \
				echo ✓ %%~nxf complete && \
				echo. > "!done_marker!" \
			) \
		) \
	)
	@if "$(ANALYSIS_NOTEBOOKS)"=="" ( \
		echo No analysis notebooks configured. Add them to ANALYSIS_NOTEBOOKS variable. \
	) else ( \
		echo ✓ All analysis notebooks complete \
	)

# Mark a notebook as complete without running (for heavy computations >10 min)
.PHONY: mark-complete
mark-complete:
	@if "$(NB)"=="" ( \
		echo Usage: make mark-complete NB=path\to\notebook.ipynb && \
		exit /b 1 \
	)
	@if not exist "$(COMPLETION_DIR)" mkdir "$(COMPLETION_DIR)"
	@for %%f in ("$(NB)") do set "nb_name=%%~nf"
	@echo. > "$(COMPLETION_DIR)\%nb_name%.done"
	@echo ✓ Marked $(NB) as complete

# List completion status
.PHONY: list-status
list-status:
	@echo === Completion Status ===
	@echo.
	@echo Data Wrangling:
	@if exist "$(FIRST_PASS_DONE)" ( \
		echo   ✓ 1st_pass.ipynb - COMPLETE \
	) else if exist "$(MASTER_PARQUET)" ( \
		echo   ✓ 1st_pass.ipynb - COMPLETE (master.parquet exists) \
	) else ( \
		echo   ✗ 1st_pass.ipynb - NOT RUN \
	)
	@echo.
	@echo Analysis Notebooks:
	@for %%f in ($(ANALYSIS_NOTEBOOKS)) do ( \
		set "nb=%%f" && \
		set "done_marker=$(COMPLETION_DIR)\%%~nf.done" && \
		if exist "!done_marker!" ( \
			echo   ✓ %%~nxf - COMPLETE \
		) else ( \
			echo   ✗ %%~nxf - NOT RUN \
		) \
	)
	@if "$(ANALYSIS_NOTEBOOKS)"=="" ( \
		echo   (No analysis notebooks configured) \
	)

# Clean master parquet to force reprocessing
.PHONY: clean-master
clean-master:
	@if exist "$(MASTER_PARQUET)" ( \
		echo Removing $(MASTER_PARQUET)... && \
		del /f "$(MASTER_PARQUET)" && \
		echo ✓ Removed. Run 'make process' to regenerate \
	) else ( \
		echo master.parquet does not exist \
	)

# Clean all completion markers
.PHONY: clean-all
clean-all:
	@if exist "$(COMPLETION_DIR)" ( \
		echo Removing all completion markers... && \
		rmdir /s /q "$(COMPLETION_DIR)" && \
		echo ✓ All completion markers removed \
	) else ( \
		echo No completion markers to remove \
	)

# Status check
.PHONY: status
status:
	@echo === CS506 Project Status ===
	@echo.
	@echo Master Data:
	@if exist "$(MASTER_PARQUET)" ( \
		echo   ✓ master.parquet exists \
	) else ( \
		echo   ✗ master.parquet missing \
	)
	@echo.
	@echo Notebooks:
	@if exist "$(FIRST_PASS_NOTEBOOK)" ( \
		echo   ✓ 1st_pass.ipynb found \
	) else ( \
		echo   ✗ 1st_pass.ipynb missing \
	)

# Help
.PHONY: help
help:
	@echo CS506 Project Makefile
	@echo.
	@echo Main Targets:
	@echo   make              - Check master.parquet and run analysis notebooks
	@echo   make process      - Run data wrangling (only if master.parquet missing)
	@echo   make run-analysis - Run all configured analysis notebooks
	@echo.
	@echo Status and Information:
	@echo   make status       - Show project status
	@echo   make list-status  - Show detailed completion status
	@echo   make check-master - Check if master.parquet exists
	@echo.
	@echo Utilities:
	@echo   make mark-complete NB=path\to\notebook.ipynb
	@echo                     - Mark a notebook as complete without running it
	@echo                       (use for notebooks with computations ^>10 minutes)
	@echo   make clean-master - Remove master.parquet to force reprocessing
	@echo   make clean-all    - Remove all completion markers
	@echo.
	@echo Configuration:
	@echo   Edit ANALYSIS_NOTEBOOKS variable to add notebooks to run after
	@echo   master.parquet exists. These will run automatically with 'make all'.
	@echo.
	@echo Examples:
	@echo   make mark-complete NB=2_FIGURES\heavy_computation.ipynb
	@echo   make run-analysis
	@echo.
	@echo Note: Heavy computations are NOT run by default.
	@echo       Notebooks taking ^>10 minutes should be marked complete manually.
