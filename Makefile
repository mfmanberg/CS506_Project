SHELL := /bin/bash

VENV ?= nyisoenv
PY := python

ifeq ($(OS),Windows_NT)
PIP_BIN := $(VENV)/Scripts/pip.exe
PY_BIN := $(VENV)/Scripts/python.exe
RM_CMD := powershell -Command "Remove-Item -Recurse -Force"
else
PIP_BIN := $(VENV)/bin/pip
PY_BIN := $(VENV)/bin/python
RM_CMD := rm -rf
endif

.PHONY: help venv install run-1stpass run-xgb run-xgb-postmid run-compare run-svr run-linear clean

help:
	@echo "Makefile targets:"
	@echo "  venv           -> create a virtualenv named $(VENV)"
	@echo "  install        -> install Python deps from requirements.txt into $(VENV)"
	@echo "  run-1stpass    -> execute data wrangling notebook (2_FIGURES/1_data_wrangling/1st_pass.ipynb)"
	@echo "  run-xgb        -> run XGBoost testing script (3_OUTPUT/3_xg_boost/XGBoost_testing.py)"
	@echo "  run-xgb-postmid-> run XGBoost post-mid script (3_OUTPUT/3_xg_boost/XGBoost_postmid.py)"
	@echo "  run-compare    -> run comparison metrics script (3_OUTPUT/3_xg_boost/comparison metrics.py)"
	@echo "  run-svr        -> execute SVR notebook (3_OUTPUT/3_svr/SVMDaily.ipynb)"
	@echo "  run-linear     -> execute linear regression notebook (3_OUTPUT/3_linear_regression/linear_regression.ipynb)"
	@echo "  clean          -> remove common output files and caches"

venv:
	@echo "Creating virtualenv $(VENV)..."
	$(PY) -m venv $(VENV)
	@echo "Virtualenv created. Activate with:"
	@echo "  On Windows: $(VENV)\\Scripts\\activate"
	@echo "  On macOS / Linux: source $(VENV)/bin/activate"

install: venv
	@echo "Installing requirements into $(VENV)..."
	$(PIP_BIN) install --upgrade pip
	$(PIP_BIN) install -r requirements.txt

# Execute the primary data-wrangling notebook (creates master parquet under 1_LIB/master if present)
run-1stpass:
	@echo "Executing 1st_pass.ipynb (data wrangling)..."
	jupyter nbconvert --to notebook --execute 2_FIGURES/1_data_wrangling/1st_pass.ipynb --ExecutePreprocessor.timeout=1800 --output 2_FIGURES/1_data_wrangling/1st_pass.ipynb

# Run XGBoost testing script
run-xgb:
	@echo "Running XGBoost testing script..."
	$(PY_BIN) 3_OUTPUT/3_xg_boost/XGBoost_testing.py

run-xgb-postmid:
	@echo "Running XGBoost postmid script..."
	$(PY_BIN) 3_OUTPUT/3_xg_boost/XGBoost_postmid.py

run-compare:
	@echo "Running comparison metrics script..."
	$(PY_BIN) "3_OUTPUT/3_xg_boost/comparison metrics.py"

# Execute SVR notebook (may require the venv to have nbconvert & dependencies installed)
run-svr:
	@echo "Executing SVMDaily.ipynb (SVR notebook)..."
	jupyter nbconvert --to notebook --execute 3_OUTPUT/3_svr/SVMDaily.ipynb --ExecutePreprocessor.timeout=3600 --output 3_OUTPUT/3_svr/SVMDaily.ipynb

run-linear:
	@echo "Executing linear_regression.ipynb..."
	jupyter nbconvert --to notebook --execute 3_OUTPUT/3_linear_regression/linear_regression.ipynb --ExecutePreprocessor.timeout=3600 --output 3_OUTPUT/3_linear_regression/linear_regression.ipynb

clean:
	@echo "Cleaning common outputs..."
	-$(RM_CMD) results_old.json results_new.json __pycache__
	@echo "Done."
