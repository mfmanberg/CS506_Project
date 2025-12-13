# CS506 NYISO Load Forecasting Project

**Organized Structure with Automated Testing and CI/CD**

## 📁 Project Organization

```
CS506_Project/
├── Dependencies/           # All dependency management
│   ├── requirements.txt    # Python packages (125 total)
│   ├── README.md          # Installation guide
│   └── test_dependencies.sh
│
├── Build/                 # Build automation (renamed from Makefile/)
│   ├── Makefile.wsl       # Linux/WSL makefile
│   ├── RunNotebooks.ps1   # Windows PowerShell wrapper
│   ├── path_utils.py      # Cross-platform paths
│   ├── extract_results.py # Result extraction
│   ├── test_execution.py  # Execution verification
│   └── README.md          # Build documentation
│
├── GitHub_Workflows/      # CI/CD automation
│   ├── README.md          # Workflow documentation
│   ├── workflows_*.yml    # Workflow definitions
│   └── (deployed to .github/workflows/)
│
├── 1_LIB/                 # Data files
│   └── master/
│       └── master.parquet # 38MB dataset (Git LFS)
│
├── 3_OUTPUT/              # Analysis notebooks
│   ├── 3_linear_regression/
│   ├── 3_svr/
│   └── 3_xg_boost/
│
└── model_results.log      # Execution results
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# WSL/Linux
cd Dependencies
pip install -r requirements.txt

# Or use the test script
chmod +x test_dependencies.sh
./test_dependencies.sh
```

### 2. Run Notebooks

**Windows:**
```powershell
.\Build\RunNotebooks.ps1
```

**WSL/Linux:**
```bash
make -f Build/Makefile.wsl
```

### 3. View Results
```bash
cat model_results.log
```

## 📊 GitHub Actions (CI/CD)

All workflows automatically run on push to `main`:

- ✅ **Linear Regression** - 4m 5s
- ✅ **SVM Daily** - 1m 23s  
- ✅ **XGBoost Testing** - 1m 34s

View at: https://github.com/mfmanberg/CS506_Project/actions

## 📖 Documentation

Each directory has its own README with detailed instructions:

- **[Dependencies/README.md](Dependencies/README.md)** - Setup & installation
- **[Build/README.md](Build/README.md)** - Build & execution
- **[GitHub_Workflows/README.md](GitHub_Workflows/README.md)** - CI/CD workflows

## 🎯 Key Features

✅ **Fully Automated** - One command to run all notebooks  
✅ **Cross-Platform** - Works on Windows, WSL, Linux, GitHub Actions  
✅ **Reproducible** - All paths are absolute and dynamic  
✅ **Well-Documented** - README in every directory  
✅ **CI/CD Integrated** - Automated testing on every push  
✅ **Git LFS** - Large files handled properly  

## 🧪 Testing

**Test dependencies:**
```bash
cd Dependencies && ./test_dependencies.sh
```

**Test folder structure:**
```bash
./test_structure.sh
```

**Test build system:**
```bash
make -f Build/Makefile.wsl help
```

**Test single notebook:**
```bash
python Makefile/test_execution.py <notebook_path>
```

**Test GitHub CLI:**
```bash
gh run list --limit 5
```

## 📦 Requirements

- **Python:** 3.12+
- **Virtual Environment:** `.venv_wsl/` (WSL) or `.venv/` (Windows)
- **WSL:** For Windows users (recommended)
- **Git LFS:** For large file handling
- **GitHub CLI:** For workflow management (optional)

## 🏆 Performance

| Model | Dataset | RMSE | R² | Execution Time |
|-------|---------|------|----|----|
| **Linear Regression** | Full | - | - | ~240s |
| **SVM Daily** | Daily+Weather | 108.99 | 0.7610 | ~90s |
| **SVMDailywoutMeso** | Daily only | 83.91 | 0.8591 | ~90s |
| **XGBoost PostMid** | Full | 64.18 | - | ~180s |
| **XGBoost Testing** | Test set | 96.33 | - | ~60s |

## 📝 License

CS506 Course Project - Boston University

## 👥 Contributors

- Matt Manberg (@mfmanberg)

---

**Last Updated:** December 13, 2025  
**Status:** ✅ All systems operational
