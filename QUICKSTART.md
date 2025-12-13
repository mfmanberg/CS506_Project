# CS506 Project - Quick Start Guide

## 🚀 One-Command Setup

**Just double-click:**
```
setup.bat
```

**Or run from terminal:**

*Command Prompt:*
```cmd
setup.bat
```

*PowerShell:*
```powershell
.\setup.bat
```

That's it! The script will:
- ✅ Check Python installation
- ✅ Create virtual environment (if needed)
- ✅ Install all dependencies (if needed)
- ✅ Verify everything works
- ✅ Activate the environment automatically

**No interaction required!** The script runs completely automatically.

## 📋 What Happens

### First Time (5-10 minutes)
1. Creates `venv/` directory
2. Installs ~30 Python packages
3. Verifies installation
4. Opens activated terminal

### Subsequent Runs (< 10 seconds)
1. Detects existing environment
2. Verifies packages are installed
3. Opens activated terminal

## 💻 Daily Usage

### Option 1: Use the activated window
After running `setup.bat`, the terminal stays open and activated.

Just type your commands:
```cmd
make status
make run-analysis
jupyter notebook
```

### Option 2: Quick activation
```cmd
setup.bat activate
```
Opens new activated terminal.

### Option 3: Run make commands directly
```cmd
setup.bat make status
setup.bat make run-analysis
setup.bat make list-status
```

### Option 4: Manual activation
```cmd
venv\Scripts\activate.bat
make status
```

## 🔧 What Gets Installed

All packages from `requirements.txt`:

**Data Science:** NumPy, Pandas, SciPy, Matplotlib, Seaborn, Plotly
**Machine Learning:** Scikit-learn, XGBoost  
**Jupyter:** Notebook, nbconvert, IPython  
**Data I/O:** PyArrow, FastParquet  
**Utilities:** Requests, BeautifulSoup4, tqdm, pytz

## ❓ Troubleshooting

### "Python is not installed"
1. Install Python: https://www.python.org/downloads/
2. **Important:** Check "Add Python to PATH"
3. Run `setup.bat` again

### Need to reinstall?
```cmd
rmdir /s /q venv
setup.bat
```

### Packages not found in notebooks?
Make sure you're using the activated environment:
```cmd
venv\Scripts\activate.bat
jupyter notebook
```

## 📁 Files Created

- **`venv/`** - Virtual environment (auto-created)
- **`setup.bat`** - All-in-one script (setup, activate, make)
- **`requirements.txt`** - Package list

## ✨ Pro Tips

1. **First time?** Just run `setup.bat`
2. **Daily use?** Run `setup.bat activate` or use the activated window
3. **Quick make?** Run `setup.bat make status`
4. **VS Code?** It will auto-detect `venv/` - just select it as interpreter
5. **Jupyter?** Run `jupyter notebook` in activated window

## 🔧 All Usage Modes

```cmd
setup.bat                  # Full setup (first time)
setup.bat                  # Verify and activate (subsequent)
setup.bat activate         # Just activate venv
setup.bat make status      # Activate + run make status
setup.bat make run-analysis # Activate + run make run-analysis
```

---

**TL;DR: `setup.bat` does everything!** 🎉
