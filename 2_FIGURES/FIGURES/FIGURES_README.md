# CS506 Project - Figures and Visualizations Repository

This directory contains all exported figures, charts, and animations from the CS506 Energy Load Forecasting project.

**Last Updated:** 2025-12-13  
**Total Files:** 17 (11 PNG images, 6 GIF animations)

---

## 📂 Directory Structure

```
2_FIGURES/FIGURES/
├── FIGURES_README.md                         # This file
├── export_manifest.json                      # Complete file inventory with metadata
│
├── Geographic and Regional Data
│   └── nyiso_zones.png                       # NYISO geographic service zones map
│
├── Historical Load Patterns
│   ├── total_load_2023_15min.png            # 2023 load at 15-minute resolution
│   ├── day_by_day_jan_2023.png              # January 2023 daily load patterns
│   ├── load_with_losses.png                 # System load with transmission losses
│   ├── 2023_total_load_per_day.png          # Complete 2023 daily load profile
│   └── 2023_top10_names_avg_load.png        # Top 10 zones by average load
│
├── Model Performance Comparisons
│   ├── 3_xg_boost_baseline_performance.png  # XGBoost performance across time scales
│   ├── 3_xg_boost_model_comparison.png      # NYISO-only vs NYISO+MesoNet fusion
│   ├── svr_readme_graph.png                 # SVR confusion during load jumps
│   ├── vault_baseline_performance.png       # Baseline performance backup
│   └── vault_model_comparison.png           # Model comparison backup
│
├── Data Tables (CSV)
│   ├── 2023_daily_total_load.csv            # Daily total load values
│   ├── 2023_summary_stats.csv               # Statistical summaries
│   └── 2023_top10_names_avg_load.csv        # Top zones data
│
└── svr_animations/                           # SVR model animations (6 GIFs)
    ├── svr_5min_animation.gif               # 5-minute resolution predictions
    ├── svr_15min_animation.gif              # 15-minute resolution predictions
    ├── svr_hourly_animation.gif             # Hourly resolution predictions
    ├── svr_hourly_trunc_animation.gif       # Hourly with truncated dataset
    ├── svr_daily_weather_animation.gif      # Daily with weather features
    └── svr_daily_loadonly_animation.gif     # Daily load-only baseline
```

---

## 📊 Figure Descriptions

### Geographic and Regional Data

#### `nyiso_zones.png`
- **Description:** Map of New York State showing 11 NYISO geographic service zones
- **Purpose:** Visualize regional distribution of energy load across NY jurisdictions
- **Key Insight:** NYC zones dominate total state consumption (>60%)
- **Source:** NYISO official zone boundaries

---

### Historical Load Patterns

#### `total_load_2023_15min.png`
- **Description:** 2023 total energy load at 15-minute resolution
- **Resolution:** 15-minute intervals across entire year
- **Key Features:** Daily periodicity, weekend/weekday patterns, seasonal variation
- **Usage:** Demonstrates high-resolution temporal behavior and data granularity

#### `day_by_day_jan_2023.png`
- **Description:** Overlay of daily load patterns for January 2023
- **Key Features:** Consistent diurnal cycles, weather-driven variations
- **Usage:** Shows day-to-day stability and weather impact on load

#### `load_with_losses.png`
- **Description:** Complete system load accounting for transmission losses
- **Components:** Total load + grid inefficiency losses
- **Purpose:** Full accounting of energy generation requirements

#### `2023_total_load_per_day.png`
- **Description:** Complete 2023 daily load profile
- **Key Features:** Clear seasonal patterns, summer cooling peaks, winter heating demand
- **Usage:** Annual trend analysis and seasonal forecasting validation

#### `2023_top10_names_avg_load.png`
- **Description:** Bar chart of top 10 NYISO zones by average load (2023)
- **Key Insight:** NYC metropolitan area zones account for majority of consumption
- **Data Source:** Aggregated from zone-level NYISO data
- **Data File:** Corresponding CSV available in same directory

---

### Model Performance Comparisons

#### `3_xg_boost_baseline_performance.png`
- **Description:** XGBoost model performance across time aggregations
- **Metrics Shown:** MAPE, MAE, RMSE for 5-min, 15-min, hourly, daily aggregations
- **Key Result:** Sub-1% MAPE at fine time scales (<1 hour)
- **Source Notebook:** `3_OUTPUT/3_xg_boost/XGBoost_Testing_output.ipynb`

#### `3_xg_boost_model_comparison.png`
- **Description:** Comparative analysis of NYISO-only vs NYISO+MesoNet fusion
- **Key Finding:** Weather integration reduces daily MAPE from 4.77% → 3.11% (35% improvement)
- **Comparison Levels:** Five-minute, quarter-hourly, hourly, daily aggregations
- **Source Notebook:** `3_OUTPUT/3_xg_boost/ComparisonMetrics.ipynb`

#### `svr_readme_graph.png`
- **Description:** SVR model showing prediction lag during rapid load transitions
- **Key Observation:** Model struggles with sudden jumps (marked with red circles)
- **Usage:** Demonstrates limitations of smooth kernel-based approaches
- **Context:** Contrast with XGBoost's superior handling of discontinuities

---

### SVR Model Animations (6 GIFs)

All animations show a trailing window of predictions (red line) vs true values (blue line):

#### `svr_5min_animation.gif`
- **Resolution:** 5-minute intervals
- **MAPE:** ~0.28%
- **Features:** Excellent tracking of sub-hourly fluctuations
- **Window Size:** 100 data points
- **Source Notebook:** `3_OUTPUT/3_svr/SVMMinute.ipynb`

#### `svr_15min_animation.gif`
- **Resolution:** 15-minute intervals
- **MAPE:** ~0.35%
- **Features:** Balance between detail and visual clarity
- **Source Notebook:** `3_OUTPUT/3_svr/SVM15Min.ipynb`

#### `svr_hourly_animation.gif`
- **Resolution:** Hourly aggregation
- **MAPE:** ~0.28%
- **Features:** Optimal balance between accuracy and smoothness
- **Training Time:** ~5 hours
- **Source Notebook:** `3_OUTPUT/3_svr/SVMHourly.ipynb`

#### `svr_hourly_trunc_animation.gif`
- **Resolution:** Hourly aggregation
- **Dataset:** Truncated/reduced training set
- **Purpose:** Demonstrates impact of reduced training data on stability
- **Source Notebook:** `3_OUTPUT/3_svr/SVM_Trunc.ipynb`

#### `svr_daily_weather_animation.gif`
- **Resolution:** Daily aggregation
- **Features:** Includes MesoNet weather features (temperature, humidity, precipitation)
- **MAPE:** ~3.5%
- **Key Insight:** Weather features significantly improve tracking during extreme events
- **Source Notebook:** `3_OUTPUT/3_svr/SVMDaily.ipynb`

#### `svr_daily_loadonly_animation.gif`
- **Resolution:** Daily aggregation
- **Features:** Load-only baseline (no weather data)
- **MAPE:** ~5.2%
- **Purpose:** Baseline comparison to demonstrate weather feature value
- **Source Notebook:** `3_OUTPUT/3_svr/SVMDailywoutMeso.ipynb`

---

## 🔄 Reproducibility

### Regenerating All Figures

All figures can be regenerated using the automated export script:

```bash
cd C:\Users\Matt\Desktop\CS506\CS506_Project
python Build/export_figures_simple.py
```

**Script Features:**
- ✅ Uses absolute paths for full reproducibility
- ✅ Copies all visualization files from source directories
- ✅ Generates standardized filenames with descriptive prefixes
- ✅ Creates JSON manifest with complete metadata
- ✅ Preserves original files without modification

### Source Locations

Figures are collected from:
- `4_VAULT/` - Historical visualizations and maps
- `3_OUTPUT/3_xg_boost/` - XGBoost model outputs
- `3_OUTPUT/3_svr/` - SVR model outputs
- `2_FIGURES/FIGURES/svr_animations/` - SVR GIF animations (generated separately)

### SVR Animations

SVR animations are generated by:
```bash
cd C:\Users\Matt\Desktop\CS506\CS506_Project
python Build/export_svr_animations.py
```

See [Build/export_svr_animations.py](../../Build/export_svr_animations.py) for animation generation code.

---

## 📋 Export Manifest

A complete machine-readable inventory of all exported files is available in:
- **File:** [export_manifest.json](export_manifest.json)
- **Format:** JSON with file paths, counts, and metadata
- **Last Updated:** Automatically regenerated on each export run

**Manifest Contents:**
```json
{
  "export_date": "ISO 8601 timestamp",
  "project_root": "C:\\Users\\Matt\\Desktop\\CS506\\CS506_Project",
  "total_files": 17,
  "files": ["relative/path/to/file1.png", "..."],
  "summary": {
    "png_count": 11,
    "gif_count": 6,
    "csv_count": 0
  }
}
```

---

## 📖 Integration with Main README

All figures in this directory are referenced in the main [README.md](../../README.md) with standardized paths:

```markdown
![Description](2_FIGURES/FIGURES/filename.png)
```

This ensures:
- ✅ **Single source of truth** - All images centralized in one location
- ✅ **Absolute reproducibility** - Scripts use full paths from project root
- ✅ **Easy maintenance** - Update figures in one place
- ✅ **Version control friendly** - Clear organization for Git tracking

---

## 🛠 Dependencies

**Python Packages Required:**
- `matplotlib` - Static chart generation
- `numpy` - Numerical operations for animations
- `pillow` - GIF animation creation (PillowWriter)
- Standard library: `pathlib`, `shutil`, `json`

**Installation:**
```bash
pip install -r Dependencies/requirements.txt
```

---

## 📞 Support

For questions about:
- **Figure generation:** See individual notebook source files linked above
- **Export process:** Review [Build/export_figures_simple.py](../../Build/export_figures_simple.py)
- **Animation creation:** Review [Build/export_svr_animations.py](../../Build/export_svr_animations.py)
- **Model details:** See main [README.md](../../README.md) and model-specific notebooks

---

**Project:** CS506 Energy Load Forecasting  
**Institution:** Boston University  
**Repository:** [github.com/mfmanberg/CS506_Project](https://github.com/mfmanberg/CS506_Project)
