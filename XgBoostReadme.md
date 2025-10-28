# XGBoost Regression for Load Prediction

* The model can be run from `3_OUTPUT/3_xg_boost/XGBoost_testing.py`

## Data Processing

The data was queried and aggregated using DuckDB, sourced from the parquet files stored in
`1_LIB/nyiso/nyiso_parquet/`.

Each parquet file contains timestamped load values for different NYISO regions.

### Aggregation and Cleaning

#### Aggregation

* All regional load values were filtered by the selected station (e.g., **LONGIL**, **GENESE**, etc) and then resampled to multiple time granularities:

  ```python
  raw = df[['Load']].copy()
  five = raw.resample('5min').sum()
  quarter = raw.resample('15min').sum()
  hourly = raw.resample('1h').sum()
  daily = raw.resample('1d').sum()
  ```
* These resampled datasets allowed the model to evaluate how different temporal resolutions affect prediction performance, identifying cyclic trends in different resolutions.

#### Cleaning

* Duplicate timestamps were dropped after being gathered for the specified station to ensure consistent time indices. This column is then set as index, in datetime format

  ```python
  df['Time Stamp'] = pd.to_datetime(df['Time Stamp'])
  df = df.drop_duplicates(subset=['Time Stamp']).set_index('Time Stamp')
  ```

#### Data Splitting

* The dataset was split chronologically:

  * **Training:** 2001–2021
  * **Validation:** 2022
  * **Testing:** 2023–2025

---

## Data Modeling Methods

### Lag Feature Construction

* Each aggregate level used different lag windows to capture temporal dependencies:

  ```python
  AGG_LAGS = {
      'raw': [1, 5, 15, 60],
      'five': [1, 7, 30],
      'quarter': [1, 7, 30],
      'hourly': [1, 24, 168],
      'daily': [1, 7, 30]
  }
  ```
* These lags were chosen based on the expected periodicity of load patterns (minutes, hours, or days).
* For example:

  * `raw`: minute-level fluctuations
  * `hourly`: daily and weekly cycles
  * `daily`: long-term seasonal trends

### Model Selection

* **XGBoost Regressor** was selected for its efficiency and ability to model non-linear temporal relationships.
* It uses **gradient boosting** over decision trees to minimize prediction error iteratively.

### Hyperparameter Configuration

The model used the following tuned parameters:

```python
{
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "early_stopping_rounds": 20,
    "random_state": 42
}
```

### Model Evaluation

Each aggregate level’s model was trained and tested individually.
The metrics computed include:

* **MAPE (Mean Absolute Percentage Error)**
* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² (Coefficient of Determination)**
* **Runtime (seconds)**

| Aggregation | MAPE  | MAE      | RMSE     | R²     | Time (s) |
| ----------- | ----- | -------- | -------- | ------ | -------- |
| raw         | 0.69  | 14.43    | 23.94    | 0.9986 | 12.74    |
| five        | 3.55  | 100.14   | 456.19   | 0.676  | 6.83     |
| quarter     | 3.41  | 283.15   | 943.72   | 0.814  | 7.65     |
| hourly      | 6.46  | 1823.64  | 2910.58  | 0.875  | 0.52     |
| daily       | 12.77 | 52313.52 | 74073.13 | 0.772  | 0.12     |

**Total runtime:** 188.26 seconds (3.14 minutes)

---

## Other Observations

* **Raw data (no aggregation)** runs the slowest (12.7 s) but gives near-perfect performance since it learns all of the fine-scale dependencies in the data.

* **5-minute and 15-minute aggregations** smooth out high-frequency noise, which reduces precision a lot since the model is not fine enough to pick up details like the raw data but not large enough to capture broader trends but speeds up computation (6.8 and 7.6 s).
* **Hourly aggregation** removes minute-level fluctuations and highlights cyclic variation like day and night peaks. Thus, the model captures long-term structure better (higher R²) but is less precise point-to-point (higher MAPE).
* **Daily aggregation** continues this pattern and outputs a smoother signal, with an R² of about 0.77. However, with fewer samples due to aggregating, the model can’t capture finer variations, thus explaining the higher error values