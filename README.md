# CS506 Project Midterm Report: Energy Load Forecasting

## Youtube Video

https://youtu.be/EbO-ILR7YLY

**Description of the project**

NYISO was birthed out of a catastrophic power outage, costing the American public millions and resulting in deaths. They have their own forecasts (they release publicly and utilize similar methodology as the private utility companies). They oversee all of NY's jurisdictions, with an imperfect picture of (my guess due to poor data sharing common in utilities) of when new load is introduced or removed in addition to other noise. This forecast is important to prevent future catastrophe. 

*NYISO Zones, Source: https://www.nyiso.com/real-time-dashboard.*


Utility companies profit is already negotiated between the state and them in the rate case. They legally cannot charge more for what they buy, they can only charge utility bills for the Distribution and carry over the buying cost. A better forecast would result in less spot buying, and save the ratepayer (the person who pays the utility bill) millions of dollars a day in addition  to further informing NYISO’s important oversight.

Previous forecasts are rooted in a deterministic methodology despite the system acting as a non-linear chaotic environment. An empirical, dynamic, and inductive data-driven approach such as Deep Learning may prove to outcompete current forecasts. Business events, from outages, industrial load spikes, residential load spikes, etc… cause a sudden seemingly-stochastic drop in load. A decision-tree may prevent further error from switching models (such as the criteria of 10% error given a time-period). 

**Clear goal(s) (e.g. Successfully predict the number of students attending lecture based on the weather report).**

There are two main goals of this project:

1. Explore data behavior of NY’s Energy Load (ACF, business events, etc…).
2. **Attempt to outcompete NYISO’s time-series forecasting of Energy Load** on an hourly or more granular scale on an aggregate or zone basis.

## Preliminary Visualizations

![alt text](TotalLoad2023Day15Min.png)
![alt text](DayByDayJan2023.png)'


# Data Processing

Web-scraped NYISO Data

Aggregated to 5 minute, 15 minute, Hourly and Daily time-scales. 

Learned to use parquets for effiecent computation and github for large data. 


## Linear Regression 

Used to tease data behavior and act as baseline. Evident big spikes in data behavior/"business events" have large effect.  Right-skewed residual distribution. At 15 minute time-scale aggregation:

Prediction window summary
  Range: 2023-01-01 00:00:00 -> 2024-12-31 23:45:00  (n=70,176)
  Actual:
    mean=52002.218, std=11464.106, min=0.000, max=225136.700
  Pred:
    mean=48757.355, std=473.322, min=47937.556, max=49577.154
  Errors:
    MAE=7818.404, MSE=142767659.736, RMSE=11948.542, MAPE=13.763%
    Bias (mean residual)=-3244.863
  Fit:
    R^2=-0.0863, Pearson r=-0.0544, Spearman ρ=-0.0575

![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)


## Support Vector Machine Regression Model for Load Prediction
- The model can be ran with 3_OUTPUT/3_svr/SVM_Trunc.ipynb

### SVM Data Processing

The data was queried and aggregated using DuckDB.

``` python
SELECT "Time Stamp", Load
FROM read_parquet('./../../1_LIB/nyiso/nyiso_parquet/**/*.parquet')
```

#### Aggregation and Cleaning
##### Aggregation:
- All regional load values were aggregated by timestamp to compute the total energy loads across regions.

    ``` python
    df_total_load = df.groupby("Time Stamp", as_index=False)["Load"].sum()
    ```

##### DateTime Processesing
- The Time Stamp column was converted to datetime format and resampled to an hourly frequency to obtain hourly average load values.
    ``` python
    df_total_load['Time'] = pd.to_datetime(df_total_load['Time'])
    df_hourly = df_total_load.resample('1H').mean().dropna().reset_index()
    ```
##### Data Splitting
- The dataset was divded chronologically into:
    - Training 2001 - 2021
    - Validation: 2022
    - Testing: 2023 - 2025

### Data Modeling Methods
#### Feature Scaling
- All load values were standardized with StandardScaler from scikit-learn.

    ``` python
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    ```
#### Lag Feature Construction
- Because energy load exhibits temporal dependencies, the model was trained using a sliding window approach with the size of the window being 5. Prediction is based on the values of the past 5 hours.

    ``` python
    TIME_STEPS = 5
    X_train, y_train = create_dataset(train_scaled, train_scaled, TIME_STEPS)
    ```
#### Model Selection
- A support Vector Regression model with a rbf kernel was chosen due to its ability to model non linear relationships between past and future load values.
- Hypterparameter Tuning:
    - A grid search was done over:
        - C in {0.1, 1, 10}
        - gamma in {'scale', 0.01, 0.001}
        - epsilon in {0.01, 0.1, 0.5, 1.0}
    - Grid search was doneon a subset of the training data of size 40000.

    - The best hyperparameters found were:
        - {'C': 10, 'cache_size': 200, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 0.01, 'kernel': 'rbf', 'max_iter': -1, 'shrinking': True, 'tol': 0.001, 'verbose': False}
#### Model Evaluation
- Our current SSVR model profuced the following metrics on Testing Data:
    - Mean Absolute Error (MAE) of 170.34232309986712
    - Root Mean Squared Error (RMSE) of 347.6162200976427
    - Mean Absolute Percentage Error (MAPE) of 1%
    - R^2 Score of 0.9872711260207307 

### Other Observations
- Our current SVR model takes roughly an hour to train. As shown in the upper graph, its predictions largely follow the trends of the true load values. However, as shown in the lower graph there are areas of large jumps that cause the model confusion as it will jump in the proper direction and subsequently jump in the opposite direction. 

![alt text](SVM_READMe_Graph.png)


---
<br>

# XGBoost Regression for Load Prediction

* The model can be run from `3_OUTPUT/3_xg_boost/XGBoost_postmid.py`

### (OLD MODEL) XGBoost on NYISO data
### Data Processing

The data was queried and aggregated using DuckDB, sourced from the parquet files stored in
`1_LIB/nyiso/nyiso_parquet/`.

Each parquet file contains timestamped load values for different NYISO regions.

#### Aggregation and Cleaning

##### Aggregation

* All regional load values were filtered by the selected station (e.g., **LONGIL**, **GENESE**, etc) and then resampled to multiple time granularities:

  ```python
  raw = df[['Load']].copy()
  five = raw.resample('5min').mean()
  quarter = raw.resample('15min').mean()
  hourly = raw.resample('1h').mean()
  daily = raw.resample('1d').mean()
  ```
* These resampled datasets allowed the model to evaluate how different temporal resolutions affect prediction performance, identifying cyclic trends in different resolutions.

##### Cleaning

* Duplicate timestamps were dropped after being gathered for the specified station to ensure consistent time indices. This column is then set as index, in datetime format

  ```python
  df['Time Stamp'] = pd.to_datetime(df['Time Stamp'])
  df = df.drop_duplicates(subset=['Time Stamp']).set_index('Time Stamp')
  ```

##### Data Splitting

* The dataset was split chronologically:

  * **Training:** 2001–2021
  * **Validation:** 2022
  * **Testing:** 2023–2025

---

### Data Modeling Methods

#### Lag Feature Construction

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

An XGBoost Regressor was selected for its efficiency and ability to model non-linear temporal relationships.


#### Hyperparameter Configuration

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

| Aggregation | MAPE    | MAE       | RMSE      | R²       | Time (s)  |
|------------|---------|-----------|-----------|----------|-----------|
| raw        | 0.27 | 4.12  | 5.72  | 0.99959 | 30.43 |
| five       | 0.37 | 5.61  | 7.6  | 0.999267 | 26.9 |
| quarter    | 0.47 | 7.45  | 10.11 | 0.998703 | 10.25 |
| hourly     | 2.7 | 42.07 | 53.79 | 0.963164 | 1.48  |
| daily      | 4.77 | 74.86 | 100.51| 0.800868 | 0.18  |


**Total runtime:** 154.98 seconds (2.58 minutes)

---




## (Current Model) XGBoost on Fused NYISO + MesoNet data


### Data Processing
The data was sourced from the master mesonet parquet.

#### Cleaning
Since the data was already cleaned, only a basic forward fill for NA values and handling the datetime column was done. 
#### Data Splitting

* The dataset was split chronologically:

  * **Training:** 2001–2021
  * **Validation:** 2022
  * **Testing:** 2023–2025

### Data Modeling Methods
Since the raw mesonet data is in 5 minute aggregations, we use five minute, quarterly, hourly and daily aggregations.

#### Lag Feature Construction
Using recognizable features on each aggregation (ex: for five minutes, lagging by 12 intervals would lag by an hour) similar to the periodicity in the older model, a grid search was constructed:

```python

LAG_GRID = {
    'raw': [
        [1, 5, 15, 60]  # we don't use raw, but leave one option just in case
    ],
    'five': [
        [1, 5, 15],          # short-term only
        [1, 5, 15, 60],      # add 5 hours
        [1, 12, 36, 72]      # 1h, 3h, 6h
    ],
    'quarter': [
        [1, 7, 30],          
        [1, 3, 7, 30],       
        [1, 7, 30, 90]       
    ],
    'hourly': [
        [1, 24, 168],        # 1 hour, day, week
        [1, 24, 72, 168],    # 3 days
        [1, 24, 168, 336]    # 2 weeks 
    ],
    'daily': [
        [1, 7, 30],          # 1 day, week, month
        [1, 3, 7, 30],       # 3 days
        [1, 7, 30, 90]       # 3 month
    ]
}
```
### Model Selection
The model used here was an XGBoost Regressor as well.
#### Hyperparameter Configuration
After running Cross Validation on the Five minute aggregation, the following model hyperparameters were chosen.

```python
"model": {
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
The subsample and colsample parameters were chosen to decorrelate the trees, whereas the early stopping rounds parameter was set to prevent over complication and overfitting of the tree structure.

### Model Evaluation
The evaluation metrics are:

| Aggregation | MAPE | MAE   | RMSE  | R²       | Time (s) |
|------------|------|-------|-------|----------|----------|
| five       | 0.27 | 4.16  | 5.86  | 0.99956  | 23.30    |
| quarter    | 0.37 | 5.81  | 8.08  | 0.99918  | 9.96     |
| hourly     | 1.63 | 24.98 | 32.84 | 0.98625  | 4.64     |
| daily      | 3.11 | 48.44 | 65.43 | 0.91405  | 1.05     |

**Total time:** 210.38 seconds

## Model Comparison

Although the fused model takes longer to run, it outperforms the old model, especially in the coarser aggregations such as hourly and daily:


![alt text](<comparison stats new.png>)

Thus, adding an additional modality helps the model learn trends faster and more efficiently.