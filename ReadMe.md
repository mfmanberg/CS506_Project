# Support Vector Machine Regression for Load Prediction
- The model can be ran with 3_OUTPUT/3_svr/SVM_Trunc.ipynb
## Data Processing

The data was queried and aggregated using DuckDB.

``` python
SELECT "Time Stamp", Load
FROM read_parquet('./../../1_LIB/nyiso/nyiso_parquet/**/*.parquet')
```

### Aggregation and Cleaning
#### Aggregation:
- All regional load values were aggregated by timestamp to compute the total energy loads across regions.

    ``` python
    df_total_load = df.groupby("Time Stamp", as_index=False)["Load"].sum()
    ```

#### DateTime Processesing
- The Time Stamp column was converted to datetime format and resampled to an hourly frequency to obtain hourly average load values.
    ``` python
    df_total_load['Time'] = pd.to_datetime(df_total_load['Time'])
    df_hourly = df_total_load.resample('1H').mean().dropna().reset_index()
    ```
#### Data Splitting
- The dataset was divded chronologically into:
    - Training 2001 - 2021
    - Validation: 2022
    - Testing: 2023 - 2025

## Data Modeling Methods
### Feature Scaling
- All load values were standardized with StandardScaler from scikit-learn.

    ``` python
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    ```
### Lag Feature Construction
- Because energy load exhibits temporal dependencies, the model was trained using a sliding window approach with the size of the window being 5. Prediction is based on the values of the past 5 hours.

    ``` python
    TIME_STEPS = 5
    X_train, y_train = create_dataset(train_scaled, train_scaled, TIME_STEPS)
    ```
### Model Selection
- A support Vector Regression model with a rbf kernel was chosen due to its ability to model non linear relationships between past and future load values.
- Hypterparameter Tuning:
    - A grid search was done over:
        - C in {0.1, 1, 10}
        - gamma in {'scale', 0.01, 0.001}
        - epsilon in {0.01, 0.1, 0.5, 1.0}
    - Grid search was doneon a subset of the training data of size 40000.

    - The best hyperparameters found were:
        - {'C': 10, 'cache_size': 200, 'coef0': 0.0, 'degree': 3, 'epsilon': 0.01, 'gamma': 0.01, 'kernel': 'rbf', 'max_iter': -1, 'shrinking': True, 'tol': 0.001, 'verbose': False}
### Model Evaluation
- Our current SSVR model profuced the following metrics on Testing Data:
    - Mean Absolute Error (MAE) of 170.34232309986712
    - Root Mean Squared Error (RMSE) of 347.6162200976427
    - Mean Absolute Percentage Error (MAPE) of 1%
    - R^2 Score of 0.9872711260207307 

## Other Observations
- Our current SVR model takes roughly an hour to train. As shown in the upper graph, its predictions largely follow the trends of the true load values. However, as shown in the lower graph there are areas of large jumps that cause the model confusion as it will jump in the proper direction and subsequently jump in the opposite direction. 

![alt text](SVM_READMe_Graph.png)

