import os, time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import json



# CONFIG

MASTER_PATH = r"C:\code\python\nyiso_project\CS506_Project\mesonet_master\master.parquet" #change this path #######################

CONFIG = {
    "train_start": 2001,
    "train_end": 2021,
    "val_year": 2022,
    "test_years": [2023, 2024, 2025],

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
}

AGG_LAGS = {
    'raw':     [1, 5, 15, 60], #just leave this here, we dont use raw anyway
    'five':    [1, 5, 15, 60],    
    'quarter': [1, 7, 30],
    'hourly':  [1, 24, 168],
    'daily':   [1, 7, 30]
}


# LOAD 

def load_master_parquet():
    print(" Loading fusion dataset...")
    df = pd.read_parquet(MASTER_PATH)

    # rename datetime
    df.rename(columns={"datetime": "Time_Stamp"}, inplace=True)
    df["Time_Stamp"] = pd.to_datetime(df["Time_Stamp"], utc=True)
    df = df.sort_values("Time_Stamp").set_index("Time_Stamp")

    # clean column names
    df.columns = (
        df.columns.str.replace(r"[\[\]\(\)/ ]", "_", regex=True)
                  .str.replace(r"__+", "_", regex=True)
                  .str.strip("_")
    )

    # split weather + load
    weather_cols = [c for c in df.columns if c not in ["Load", "PTID"]]

    # fill NAs safely (forward fill → 0)
    df[weather_cols] = df[weather_cols].ffill().fillna(0)

    print(f" Loaded fusion dataset: {len(df):,} rows")
    print("Columns:", df.columns.tolist()[:10], "...")

    #debyg
    print("\n First 8 rows of raw data:")
    print(df.head(8))

    #debug text to find all column names

    print("\n Columns:")
    print(df.columns.tolist())

    print("\n Time difference between first 10 rows:")
    print(df.index.to_series().diff().head(10))


    

    return df


# AGGREGATIONS

def create_aggregates(df):
    print("\n Creating aggregates...")

    weather_cols = [c for c in df.columns if c not in ["Load", "PTID"]]

    raw = df.copy()
    five = df.copy()

    quarter = df.resample("15min").agg({
        **{c: "mean" for c in weather_cols},
        "Load": "mean"
    })

    hourly = df.resample("1h").agg({
        **{c: "mean" for c in weather_cols},
        "Load": "mean"
    })

    daily = df.resample("1d").agg({
        **{c: "mean" for c in weather_cols},
        "Load": "mean"
    })

    # DEBUG PRINT
    print("\n Quarter-hour sample (first 5 rows):")
    print(quarter.head())

    return {
        'raw': raw,
        'five': five,
        'quarter': quarter,
        'hourly': hourly,
        'daily': daily
    }


# XGBOOST 

def run_xgboost(df, lags, agg_name):
    start = time.time()

    df = df.copy().reset_index()
    df["year"] = df["Time_Stamp"].dt.year

    # lag features
    for lag in lags:
        df[f"lag_{lag}"] = df["Load"].shift(lag)

    df = df.dropna()

    # weather columns
    weather_cols = [c for c in df.columns if c not in [
        "Time_Stamp", "Load", "PTID", "year"
    ] and not c.startswith("lag_")]

    lag_cols = [f"lag_{l}" for l in lags]
    feature_cols = weather_cols + lag_cols

    # splits
    train = df[(df["year"] >= CONFIG["train_start"]) & (df["year"] <= CONFIG["train_end"])]
    val = df[df["year"] == CONFIG["val_year"]]
    test = df[df["year"].isin(CONFIG["test_years"])]

    X_train = train[feature_cols]
    y_train = train["Load"]
    X_val = val[feature_cols]
    y_val = val["Load"]
    X_test = test[feature_cols]
    y_test = test["Load"]

    print(f"\n [{agg_name}] Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")

    model = xgb.XGBRegressor(**CONFIG["model"])
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mask = y_test != 0
    mape = (abs((y_test[mask] - y_pred[mask]) / y_test[mask])).mean() * 100

    print(f" MAPE={mape:.3f}% | MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.4f}")

    # DEBUG
    plt.figure(figsize=(14, 6))
    plt.plot(test["Time_Stamp"], y_test, label="Actual", alpha=0.6)
    plt.plot(test["Time_Stamp"], y_pred, label="Predicted", alpha=0.8)
    plt.title(f"{agg_name} — Actual vs Predicted Load")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "MAPE": mape,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Time_s": time.time() - start
    }


# MAIN

if __name__ == "__main__":
    total_start = time.time()

    df_master = load_master_parquet()
    AGG_DFS = create_aggregates(df_master)

    results = {}
    for agg_name, df_agg in AGG_DFS.items():
        results[agg_name] = run_xgboost(df_agg, AGG_LAGS[agg_name], agg_name)

    print("\n================ SUMMARY ================")
    print(pd.DataFrame(results).T)

    print(f"\n Total runtime: {time.time() - total_start:.2f} seconds")

    #json dump for comparison
    with open("results_new.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Saved → results_new.json")
    print(pd.DataFrame(results).T)