import os, time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import json


# CONFIG

MASTER_PATH = r"C:\code\python\nyiso_project\CS506_Project\mesonet_master\master.parquet"  # change this path if needed

CONFIG = {
    "train_start": 2001,
    "train_end": 2021,
    "val_year": 2022,
    "test_years": [2023, 2024, 2025],

    # FIXED model config
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

    # debug
    print("\n First 8 rows of raw data:")
    print(df.head(8))

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

def run_xgboost(df, lags, agg_name, plot=False):
    """
    Uses FIXED model hyperparameters from CONFIG['model'].
    Only lags are being grid-searched.
    """
    start = time.time()

    df = df.copy().reset_index()
    df["year"] = df["Time_Stamp"].dt.year

    # lag features
    for lag in lags:
        df[f"lag_{lag}"] = df["Load"].shift(lag)

    df = df.dropna()

    # weather columns
    weather_cols = [
        c for c in df.columns
        if c not in ["Time_Stamp", "Load", "PTID", "year"]
        and not c.startswith("lag_")
    ]

    lag_cols = [f"lag_{l}" for l in lags]
    feature_cols = weather_cols + lag_cols

    # splits
    train = df[
        (df["year"] >= CONFIG["train_start"]) &
        (df["year"] <= CONFIG["train_end"])
    ]
    val = df[df["year"] == CONFIG["val_year"]]
    test = df[df["year"].isin(CONFIG["test_years"])]

    X_train = train[feature_cols]
    y_train = train["Load"]
    X_val = val[feature_cols]
    y_val = val["Load"]
    X_test = test[feature_cols]
    y_test = test["Load"]

    print(f"\n [{agg_name}] Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")
    print(f" Using lags={lags} and fixed params={CONFIG['model']}")

    model_cfg = CONFIG["model"].copy()
    model = xgb.XGBRegressor(**model_cfg)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # predictions
    y_val_pred = model.predict(X_val)
    y_pred = model.predict(X_test)

    # validation metrics
    rmse_val = root_mean_squared_error(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    mask_val = y_val != 0
    mape_val = (abs((y_val[mask_val] - y_val_pred[mask_val]) / y_val[mask_val])).mean() * 100

    # test metrics
    rmse_test = root_mean_squared_error(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    mask_test = y_test != 0
    mape_test = (abs((y_test[mask_test] - y_pred[mask_test]) / y_test[mask_test])).mean() * 100

    elapsed = time.time() - start

    print(f" VAL   → MAPE={mape_val:.3f}% | MAE={mae_val:.2f} | RMSE={rmse_val:.2f} | R2={r2_val:.4f}")
    print(f" TEST  → MAPE={mape_test:.3f}% | MAE={mae_test:.2f} | RMSE={rmse_test:.2f} | R2={r2_test:.4f}")
    print(f" Time  → {elapsed:.2f} seconds")

    if plot:
        # DEBUG PLOT for the chosen "best" config only
        plt.figure(figsize=(14, 6))
        plt.plot(test["Time_Stamp"], y_test, label="Actual", alpha=0.6)
        plt.plot(test["Time_Stamp"], y_pred, label="Predicted", alpha=0.8)
        plt.title(f"{agg_name} — Actual vs Predicted Load\nlags={lags}")
        plt.legend()
        plt.tight_layout()
        plt.show()


    return {
        # validation metrics
        "MAPE_val": mape_val,
        "MAE_val": mae_val,
        "RMSE_val": rmse_val,
        "R2_val": r2_val,

        # test metrics
        "MAPE_test": mape_test,
        "MAE_test": mae_test,
        "RMSE_test": rmse_test,
        "R2_test": r2_test,

        # aliases to match results_old.json
        "MAPE": mape_test,
        "MAE": mae_test,
        "RMSE": rmse_test,
        "R2": r2_test,

        "Time_s": elapsed
    }


def gridsearch_for_agg(df, agg_name):

    lag_candidates = LAG_GRID[agg_name]
    print(f"[{agg_name}] Using fixed hyperparameters: {CONFIG['model']}")
    print(f"[{agg_name}] Lag candidates: {lag_candidates}")

    best = None
    best_lags = None

    for lags in lag_candidates:
        print(f"\n=== [{agg_name}] Trying lags={lags} ===")
        metrics = run_xgboost(df, lags, agg_name, plot=False)

        metrics = metrics.copy()
        metrics["lags"] = lags
        metrics["params"] = CONFIG["model"].copy()

        if (best is None) or (metrics["MAPE_val"] < best["MAPE_val"]):
            best = metrics
            best_lags = lags

    print(f"\n>>> Best config for {agg_name}:")
    print(f"    lags   = {best_lags}")
    print(f"    params = {CONFIG['model']}")
    print(f"    VAL MAPE = {best['MAPE_val']:.3f}% | TEST MAPE = {best['MAPE']:.3f}%")

    # One final plot for the winning config
    _ = run_xgboost(df, best_lags, agg_name, plot=True)

    best["lags"] = best_lags
    best["params"] = CONFIG["model"].copy()
    return best


# MAIN

if __name__ == "__main__":
    total_start = time.time()

    df_master = load_master_parquet()
    AGG_DFS = create_aggregates(df_master)

    results = {}

    for agg_name, df_agg in AGG_DFS.items():
        if agg_name == "raw":
            # skip raw
            print("\n[raw] aggregation skipped (not used in final analysis).")
            continue

        print(f"\n########## GRID SEARCH for {agg_name.upper()} ##########")
        results[agg_name] = gridsearch_for_agg(df_agg, agg_name)

    # Build a clean summary (TEST metrics only)
    summary = {}
    for agg_name, res in results.items():
        summary[agg_name] = {
            "MAPE": res["MAPE"],
            "MAE": res["MAE"],
            "RMSE": res["RMSE"],
            "R2": res["R2"],
            "Time_s": res["Time_s"],
        }

    print("\n================ SUMMARY ================")
    summary_df = pd.DataFrame(summary).T[["MAPE", "MAE", "RMSE", "R2", "Time_s"]]
    print(summary_df)
    print(f"\n Total runtime: {time.time() - total_start:.2f} seconds")

    # Print optimal lags separately for your methods section
    print("\nOptimal lags per aggregation:")
    for agg_name, res in results.items():
        print(f" {agg_name}: {res['lags']}")

    # Save JSON in the same format as results_new.json
    with open("results_new.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved → results_new.json")
