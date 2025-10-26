import duckdb
import matplotlib.pyplot as plt
import pandas as pd

def analyze_station(dataset_dir, station, window=30):
    """
    Analyze and visualize load data for a given station using DuckDB.
    
    Args:
        dataset_dir (str): Path to partitioned parquet dataset.
        station (str): Station name to filter (e.g., 'GENESE').
        window (int): Window size for rolling mean smoothing (days).
    """
    con = duckdb.connect()

    # Query only the relevant station across all parquets
    df = con.execute(f"""
        SELECT "Time Stamp", Load
        FROM parquet_scan('{dataset_dir}/**/*.parquet')
        WHERE Name = '{station}'
        ORDER BY "Time Stamp"
    """).df()

    if df.empty:
        print(f"No data found for station '{station}'")
        return

    # Convert timestamp
    df["Time Stamp"] = pd.to_datetime(df["Time Stamp"], errors="coerce")
    df = df.dropna(subset=["Time Stamp"])
    df = df.set_index("Time Stamp").sort_index()

    # Resample daily averages
    daily = df["Load"].resample("D").mean()

    # Rolling mean smoothing
    smooth = daily.rolling(window=window, center=True).mean()

    # ---- Stats ----
    print(f"Stats for {station}:")
    print(f"Mean load: {daily.mean():.2f} MW")
    print(f"Median load: {daily.median():.2f} MW")
    print(f"Std dev: {daily.std():.2f} MW")
    print(f"Min load: {daily.min():.2f} MW")
    print(f"Max load: {daily.max():.2f} MW")
    print("\nSeasonality (average load by month):")
    print(daily.groupby(daily.index.month).mean().round(2))

    # ---- Plot ----
    plt.figure(figsize=(14,6))
    plt.plot(daily.index, daily.values, alpha=0.4, label="Daily Avg Load")
    plt.plot(smooth.index, smooth.values, color="red", linewidth=2, label=f"{window}-day Rolling Mean")
    plt.title(f"Load over time for {station}")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.show()




analyze_station("nyiso_dataset", "GENESE", window=30)