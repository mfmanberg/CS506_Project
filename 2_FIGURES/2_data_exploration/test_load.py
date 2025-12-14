"""Quick test to verify data loading works"""
import pandas as pd
from pathlib import Path

# Find project root
project_root = Path(__file__).parent
while not (project_root / "1_LIB").exists() and project_root.parent != project_root:
    project_root = project_root.parent

MASTER_PATH = project_root / "1_LIB" / "master" / "master.parquet"
print(f"Loading from: {MASTER_PATH}")
print(f"File exists: {MASTER_PATH.exists()}")

# Load data
df = pd.read_parquet(MASTER_PATH)
print(f"Loaded {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Filter to 2023
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df_2023 = df[df['datetime'].dt.year == 2023]
    print(f"2023 data: {len(df_2023):,} rows")
elif 'Time Stamp' in df.columns:
    df['datetime'] = pd.to_datetime(df['Time Stamp'], utc=True)
    df_2023 = df[df['datetime'].dt.year == 2023]
    print(f"2023 data: {len(df_2023):,} rows")

print("Success!")
