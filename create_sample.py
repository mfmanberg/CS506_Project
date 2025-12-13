#!/usr/bin/env python
"""
Quick script to create a 1000-row sample of master.parquet
Run this if run_makefile.py is taking too long.
"""
import pandas as pd
from pathlib import Path

print("Creating 1000-row sample from master.parquet...")
print("This will take 30-60 seconds...")

master_path = Path("1_LIB/master/master.parquet")
sample_path = Path("1_LIB/master/master_sample.parquet")

if sample_path.exists():
    print(f"✓ {sample_path} already exists")
    print(f"   Delete it first if you want to recreate it")
else:
    # Load and sample
    df = pd.read_parquet(master_path)
    print(f"  Loaded {len(df):,} rows")
    
    # Create sample
    df_sample = df.sample(n=min(1000, len(df)), random_state=42)
    
    # Save
    df_sample.to_parquet(sample_path, index=False)
    print(f"✓ Created {sample_path} with {len(df_sample):,} rows")
    
    # Show file size
    size_mb = sample_path.stat().st_size / 1024 / 1024
    print(f"  File size: {size_mb:.2f} MB")

print("\nDone! Now run: python run_makefile.py run-analysis")
