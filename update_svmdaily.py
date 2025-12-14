#!/usr/bin/env python3
"""Update SVMDaily.ipynb to include weather columns"""
import json
import sys

def update_notebook():
    notebook_path = '3_OUTPUT/3_svr/SVMDaily.ipynb'
    
    # Read notebook
    print(f"Reading {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Update 1: Change MASTER_TOTAL_PARQUET to MASTER_PARQUET
    updated_parquet = False
    for i, cell in enumerate(nb['cells']):
        if 'source' in cell and any('MASTER_TOTAL_PARQUET' in line for line in cell['source']):
            print(f'Found cell at index {i} with MASTER_TOTAL_PARQUET')
            for j, line in enumerate(cell['source']):
                if 'MASTER_TOTAL_PARQUET' in line:
                    cell['source'][j] = line.replace('MASTER_TOTAL_PARQUET', 'MASTER_PARQUET')
                    updated_parquet = True
            print('Changed MASTER_TOTAL_PARQUET to MASTER_PARQUET')
            break
    
    if not updated_parquet:
        print('WARNING: Could not find MASTER_TOTAL_PARQUET to update')
    
    # Update 2: Add weather columns to daily resampling
    updated_daily = False
    for i, cell in enumerate(nb['cells']):
        if 'source' in cell and any('df_daily = df_final.resample' in line for line in cell['source']):
            print(f'Found cell at index {i} with daily resampling')
            # Replace the source
            new_source = [
                '# Select Load and weather columns for daily aggregation\n',
                'weather_cols = [\n',
                "    'temp_2m [degF]',\n",
                "    'apparent_temperature [degF]',\n",
                "    'relative_humidity [percent]',\n",
                "    'precip_1hr [inch]',\n",
                "    'avg_wind_speed_merge [mile/hr]',\n",
                "    'solar_insolation [W/m^2]'\n",
                ']\n',
                "df_daily = df_final[['Load'] + weather_cols].resample('D').mean().reset_index()\n",
                'df_daily = df_daily.dropna()\n',
                '\n',
                'print(f"Daily aggregated data shape: {df_daily.shape}")\n',
                'print(df_daily.head())\n',
                '\n',
                '# Data is ready to use\n',
                'merged = df_daily.copy()\n',
                'df_total_load = merged\n'
            ]
            cell['source'] = new_source
            updated_daily = True
            print('Cell updated successfully')
            break
    
    if not updated_daily:
        print('ERROR: Could not find the daily resampling cell to update')
        sys.exit(1)
    
    # Write back
    print(f"Writing updated notebook to {notebook_path}...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print('✓ Updated successfully')
    print('✓ Changes made:')
    print('  1. Changed from MASTER_TOTAL_PARQUET to MASTER_PARQUET')
    print('  2. Added weather columns to daily aggregation:')
    print('     - temp_2m [degF]')
    print('     - apparent_temperature [degF]')
    print('     - relative_humidity [percent]')
    print('     - precip_1hr [inch]')
    print('     - avg_wind_speed_merge [mile/hr]')
    print('     - solar_insolation [W/m^2]')

if __name__ == '__main__':
    update_notebook()
