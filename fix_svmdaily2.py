import json

with open('3_OUTPUT/3_svr/SVMDaily.ipynb', 'r') as f:
    nb = json.load(f)

# Update cell 4 - Daily aggregation - keep only Load column
nb['cells'][6]['source'] = [
    'df_daily = df_final[[\'Load\']].resample(\'D\').mean().reset_index()\n',
    'df_daily = df_daily.dropna()\n',
    '\n',
    'print(f\"Daily aggregated data shape: {df_daily.shape}\")\n',
    'print(df_daily.head())\n',
    '\n',
    '# Data is ready to use\n',
    'merged = df_daily.copy()\n',
    'df_total_load = merged\n',
    '\n'
]

with open('3_OUTPUT/3_svr/SVMDaily.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('✓ Updated cell 6 (daily aggregation) to keep only Load column')
