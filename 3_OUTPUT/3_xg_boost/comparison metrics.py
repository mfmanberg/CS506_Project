import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# LOAD JSON RESULTS


with open("results_old.json", "r") as f:
    old = json.load(f)

with open("results_new.json", "r") as f:
    new = json.load(f)

# Convert to DataFrame
df_old = pd.DataFrame(old).T
df_new = pd.DataFrame(new).T

# Remove raw cuz mesonet only starts from five minute aggregation
keep = ["five", "quarter", "hourly", "daily"]

df_old = df_old.loc[keep]
df_new = df_new.loc[keep]


# COMPARISON PLOT


metrics = ["MAPE", "MAE", "RMSE", "R2"]
titles = {
    "MAPE": "MAPE (%)",
    "MAE":  "MAE",
    "RMSE": "RMSE",
    "R2":   "R²"
}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

x = np.arange(len(keep))  # positions
bar_width = 0.35

for i, metric in enumerate(metrics):
    ax = axes[i]

    old_vals = df_old[metric].values
    new_vals = df_new[metric].values

    ax.bar(x - bar_width/2, old_vals, width=bar_width, label="NYISO Only Model", alpha=0.7)
    ax.bar(x + bar_width/2, new_vals, width=bar_width, label="NYISO + Mesonet Model", alpha=0.7)

    ax.set_title(titles[metric], fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(keep, fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # improve spacing for tight metrics
    if metric == "R2":
        ax.set_ylim(0, 1.05)

    ax.legend()

plt.tight_layout()
plt.show()

