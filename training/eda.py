import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
df = load_raw(); F = f"{OUT}/figs"
y = df[TARGET]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].hist(y[y > 0], bins=60, color="#2a6f97"); ax[0].set_yscale("log"); ax[0].set_title(f"Non-zero hourly rainfall (zeros = {100*(y==0).mean():.1f}% of rows)"); ax[0].set_xlabel("mm/h")
m = df.groupby("month")[TARGET].mean(); ax[1].bar(m.index, m.values, color="#2a6f97"); ax[1].set_title("Mean hourly rainfall by month"); ax[1].set_xlabel("month")
h = df.groupby("hour")[TARGET].mean(); ax[2].plot(h.index, h.values, marker="o", color="#bc4749"); ax[2].set_title("Diurnal cycle (mean mm/h by hour)"); ax[2].set_xlabel("hour")
plt.tight_layout(); plt.savefig(f"{F}/01_target_distribution_seasonality.png"); plt.close()

num = ["rainfall_mm", "temperature_c", "humidity_percent", "wind_speed_kmh", "daily_rainfall_sum_mm", "daily_temp_max_c", "daily_temp_min_c"]
c = df[num].corr(method="spearman")
fig, ax = plt.subplots(figsize=(6.2, 5)); im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=45, ha="right"); ax.set_yticks(range(len(num))); ax.set_yticklabels(num)
for i in range(len(num)):
    for j in range(len(num)): ax.text(j, i, f"{c.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, shrink=.8); ax.set_title("Spearman correlation (raw columns)"); plt.tight_layout(); plt.savefig(f"{F}/02_correlation.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
a = df.groupby(["year"])[TARGET].sum() / df.lga.nunique(); ax[0].bar(a.index, a.values, color="#2a6f97"); ax[0].set_title("Mean annual rainfall per LGA (mm)")
l = df.groupby("lga")[TARGET].sum().div(5).sort_values(); ax[1].barh(l.index, l.values, color="#6a994e"); ax[1].set_title("Mean annual rainfall by LGA (mm)"); ax[1].tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.savefig(f"{F}/03_annual_and_lga.png"); plt.close()
# audit
pd.Series({"rows": len(df), "lgas": df.lga.nunique(), "period": f"{df.datetime.min()} to {df.datetime.max()}",
           "missing_values": int(df.isna().sum().sum()), "duplicate_lga_hours": int(df.duplicated(['lga','datetime']).sum()),
           "rain_mm identical to rainfall_mm": bool((df.rainfall_mm == df.rain_mm).all()),
           "share_zero_hours_%": round(100*(y==0).mean(),1)}).to_csv(f"{OUT}/results/data_audit.csv")
print("eda done")
