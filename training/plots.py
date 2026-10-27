import numpy as np, pandas as pd, joblib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from common import *
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
F = f"{OUT}/figs"
store, imps = joblib.load("/home/claude/rainfall_ml/preds.pkl")
res = pd.read_csv(f"{OUT}/results/model_comparison.csv")

# 1 model comparison
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, sc in zip(axs, res.scenario.unique()):
    r = res[(res.scenario == sc) & ~res.model.str.startswith("FINAL")].set_index("model")
    cols = ["#adb5bd" if m.startswith("Baseline") else "#2a6f97" for m in r.index]
    ax.barh(r.index, r.test_RMSE, color=cols); ax.invert_yaxis(); ax.set_title(f"{sc}: test RMSE (2024, lower is better)")
    for i, (v, r2) in enumerate(zip(r.test_RMSE, r.test_R2)): ax.text(v, i, f" {v:.3f} (R²={r2:.2f})", va="center", fontsize=7)
    ax.set_xlim(0, r.test_RMSE.max() * 1.35)
plt.tight_layout(); plt.savefig(f"{F}/04_model_comparison.png"); plt.close()

# 2 daily totals scatter, 3 monthly profile
fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
for k, (sc, d) in enumerate(store.items()):
    p = d["P"]["FINAL"]; m = d["meta"]; y = d["y"]
    dd = pd.DataFrame({"k": m.lga + m.date.astype(str), "y": y, "p": p}).groupby("k").sum()
    ax = axs[0, k]; ax.scatter(dd.y, dd.p, s=3, alpha=.25, color="#2a6f97"); lim = max(dd.y.max(), dd.p.max())
    ax.plot([0, lim], [0, lim], "k--", lw=.8); ax.set_xlabel("observed daily total (mm)"); ax.set_ylabel("predicted"); ax.set_title(f"{sc}\n{d['champ']} – daily totals, test 2024")
    mm = pd.DataFrame({"month": m.month, "obs": y, "pred": p}).groupby("month").mean()
    ax = axs[1, k]; ax.plot(mm.index, mm.obs, "o-", label="observed", color="#bc4749"); ax.plot(mm.index, mm.pred, "s--", label="predicted", color="#2a6f97")
    ax.set_title("Monthly mean hourly rainfall, test 2024"); ax.set_xlabel("month"); ax.legend()
plt.tight_layout(); plt.savefig(f"{F}/05_daily_scatter_monthly_profile.png"); plt.close()

# 4 ROC / PR / calibration of rain-occurrence probabilities
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for sc, d, c in zip(store, store.values(), ["#2a6f97", "#bc4749"]):
    yb = (d["y"] > 0).astype(int); pr = d["P"]["_prob"]
    f, t, _ = roc_curve(yb, pr); axs[0].plot(f, t, color=c, label=f"{sc} (AUC {roc_auc_score(yb, pr):.3f})")
    a, b, _ = precision_recall_curve(yb, pr); axs[1].plot(b, a, color=c, label=f"{sc} (AP {average_precision_score(yb, pr):.3f})")
    bins = pd.qcut(pr, 15, duplicates="drop"); g = pd.DataFrame({"p": pr, "y": yb}).groupby(bins, observed=True).mean(); axs[2].plot(g.p, g.y, "o-", color=c, label=sc)
axs[0].plot([0, 1], [0, 1], "k--", lw=.7); axs[0].set_title("ROC – rain vs no rain"); axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR"); axs[0].legend(fontsize=7)
axs[1].set_title("Precision–recall"); axs[1].set_xlabel("recall"); axs[1].set_ylabel("precision"); axs[1].legend(fontsize=7)
axs[2].plot([0, 1], [0, 1], "k--", lw=.7); axs[2].set_title("Calibration of P(rain)"); axs[2].set_xlabel("mean predicted"); axs[2].set_ylabel("observed frequency"); axs[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{F}/06_classification_curves.png"); plt.close()

# 5 permutation importance
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (sc, s) in zip(axs, imps.items()):
    s = s.head(15)[::-1]; ax.barh(s.index, s.values, color="#6a994e"); ax.set_title(f"{sc}\npermutation importance (ΔRMSE when shuffled)"); ax.tick_params(axis="y", labelsize=8)
plt.tight_layout(); plt.savefig(f"{F}/07_feature_importance.png"); plt.close()

# 6 hourly time series, wettest 10 days at Makurdi (scenario B), 7 per-LGA RMSE
d = store["B_with_rain_history"]; m = d["meta"]; sel = (m.lga == "Makurdi").values
ts = pd.DataFrame({"t": m.datetime[sel], "obs": d["y"][sel], "pred": d["P"]["FINAL"][sel]}).set_index("t")
w = ts.obs.resample("D").sum().rolling(5).sum().idxmax(); seg = ts[w - pd.Timedelta(days=4): w + pd.Timedelta(days=1)]
fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
axs[0].bar(seg.index, seg.obs, width=.03, color="#bc4749", label="observed"); axs[0].plot(seg.index, seg.pred, color="#2a6f97", label="predicted"); axs[0].legend(); axs[0].set_title("Makurdi – wettest 5-day spell of 2024 (hourly, scenario B)")
r = pd.DataFrame({"lga": m.lga, "e": (d["y"] - d["P"]["FINAL"]) ** 2}).groupby("lga").e.mean().pow(.5).sort_values()
axs[1].barh(r.index, r.values, color="#2a6f97"); axs[1].tick_params(axis="y", labelsize=6); axs[1].set_title("Test RMSE by LGA (scenario B)")
plt.tight_layout(); plt.savefig(f"{F}/08_timeseries_and_lga_error.png"); plt.close()
print("plots done")
