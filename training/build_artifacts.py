"""Train/finalise deployable models + export reference data & metrics for the web app."""
import json, joblib, time, shutil, numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR, HistGradientBoostingClassifier as HGBC
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.inspection import permutation_importance
from common import *
warnings.filterwarnings("ignore")
APP = "/home/claude/rainfall_app"; T0 = time.time()
def log(*a): print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)

# ---------------- reference data from raw file
raw = load_raw()
ref = {"rows": int(len(raw)), "n_lga": int(raw.lga.nunique()), "start": str(raw.datetime.min().date()), "end": str(raw.datetime.max().date()),
       "zero_share": float((raw.rainfall_mm == 0).mean()), "max_hourly_mm": float(raw.rainfall_mm.max()),
       "lga": raw.groupby("lga")[["lat", "lon"]].first().round(4).reset_index().to_dict("records")}
mon = raw.groupby(["lga", "year", "month"]).rainfall_mm.sum().groupby(["lga", "month"]).mean().round(1).unstack()
ref["monthly_mm"] = {k: v.tolist() for k, v in mon.iterrows()}
ref["monthly_mm_region"] = mon.mean().round(1).tolist()
ref["hourly_profile"] = raw.groupby("hour").rainfall_mm.mean().round(4).tolist()
ref["annual_mm_by_lga"] = (raw.groupby("lga").rainfall_mm.sum() / 5).round(0).sort_values(ascending=False).to_dict()
ref["yearly_mm_region"] = (raw.groupby("year").rainfall_mm.sum() / raw.lga.nunique()).round(0).to_dict()
rd = raw.assign(wet=raw.rainfall_mm > 0).groupby("month").wet.mean().round(4)
ref["wet_hour_share_by_month"] = rd.tolist()
json.dump(ref, open(f"{APP}/data/reference.json", "w"))
# demo upload: all LGAs, 2024-07-29 .. 2024-08-07
s = raw[(raw.datetime >= "2024-07-29") & (raw.datetime < "2024-08-08")][["datetime", "lga", "temperature_c", "humidity_percent", "wind_speed_kmh", "rainfall_mm"]]
s.sort_values(["datetime", "lga"]).to_csv(f"{APP}/data/sample_upload.csv", index=False)
log("reference + sample written", len(s), "rows")
del raw

# ---------------- models
X, meta, mc, rc = joblib.load(CACHE)
tr, va, te = split_masks(meta); y = meta[TARGET].values.astype("float64"); trva = tr | va
mte = meta[te]

def evaluate(clf, amt, cols, thr, name):
    Xte = X.loc[te, cols]; yte = y[te]
    p = clf.predict_proba(Xte)[:, 1]; mu = amt.predict(Xte); pred = p * mu
    out = {"regression": reg_metrics(yte, pred, mte)}
    yb = (yte > 0).astype(int); pb = (p >= thr).astype(int)
    tp = int(((pb == 1) & (yb == 1)).sum()); fp = int(((pb == 1) & (yb == 0)).sum()); fn = int(((pb == 0) & (yb == 1)).sum()); tn = int(((pb == 0) & (yb == 0)).sum())
    out["classification"] = {"threshold": thr, "roc_auc": float(roc_auc_score(yb, p)), "pr_auc": float(average_precision_score(yb, p)),
        "brier": float(brier_score_loss(yb, p)), "brier_climatology": float(brier_score_loss(yb, np.full_like(p, (y[tr] > 0).mean()))),
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "accuracy": (tp + tn) / len(yb),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    b = pd.qcut(p, 10, duplicates="drop"); g = pd.DataFrame({"p": p, "y": yb}).groupby(b, observed=True).mean()
    out["calibration"] = {"pred": g.p.round(4).tolist(), "obs": g.y.round(4).tolist()}
    mm = pd.DataFrame({"m": mte.month.values, "obs": yte, "pred": pred}).groupby("m").mean()
    out["monthly"] = {"obs": mm.obs.round(4).tolist(), "pred": mm.pred.round(4).tolist()}
    class _H:
        def fit(self, *a, **k): return self
        def predict(self, Z): return clf.predict_proba(Z)[:, 1] * amt.predict(Z)
    ix = np.random.RandomState(0).choice(len(Xte), 20000, replace=False)
    pi = permutation_importance(_H(), Xte.iloc[ix], yte[ix], n_repeats=2, random_state=0, scoring=lambda e, Z, t: -np.sqrt(np.mean((t - e.predict(Z)) ** 2)))
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False).head(15)
    out["importance"] = {k: round(float(v), 5) for k, v in imp.items()}
    log(name, {k: round(v, 4) for k, v in out["regression"].items()}, "AUC", round(out["classification"]["roc_auc"], 4))
    return out

metrics = {}
# Scenario A: already trained (hurdle refit on 2020-23); threshold selected on 2023 validation in the notebook run
A = joblib.load("/mnt/user-data/outputs/rainfall_ml/models/rainfall_A_weather_only.joblib")
A["threshold"] = 0.3713; joblib.dump(A, f"{APP}/models/rainfall_A_weather_only.joblib", compress=3)
metrics["A_weather_only"] = evaluate(A["classifier"], A["amount_model"], A["features"], 0.3713, "A")

# Scenario B: train (train only) -> choose iterations/threshold on 2023 -> refit on 2020-23
cols = mc + rc; prm = {"min_samples_leaf": 50, "max_leaf_nodes": 15, "max_features": 0.5, "learning_rate": 0.15, "l2_regularization": 10.0}
Xtr, Xva = X.loc[tr, cols], X.loc[va, cols]; ytr, yva = y[tr], y[va]
clf = HGBC(max_iter=400, early_stopping=False, random_state=0, **prm).fit(Xtr, (ytr > 0).astype(int))
best = (9, None, None)
for i, p in enumerate(clf.staged_predict_proba(Xva)):
    b = brier_score_loss((yva > 0).astype(int), p[:, 1])
    if b < best[0]: best = (b, i + 1, p[:, 1].copy())
nc, pva = best[1], best[2]
pr, rc_, th = precision_recall_curve((yva > 0).astype(int), pva); f1 = 2 * pr * rc_ / (pr + rc_ + 1e-12); thr = float(th[np.argmax(f1[:-1])])
log("B classifier iters", nc, "threshold", round(thr, 4))
ap = {**prm, "min_samples_leaf": 50}
wr = ytr > 0; amt = HGBR(loss="gamma", max_iter=300, early_stopping=False, random_state=0, **ap).fit(Xtr[wr], ytr[wr])
Xvw, yvw = Xva[yva > 0], yva[yva > 0]
errs = [np.sqrt(np.mean((yvw - p) ** 2)) for p in amt.staged_predict(Xvw)]; na = int(np.argmin(errs)) + 1
log("B amount iters", na)
clf_f = HGBC(max_iter=int(nc * 1.15), early_stopping=False, random_state=0, **prm).fit(X.loc[trva, cols], (y[trva] > 0).astype(int))
w2 = trva & (y > 0)
amt_f = HGBR(loss="gamma", max_iter=int(na * 1.15), early_stopping=False, random_state=0, **ap).fit(X.loc[w2, cols], y[w2])
B = {"classifier": clf_f, "amount_model": amt_f, "features": cols, "params": prm, "threshold": thr}
joblib.dump(B, f"{APP}/models/rainfall_B_with_rain_history.joblib", compress=3)
metrics["B_with_rain_history"] = evaluate(clf_f, amt_f, cols, thr, "B")

metrics["benchmark"] = {  # held-out 2024 test, models trained on 2020-22 (modelling notebook run)
 "A_weather_only": [["Baseline: mean",0.5649,0.1797,-0.0018],["Baseline: climatology",0.5591,0.1601,0.0183],["Ridge regression",0.5202,0.1493,0.1503],["Random forest",0.4714,0.1028,0.3024],["HistGB (squared error)",0.4797,0.1088,0.2773],["HistGB (Poisson)",0.5317,0.1086,0.1124],["Hurdle (classifier x amount)",0.4661,0.0898,0.3179]],
 "B_with_rain_history": [["Baseline: mean",0.5649,0.1797,-0.0018],["Baseline: climatology",0.5591,0.1601,0.0183],["Baseline: persistence",0.5644,0.1003,-0.0001],["Ridge regression",0.4463,0.1190,0.3747],["Random forest",0.4217,0.0809,0.4415],["HistGB (squared error)",0.3891,0.0824,0.5246],["HistGB (Poisson)",0.4542,0.0797,0.3522],["Hurdle (classifier x amount)",0.3945,0.0732,0.5114]]}
metrics["split"] = {"train": "2020-2022", "validation": "2023", "test": "2024", "deployed_fit": "2020-2023"}
json.dump(metrics, open(f"{APP}/data/metrics.json", "w"), indent=1)
log("DONE")
