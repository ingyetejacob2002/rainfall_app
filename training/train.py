"""Model training / evaluation for hourly rainfall (rainfall_mm) in Benue State."""
import sys, time, json, warnings, joblib
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR, HistGradientBoostingClassifier as HGBC, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve, f1_score
from sklearn.inspection import permutation_importance
from common import *
warnings.filterwarnings("ignore")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

X, meta, met_cols, rain_cols = joblib.load(CACHE)
tr, va, te = split_masks(meta)
y = meta[TARGET].values.astype("float64")
rng = np.random.RandomState(42)
SCEN = {"A_weather_only": met_cols, "B_with_rain_history": met_cols + rain_cols}
results, preds_store, cls_results, importances, tuning_log = [], {}, [], {}, []

def best_stage(model, Xv, yv):
    errs = [np.sqrt(np.mean((yv - p) ** 2)) for p in model.staged_predict(Xv)]
    return int(np.argmin(errs)) + 1, float(np.min(errs))

for scen, cols in SCEN.items():
    log("=== scenario", scen, len(cols), "features")
    Xtr, Xva, Xte = X.loc[tr, cols], X.loc[va, cols], X.loc[te, cols]
    ytr, yva, yte = y[tr], y[va], y[te]
    mte, mva = meta[te], meta[va]
    P = {}

    def record(name, p_va, p_te, secs):
        m = reg_metrics(yte, p_te, mte); mv = reg_metrics(yva, p_va)
        results.append({"scenario": scen, "model": name, **{f"test_{k}": v for k, v in m.items()},
                        "val_RMSE": mv["RMSE"], "val_R2": mv["R2"], "fit_seconds": round(secs, 1)})
        P[name] = p_te
        log(f"{name:28s} test RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f} dailyR2={m['Daily_R2']:.4f}")

    # ---------- baselines
    record("Baseline: train mean", np.full(va.sum(), ytr.mean()), np.full(te.sum(), ytr.mean()), 0)
    clim = pd.DataFrame({"k": meta.lga[tr].values + "_" + meta.month[tr].astype(str).values + "_" + meta.datetime.dt.hour[tr].astype(str).values, "y": ytr}).groupby("k").y.mean()
    key = lambda m_: (m_.lga.values + "_" + m_.month.astype(str).values + "_" + m_.datetime.dt.hour.astype(str).values)
    record("Baseline: climatology", pd.Series(key(mva)).map(clim).fillna(ytr.mean()).values,
           pd.Series(key(mte)).map(clim).fillna(ytr.mean()).values, 0)
    if scen.startswith("B"):
        record("Baseline: persistence", X.loc[va, "rain_lag1"].values, X.loc[te, "rain_lag1"].values, 0)

    # ---------- Ridge
    t = time.time()
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, ytr)
    record("Ridge regression", np.clip(ridge.predict(Xva), 0, None), np.clip(ridge.predict(Xte), 0, None), time.time() - t)

    # ---------- Random forest (subsample for 1-core budget)
    t = time.time()
    sub = rng.choice(len(Xtr), 120_000, replace=False)
    rf = RandomForestRegressor(n_estimators=60, max_depth=16, min_samples_leaf=40, max_features=0.4, n_jobs=1, random_state=42)
    rf.fit(Xtr.iloc[sub], ytr[sub])
    record("Random forest", rf.predict(Xva), rf.predict(Xte), time.time() - t)

    # ---------- Hyper-parameter search (random search, temporal validation = 2023)
    t = time.time()
    sub = rng.choice(len(Xtr), 150_000, replace=False)
    grid = {"learning_rate": [0.05, 0.1, 0.15], "max_leaf_nodes": [15, 31, 63, 127], "min_samples_leaf": [50, 200, 500],
            "l2_regularization": [0.0, 1.0, 10.0], "max_features": [0.5, 0.8, 1.0]}
    best = None
    for i, prm in enumerate(ParameterSampler(grid, 8, random_state=7)):
        m = HGBR(loss="squared_error", max_iter=250, early_stopping=False, random_state=0, **prm).fit(Xtr.iloc[sub], ytr[sub])
        n, e = best_stage(m, Xva, yva)
        tuning_log.append({"scenario": scen, **prm, "best_iter": n, "val_RMSE": e})
        log(f"  tune {i}: {prm} -> val RMSE {e:.4f} @ {n}")
        if best is None or e < best[0]: best = (e, prm)
    prm = best[1]; log("best params", prm, f"(search {time.time()-t:.0f}s)")

    # ---------- Gradient boosting: squared error & Poisson
    fitted = {}
    for name, loss in [("HistGB (squared error)", "squared_error"), ("HistGB (Poisson)", "poisson")]:
        t = time.time()
        m = HGBR(loss=loss, max_iter=500, early_stopping=False, random_state=0, **prm).fit(Xtr, ytr)
        n, e = best_stage(m, Xva, yva)
        m = HGBR(loss=loss, max_iter=n, early_stopping=False, random_state=0, **prm).fit(Xtr, ytr)
        fitted[name] = (m, n)
        record(name, m.predict(Xva), m.predict(Xte), time.time() - t)

    # ---------- Hurdle model: P(rain) x E[amount | rain]
    t = time.time()
    clf = HGBC(max_iter=400, early_stopping=False, random_state=0, **prm).fit(Xtr, (ytr > 0).astype(int))
    stg = [brier_score_loss((yva > 0).astype(int), p[:, 1]) for p in clf.staged_predict_proba(Xva)]
    nc = int(np.argmin(stg)) + 1
    clf = HGBC(max_iter=nc, early_stopping=False, random_state=0, **prm).fit(Xtr, (ytr > 0).astype(int))
    wr = ytr > 0
    amt = HGBR(loss="gamma", max_iter=300, early_stopping=False, random_state=0, **{**prm, "min_samples_leaf": min(prm["min_samples_leaf"], 100)}).fit(Xtr[wr], ytr[wr])
    na, _ = best_stage(amt, Xva[yva > 0], yva[yva > 0])
    amt = HGBR(loss="gamma", max_iter=na, early_stopping=False, random_state=0, **{**prm, "min_samples_leaf": min(prm["min_samples_leaf"], 100)}).fit(Xtr[wr], ytr[wr])
    pva, pte = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    record("Hurdle (clf x amount)", pva * amt.predict(Xva), pte * amt.predict(Xte), time.time() - t)

    # ---------- Rain / no-rain classification report (from hurdle classifier)
    pr, rc, th = precision_recall_curve((yva > 0).astype(int), pva)
    f1s = 2 * pr * rc / (pr + rc + 1e-12); thr = float(th[np.argmax(f1s[:-1])])
    yb = (yte > 0).astype(int); pb = (pte >= thr).astype(int)
    tp, fp = int(((pb == 1) & (yb == 1)).sum()), int(((pb == 1) & (yb == 0)).sum())
    fn, tn = int(((pb == 0) & (yb == 1)).sum()), int(((pb == 0) & (yb == 0)).sum())
    cls_results.append({"scenario": scen, "threshold(val F1-opt)": thr, "ROC_AUC": roc_auc_score(yb, pte),
                        "PR_AUC": average_precision_score(yb, pte), "Brier": brier_score_loss(yb, pte),
                        "Brier_climatology": brier_score_loss(yb, np.full_like(pte, (ytr > 0).mean())),
                        "Precision": tp / max(tp + fp, 1), "Recall": tp / max(tp + fn, 1),
                        "F1": f1_score(yb, pb), "Accuracy": (tp + tn) / len(yb), "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    log("classification:", {k: round(v, 4) if isinstance(v, float) else v for k, v in cls_results[-1].items()})
    P["_prob"] = pte

    # ---------- Champion: lowest validation RMSE among ML models -> refit on train+val
    ml = [r for r in results if r["scenario"] == scen and not r["model"].startswith("Baseline")]
    champ = min(ml, key=lambda r: r["val_RMSE"])["model"]
    log("champion by validation RMSE:", champ)
    trva = tr | va
    if champ.startswith("HistGB"):
        m0, n = fitted[champ]
        loss = "poisson" if "Poisson" in champ else "squared_error"
        final = HGBR(loss=loss, max_iter=int(n * 1.15), early_stopping=False, random_state=0, **prm).fit(X.loc[trva, cols], y[trva])
        pf = final.predict(Xte)
        log("champion refit on 2020-2023, test:", {k: round(v, 4) for k, v in reg_metrics(yte, pf, mte).items()})
        results.append({"scenario": scen, "model": f"FINAL {champ} (refit 2020-23)", **{f"test_{k}": v for k, v in reg_metrics(yte, pf, mte).items()}, "val_RMSE": np.nan, "val_R2": np.nan, "fit_seconds": np.nan})
        P["FINAL"] = pf
        joblib.dump({"model": final, "features": cols, "params": prm, "n_iter": int(n * 1.15), "loss": loss},
                    f"{OUT}/models/rainfall_{scen}.joblib", compress=3)
        imp_model = final
    elif champ.startswith("Hurdle"):
        ap = {**prm, "min_samples_leaf": min(prm["min_samples_leaf"], 100)}
        clf_f = HGBC(max_iter=int(nc * 1.15), early_stopping=False, random_state=0, **prm).fit(X.loc[trva, cols], (y[trva] > 0).astype(int))
        wr2 = trva & (y > 0)
        amt_f = HGBR(loss="gamma", max_iter=int(na * 1.15), early_stopping=False, random_state=0, **ap).fit(X.loc[wr2, cols], y[wr2])
        pf = clf_f.predict_proba(Xte)[:, 1] * amt_f.predict(Xte)
        mf = reg_metrics(yte, pf, mte)
        log("champion refit on 2020-2023, test:", {k: round(v, 4) for k, v in mf.items()})
        results.append({"scenario": scen, "model": f"FINAL {champ} (refit 2020-23)", **{f"test_{k}": v for k, v in mf.items()}, "val_RMSE": np.nan, "val_R2": np.nan, "fit_seconds": np.nan})
        P["FINAL"] = pf
        joblib.dump({"classifier": clf_f, "amount_model": amt_f, "features": cols, "params": prm,
                     "usage": "rain_mm_expected = classifier.predict_proba(X)[:,1] * amount_model.predict(X)"},
                    f"{OUT}/models/rainfall_{scen}.joblib", compress=3)
        class _H:
            def fit(self, *a, **k): return self
            def predict(self, Z): return clf_f.predict_proba(Z)[:, 1] * amt_f.predict(Z)
        imp_model = _H()
    else:
        imp_model = rf
    # ---------- Permutation importance on 40k test rows
    ix = rng.choice(len(Xte), 40_000, replace=False)
    pi = permutation_importance(imp_model, Xte.iloc[ix], yte[ix], n_repeats=3, random_state=0, scoring=lambda est, Z, t_: -np.sqrt(np.mean((t_ - est.predict(Z)) ** 2)))
    importances[scen] = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    log("top features:", importances[scen].head(8).round(4).to_dict())
    preds_store[scen] = {"P": P, "champ": champ, "meta": mte.reset_index(drop=True), "y": yte, "cols": cols}

# ---------- Persist
joblib.dump((preds_store, importances), '/home/claude/rainfall_ml/preds.pkl')
pd.DataFrame(results).to_csv(f"{OUT}/results/model_comparison.csv", index=False)
pd.DataFrame(cls_results).to_csv(f"{OUT}/results/rain_occurrence_classification.csv", index=False)
pd.DataFrame(tuning_log).to_csv(f"{OUT}/results/hyperparameter_search.csv", index=False)
pd.concat(importances, axis=1).to_csv(f"{OUT}/results/permutation_importance.csv")
joblib.dump((preds_store, importances), "/home/claude/rainfall_ml/preds.pkl")
log("DONE")
