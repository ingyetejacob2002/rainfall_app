"""Model loading, input validation and inference."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from features import build_inference_features, TARGET

REQUIRED = ["datetime", "lga", "temperature_c", "humidity_percent", "wind_speed_kmh"]
MAX_ROWS = 400_000
SCENARIOS = {"A_weather_only": "Weather only", "B_with_rain_history": "Weather + rainfall history"}


def risk_level(p, thr):
    return "Low" if p < 0.15 else "Moderate" if p < thr else "High" if p < 0.70 else "Very high"


class RainfallPredictor:
    def __init__(self, base_dir):
        base = Path(base_dir)
        self.models = {k: joblib.load(base / "models" / f"rainfall_{k}.joblib") for k in SCENARIOS}
        self.metrics = json.loads((base / "data" / "metrics.json").read_text())
        self.ref = json.loads((base / "data" / "reference.json").read_text())
        self.lga_coords = {r["lga"]: (r["lat"], r["lon"]) for r in self.ref["lga"]}
        self.lga_lookup = {k.lower(): k for k in self.lga_coords}
        self.lga_lookup["oturkpo"] = "Otukpo"   # alternate spelling used by some front-ends

    # ------------------------------------------------------------------ validation
    def normalise(self, df, need_rain):
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}. Required: {', '.join(REQUIRED)}.")
        if len(df) == 0:
            raise ValueError("The file contains no rows.")
        if len(df) > MAX_ROWS:
            raise ValueError(f"Too many rows ({len(df):,}); the limit is {MAX_ROWS:,}.")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if df["datetime"].isna().any():
            raise ValueError(f"{int(df['datetime'].isna().sum())} row(s) have an unreadable datetime (use e.g. 2024-08-01 13:00:00).")
        df["lga"] = df["lga"].astype(str).str.strip().str.lower().map(self.lga_lookup)
        if df["lga"].isna().any():
            raise ValueError("Unknown LGA name(s). Valid names: " + ", ".join(sorted(self.lga_coords)))
        for c, lo, hi in [("temperature_c", -10, 55), ("humidity_percent", 0, 100), ("wind_speed_kmh", 0, 150)]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].isna().any():
                raise ValueError(f"Column '{c}' has {int(df[c].isna().sum())} missing / non-numeric value(s).")
            if not df[c].between(lo, hi).all():
                raise ValueError(f"Column '{c}' has values outside the plausible range {lo}..{hi}.")
        if TARGET in df.columns:
            df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        else:
            df[TARGET] = np.nan
        if need_rain and df[TARGET].isna().any():
            raise ValueError("The rainfall-history model needs a complete 'rainfall_mm' column (no blanks). "
                             "Untick 'use rainfall history' to use the weather-only model.")
        if (df[TARGET].dropna() < 0).any():
            raise ValueError("'rainfall_mm' contains negative values.")
        df["lat"] = df["lga"].map(lambda k: self.lga_coords[k][0])
        df["lon"] = df["lga"].map(lambda k: self.lga_coords[k][1])
        if df.duplicated(["lga", "datetime"]).any():
            raise ValueError("Duplicate (lga, datetime) rows found.")
        df = df.sort_values(["lga", "datetime"]).reset_index(drop=True)
        step = df.groupby("lga")["datetime"].diff().dropna()
        if (step != pd.Timedelta(hours=1)).any():
            bad = df.loc[step[step != pd.Timedelta(hours=1)].index[0]]
            raise ValueError(f"Data must be strictly hourly with no gaps. First problem: {bad['lga']} at {bad['datetime']}.")
        return df

    # ------------------------------------------------------------------ inference
    def predict_frame(self, df, use_rain_history=False):
        scenario = "B_with_rain_history" if use_rain_history else "A_weather_only"
        m = self.models[scenario]
        df = self.normalise(df, use_rain_history)
        X, df, warm = build_inference_features(df)
        Xm = X[m["features"]]
        ok = warm & Xm.notna().all(axis=1).values
        n_skipped = int((~ok).sum())
        if ok.sum() == 0:
            raise ValueError("No rows can be predicted: each LGA needs at least 74 consecutive hourly records "
                             "(73 hours of history before the first predicted hour).")
        Xo = Xm[ok]
        p = m["classifier"].predict_proba(Xo)[:, 1]
        mu = m["amount_model"].predict(Xo)
        thr = float(m["threshold"])
        keep = [c for c in ["datetime", "lga", "temperature_c", "humidity_percent", "wind_speed_kmh"] if c in df]
        out = df.loc[ok, keep].copy()
        if df[TARGET].notna().any():
            out["observed_rainfall_mm"] = df.loc[ok, TARGET].values
        out["rain_probability"] = p.round(4)
        out["expected_rain_mm"] = (p * mu).round(4)
        out["rain_amount_if_wet_mm"] = mu.round(3)
        out["rain_predicted"] = (p >= thr).astype(int)
        out["risk_level"] = [risk_level(v, thr) for v in p]
        return out.reset_index(drop=True), {"scenario": scenario, "scenario_label": SCENARIOS[scenario],
                                            "threshold": thr, "rows_predicted": int(ok.sum()), "rows_skipped_warmup": n_skipped}

    def quick(self, lga, when, temp, hum, wind, temp_1h=None, hum_1h=None, rain=None):
        """Single-hour estimate from a synthetic 80 h history (steady-state assumption)."""
        when = pd.Timestamp(when).floor("h")
        times = pd.date_range(end=when, periods=80, freq="h")
        d = pd.DataFrame({"datetime": times, "lga": lga, "temperature_c": float(temp),
                          "humidity_percent": float(hum), "wind_speed_kmh": float(wind)})
        if temp_1h is not None: d.loc[len(d) - 2, "temperature_c"] = float(temp_1h)
        if hum_1h is not None: d.loc[len(d) - 2, "humidity_percent"] = float(hum_1h)
        if rain is not None:
            r1, r6, r24 = rain
            if not (0 <= r1 <= r6 <= r24):
                raise ValueError("Rainfall totals must satisfy: last hour <= last 6 h <= last 24 h.")
            k = np.arange(len(d))[::-1]            # hours before 'when'
            r = np.zeros(len(d))
            r[k == 1] = r1
            r[(k >= 2) & (k <= 6)] = (r6 - r1) / 5
            r[(k >= 7) & (k <= 24)] = (r24 - r6) / 18
            d[TARGET] = r
        out, info = self.predict_frame(d, use_rain_history=rain is not None)
        return out.iloc[-1].to_dict(), info
