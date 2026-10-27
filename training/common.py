"""Shared helpers: data loading, leakage-safe feature engineering, metrics."""
import numpy as np, pandas as pd

DATA = "/mnt/user-data/uploads/benue_weather_with_time_and_rainfall.csv"
OUT = "/mnt/user-data/outputs/rainfall_ml"
CACHE = "/home/claude/rainfall_ml/features.pkl"
TARGET = "rainfall_mm"
TRAIN_END, VAL_YEAR, TEST_YEAR = 2022, 2023, 2024
MET = ["temperature_c", "humidity_percent", "wind_speed_kmh", "dewpoint_depr"]


def load_raw():
    df = pd.read_csv(DATA, parse_dates=["datetime"])
    return df.sort_values(["lga", "datetime"]).reset_index(drop=True)


def _lag_roll(df, col, lags=(), rolls=(), prefix=None):
    """Past-only lags / rolling statistics per LGA (rows are strictly hourly)."""
    prefix = prefix or col
    g = df.groupby("lga")[col]
    out = {}
    for l in lags:
        out[f"{prefix}_lag{l}"] = g.shift(l)
    if rolls:
        cs = g.cumsum()
        gcs = cs.groupby(df["lga"])
        for w in rolls:
            out[f"{prefix}_roll{w}"] = (gcs.shift(1) - gcs.shift(1 + w))  # sum of t-w .. t-1
    return pd.DataFrame(out)


def build_features(df):
    df = df.copy()
    # --- physically-motivated variable: dew-point depression (Magnus formula)
    a, b = 17.62, 243.12
    gam = np.log(df.humidity_percent.clip(1, 100) / 100) + a * df.temperature_c / (b + df.temperature_c)
    df["dewpoint_c"] = b * gam / (a - gam)
    df["dewpoint_depr"] = df.temperature_c - df.dewpoint_c

    feats = {}
    # --- calendar (cyclic); `year` deliberately excluded (cannot extrapolate)
    doy = df.datetime.dt.dayofyear
    feats["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    base = df[["lat", "lon", "temperature_c", "humidity_percent", "wind_speed_kmh", "dewpoint_c", "dewpoint_depr"]]

    # --- regional (all-LGA) mean of concurrent weather
    reg = {}
    for c in ["temperature_c", "humidity_percent", "wind_speed_kmh"]:
        reg[f"reg_{c}"] = df.groupby("datetime")[c].transform("mean")

    # --- lagged / rolling meteorology (past only)
    parts = [base, pd.DataFrame(feats), pd.DataFrame(reg)]
    for c in MET:
        lr = _lag_roll(df, c, lags=(1, 3, 6, 24), rolls=(6, 24))
        for w in (6, 24):
            lr[f"{c}_roll{w}"] = lr[f"{c}_roll{w}"] / w   # -> rolling mean
        parts.append(lr)
    tmp = pd.concat(parts, axis=1)
    for c, l in [("temperature_c", 1), ("temperature_c", 3), ("humidity_percent", 1),
                 ("humidity_percent", 3), ("wind_speed_kmh", 1)]:
        tmp[f"{c}_d{l}"] = df[c] - tmp[f"{c}_lag{l}"]
    met_cols = list(tmp.columns)

    # --- rainfall memory (past only) -> scenario B
    rain = _lag_roll(df, TARGET, lags=(1, 2, 3, 6, 12, 24), rolls=(3, 6, 24, 72), prefix="rain")
    wet = (df[TARGET] > 0).astype(float)
    wet_prev = wet.groupby(df.lga).shift(1)
    idx = df.groupby("lga").cumcount().astype(float)
    last_wet = idx.where(wet_prev == 1).groupby(df.lga).ffill()
    rain["hours_since_rain"] = (idx - last_wet).fillna(720).clip(upper=720)
    regr = df.groupby("datetime")[TARGET].mean().sort_index()
    rain["reg_rain_lag1"] = df.datetime.map(regr.shift(1))
    rain["reg_rain_roll6"] = df.datetime.map(regr.shift(1).rolling(6).sum())
    rain_cols = list(rain.columns)

    X = pd.concat([tmp, rain], axis=1)
    meta = df[["datetime", "lga", "year", "month", "date", TARGET]].copy()
    # drop warm-up rows with incomplete lag windows (first 73 h per LGA)
    keep = (idx >= 73).values
    X, meta = X[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)
    return X.astype("float32"), meta, met_cols, rain_cols


def split_masks(meta):
    tr = (meta.year <= TRAIN_END).values
    va = (meta.year == VAL_YEAR).values
    te = (meta.year == TEST_YEAR).values
    return tr, va, te


def reg_metrics(y, p, meta=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    r2 = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    out = {"RMSE": rmse, "MAE": mae, "R2": r2}
    if meta is not None:  # daily totals per LGA (hydrologically more meaningful)
        d = pd.DataFrame({"k": meta["lga"].values + meta["date"].astype(str).values, "y": y, "p": p}).groupby("k").sum()
        out["Daily_RMSE"] = float(np.sqrt(np.mean((d.y - d.p) ** 2)))
        out["Daily_R2"] = float(1 - ((d.y - d.p) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum())
    return out
