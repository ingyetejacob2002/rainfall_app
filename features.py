"""Feature engineering used at inference time.

This is the same transformation used when the models were trained (see the modelling
notebook), except that rows are *kept* (warm-up rows are flagged instead of dropped).
All lag / rolling features use past observations only.
"""
import numpy as np
import pandas as pd

TARGET = "rainfall_mm"
MET = ["temperature_c", "humidity_percent", "wind_speed_kmh", "dewpoint_depr"]
WARMUP_HOURS = 73  # first 73 hours of every LGA lack a complete 72 h window


def _lag_roll(df, col, lags=(), rolls=(), prefix=None):
    prefix = prefix or col
    g = df.groupby("lga")[col]
    out = {f"{prefix}_lag{l}": g.shift(l) for l in lags}
    if rolls:
        gcs = g.cumsum().groupby(df["lga"])
        for w in rolls:
            out[f"{prefix}_roll{w}"] = gcs.shift(1) - gcs.shift(1 + w)
    return pd.DataFrame(out)


def build_inference_features(df):
    """df needs: datetime, lga, lat, lon, temperature_c, humidity_percent, wind_speed_kmh, rainfall_mm
    (rainfall_mm may be NaN when only the weather-only model is used). Rows must be strictly hourly per LGA."""
    df = df.sort_values(["lga", "datetime"]).reset_index(drop=True)
    a, b = 17.62, 243.12
    gam = np.log(df.humidity_percent.clip(1, 100) / 100) + a * df.temperature_c / (b + df.temperature_c)
    df["dewpoint_c"] = b * gam / (a - gam)
    df["dewpoint_depr"] = df.temperature_c - df.dewpoint_c

    doy = df.datetime.dt.dayofyear
    hour = df.datetime.dt.hour
    cal = pd.DataFrame({"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25)})
    base = df[["lat", "lon", "temperature_c", "humidity_percent", "wind_speed_kmh", "dewpoint_c", "dewpoint_depr"]]
    reg = pd.DataFrame({f"reg_{c}": df.groupby("datetime")[c].transform("mean")
                        for c in ["temperature_c", "humidity_percent", "wind_speed_kmh"]})
    parts = [base, cal, reg]
    for c in MET:
        lr = _lag_roll(df, c, lags=(1, 3, 6, 24), rolls=(6, 24))
        for w in (6, 24):
            lr[f"{c}_roll{w}"] = lr[f"{c}_roll{w}"] / w
        parts.append(lr)
    tmp = pd.concat(parts, axis=1)
    for c, l in [("temperature_c", 1), ("temperature_c", 3), ("humidity_percent", 1), ("humidity_percent", 3), ("wind_speed_kmh", 1)]:
        tmp[f"{c}_d{l}"] = df[c] - tmp[f"{c}_lag{l}"]

    rain = _lag_roll(df, TARGET, lags=(1, 2, 3, 6, 12, 24), rolls=(3, 6, 24, 72), prefix="rain")
    wet_prev = (df[TARGET] > 0).astype(float).groupby(df.lga).shift(1)
    idx = df.groupby("lga").cumcount().astype(float)
    last_wet = idx.where(wet_prev == 1).groupby(df.lga).ffill()
    rain["hours_since_rain"] = (idx - last_wet).fillna(720).clip(upper=720)
    regr = df.groupby("datetime")[TARGET].mean().sort_index()
    rain["reg_rain_lag1"] = df.datetime.map(regr.shift(1))
    rain["reg_rain_roll6"] = df.datetime.map(regr.shift(1).rolling(6).sum())

    X = pd.concat([tmp, rain], axis=1).astype("float32")
    return X, df, (idx >= WARMUP_HOURS).values
