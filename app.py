"""Benue Weather – Flask application (serves the web interface and the prediction API)."""
import io, json, logging, os, re, time, uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, abort
from sklearn.metrics import mean_absolute_error, roc_auc_score

from predictor import RainfallPredictor

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("benue-weather")

predictor = RainfallPredictor(BASE)
RESULTS, TTL_SECONDS, MAX_STORED = {}, 3600, 25
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").strip()   # e.g. "*" or "https://my-site.example"


def _store(df, info):
    now = time.time()
    for k in [k for k, v in RESULTS.items() if now - v["ts"] > TTL_SECONDS]:
        RESULTS.pop(k, None)
    while len(RESULTS) >= MAX_STORED:
        RESULTS.pop(min(RESULTS, key=lambda k: RESULTS[k]["ts"]))
    rid = uuid.uuid4().hex
    RESULTS[rid] = {"df": df, "info": info, "ts": now}
    return rid


def _result(rid):
    r = RESULTS.get(rid) if re.fullmatch(r"[0-9a-f]{32}", rid or "") else None
    if r is None:
        abort(404, "Batch result not found or expired – please run the batch again.")
    return r


# ------------------------------------------------------------------ interface
@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------------ API
@app.route("/api/health")
def health():
    return jsonify(status="ok", message="Models loaded", models=list(predictor.models), lgas=len(predictor.lga_coords))


@app.route("/api/lgas")
def lgas():
    return jsonify([r["lga"] for r in sorted(predictor.ref["lga"], key=lambda r: r["lga"])])


@app.route("/api/climate")
def climate():
    return jsonify(predictor.ref)


@app.route("/api/metrics")
def metrics():
    m = predictor.metrics
    b, a = m["B_with_rain_history"], m["A_weather_only"]
    brief = lambda s: {"mae": s["regression"]["MAE"], "rmse": s["regression"]["RMSE"], "r2": s["regression"]["R2"],
                       "roc_auc": s["classification"]["roc_auc"], "f1": s["classification"]["f1"]}
    return jsonify(model="Hurdle gradient boosting (weather + rainfall history)", evaluation="Held-out test year 2024, hourly",
                   mae=b["regression"]["MAE"], rmse=b["regression"]["RMSE"], r2=b["regression"]["R2"],
                   roc_auc=b["classification"]["roc_auc"], f1=b["classification"]["f1"],
                   scenarios={"weather_only": brief(a), "with_rain_history": brief(b)})


@app.route("/api/reference")
def reference():
    m, ref = predictor.metrics, predictor.ref
    def block(s):
        r, c = s["regression"], s["classification"]
        return {"R²": round(r["R2"], 3), "RMSE (mm/h)": round(r["RMSE"], 3), "MAE (mm/h)": round(r["MAE"], 3),
                "Rain detection AUC": round(c["roc_auc"], 3), "F1 (rain)": round(c["f1"], 3), "Decision threshold": round(c["threshold"], 3)}
    top = lambda s: [k.replace("_", " ") for k in list(s["importance"])[:5]]
    return jsonify({
        "Model family": "Hurdle gradient boosting: P(rain) × expected amount",
        "Target": "Hourly rainfall (mm)",
        "Training data": f"{ref['rows']:,} hourly records · {ref['n_lga']} LGAs · {ref['start']} to {ref['end']}",
        "Validation": "Train 2020–2022 · tune 2023 · test 2024 (deployed models refit on 2020–2023)",
        "Weather-only model": block(m["A_weather_only"]),
        "Rainfall-history model": block(m["B_with_rain_history"]),
        "Top predictors (weather only)": top(m["A_weather_only"]),
        "Top predictors (with rain history)": top(m["B_with_rain_history"]),
        "Risk levels": "Low < 15% · Moderate up to threshold · High to 70% · Very high above",
        "Batch columns": "datetime, lga, temperature_c, humidity_percent, wind_speed_kmh (+ optional rainfall_mm)",
    })


def _pick(d, *keys):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def _num(d, label, *keys, required=True):
    v = _pick(d, *keys)
    if v is None:
        if required:
            raise ValueError(f"'{label}' is required.")
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"'{label}' must be a number.")


@app.post("/api/predict")
def api_predict():
    d = request.get_json(silent=True)
    if not isinstance(d, dict):
        raise ValueError("Send a JSON object.")
    if _pick(d, "datetime"):
        when = pd.to_datetime(_pick(d, "datetime"), errors="coerce")
    else:
        date, hour = _pick(d, "date"), _pick(d, "hour")
        if date is None or hour is None:
            raise ValueError("Provide 'date' and 'hour' (or 'datetime').")
        when = pd.to_datetime(f"{date} {int(float(hour)):02d}:00", errors="coerce")
    if pd.isna(when):
        raise ValueError("Unreadable date/hour.")

    mode = str(d.get("use_rain", "auto")).lower()
    if mode not in ("auto", "0", "1"):
        raise ValueError("'use_rain' must be auto, 0 or 1.")
    rain = [_num(d, k, k, required=False) for k in ("rain_last_1h", "rain_last_6h", "rain_last_24h")]
    if mode == "1" and any(v is None for v in rain):
        raise ValueError("Enter rainfall for the last 1 h, 6 h and 24 h to include precipitation.")
    if any(v is None for v in rain) and any(v is not None for v in rain):
        raise ValueError("Provide all three rainfall totals (last 1 h, 6 h, 24 h) or none.")
    use_rain = mode == "1" or (mode == "auto" and all(v is not None for v in rain))

    res, info = predictor.quick(
        d.get("lga", ""), when,
        _num(d, "temperature", "temperature", "temperature_c"), _num(d, "humidity", "humidity", "humidity_percent"),
        _num(d, "wind speed", "wind_speed", "wind_speed_kmh"),
        _num(d, "temperature 1 hour ago", "temperature_1hr_ago", "temperature_1h_ago", required=False),
        _num(d, "humidity 1 hour ago", "humidity_1hr_ago", "humidity_1h_ago", required=False),
        rain if use_rain else None)
    p, thr = float(res["rain_probability"]), info["threshold"]
    risk = res["risk_level"]
    return jsonify(prediction=res["expected_rain_mm"], unit="mm/h",
                   prediction_label=f"{'Rain likely' if p >= thr else 'Rain unlikely'} · {risk} risk ({p:.0%} chance)",
                   probability=p, confidence=max(p, 1 - p), expected_mm=res["expected_rain_mm"],
                   amount_if_wet_mm=res["rain_amount_if_wet_mm"], rain_predicted=bool(res["rain_predicted"]), risk=risk,
                   model=info["scenario_label"], scenario=info["scenario"], threshold=thr)


@app.post("/api/batch")
def api_batch():
    file = request.files.get("file")
    if not file or not file.filename:
        raise ValueError("Choose a CSV file to upload.")
    name = file.filename.lower()
    if name.endswith((".xlsx", ".xlsm")):
        raw = pd.read_excel(file)
    elif name.endswith((".csv", ".txt")):
        raw = pd.read_csv(file)
    else:
        raise ValueError("Unsupported file type. Upload a .csv file.")
    cols = [str(c).strip().lower() for c in raw.columns]
    mode = request.form.get("use_rain", "auto")
    has_rain = "rainfall_mm" in cols and raw.iloc[:, cols.index("rainfall_mm")].notna().all()
    use_rain = has_rain if mode == "auto" else mode == "1"
    out, info = predictor.predict_frame(raw, use_rain_history=use_rain)
    rid = _store(out, info)

    s = {"rows_predicted": info["rows_predicted"], "rows_skipped_warmup": info["rows_skipped_warmup"], "model": info["scenario_label"],
         "n_lgas": int(out.lga.nunique()), "start": str(out.datetime.min()), "end": str(out.datetime.max()),
         "share_predicted_wet": float(out.rain_predicted.mean()), "mean_probability": float(out.rain_probability.mean())}
    msg = (f"Predicted {s['rows_predicted']:,} hours for {s['n_lgas']} LGA(s) with the {info['scenario_label'].lower()} model "
           f"({s['rows_skipped_warmup']:,} warm-up hours used as history only).")
    if "observed_rainfall_mm" in out:
        y, p = out.observed_rainfall_mm.values, out.expected_rain_mm.values
        yb = (y > 0).astype(int)
        s["evaluation"] = {"rmse": float(np.sqrt(np.mean((y - p) ** 2))), "mae": float(mean_absolute_error(y, p)),
                           "r2": float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)),
                           "roc_auc": float(roc_auc_score(yb, out.rain_probability)) if 0 < yb.sum() < len(yb) else None}
        e = s["evaluation"]
        msg += f" Against observed rainfall: R² {e['r2']:.2f}, RMSE {e['rmse']:.3f}, MAE {e['mae']:.3f}" + (f", AUC {e['roc_auc']:.3f}." if e["roc_auc"] else ".")
    return jsonify(id=rid, message=msg, summary=s, lgas=sorted(out.lga.unique()))


@app.get("/api/batch/<rid>/series")
def api_series(rid):
    df = _result(rid)["df"]
    lga = request.args.get("lga") or df.lga.iloc[0]
    d = df[df.lga.str.lower() == lga.lower()]
    if d.empty:
        abort(404, f"No rows for '{lga}' in this batch.")
    has_obs = "observed_rainfall_mm" in d
    daily = len(d) > 1500
    if daily:
        g = d.assign(day=d.datetime.dt.strftime("%Y-%m-%d")).groupby("day")
        t = pd.DataFrame({"time": g.size().index, "prediction": g.expected_rain_mm.sum().values, "probability": g.rain_probability.max().values})
        if has_obs: t["actual"] = g.observed_rainfall_mm.sum().values
    else:
        t = pd.DataFrame({"time": d.datetime.dt.strftime("%Y-%m-%d %H:%M").values, "prediction": d.expected_rain_mm.values, "probability": d.rain_probability.values})
        if has_obs: t["actual"] = d.observed_rainfall_mm.values
    t = t.round(4)
    metrics_ = {}
    if has_obs:
        y, p = d.observed_rainfall_mm.values, d.expected_rain_mm.values
        metrics_ = {"rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 4), "mae": round(float(np.mean(np.abs(y - p))), 4)}
    return jsonify(lga=d.lga.iloc[0], granularity="daily totals" if daily else "hourly", series=t.to_dict("records"), metrics=metrics_)


@app.get("/download/<rid>.csv")
def download(rid):
    r = _result(rid)
    buf = io.BytesIO(r["df"].to_csv(index=False).encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="rainfall_predictions.csv")


@app.get("/sample.csv")
def sample():
    return send_file(BASE / "data" / "sample_upload.csv", mimetype="text/csv", as_attachment=True, download_name="sample_upload.csv")


# ------------------------------------------------------------------ cross-origin (optional), errors
@app.after_request
def cors(resp):
    if CORS_ORIGINS and request.path.startswith(("/api", "/download")):
        resp.headers["Access-Control-Allow-Origin"] = CORS_ORIGINS
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.errorhandler(ValueError)
def bad_input(e):
    return jsonify(error=str(e)), 400

@app.errorhandler(413)
def too_big(e):
    return jsonify(error="File too large (limit 30 MB)."), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify(error=getattr(e, "description", "Not found")), 404

@app.errorhandler(Exception)
def server_error(e):
    if hasattr(e, "code") and e.code and e.code < 500:
        return jsonify(error=getattr(e, "description", str(e))), e.code
    log.exception("Unhandled error")
    return jsonify(error="Unexpected server error. Check the input and try again."), 500


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
