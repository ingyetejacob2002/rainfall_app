# Benue Weather – rainfall prediction app

A web application for predicting **hourly rainfall** in the 23 local government areas of Benue State, Nigeria, built on the machine-learning workflow in `rainfall_prediction_workflow.ipynb`.

## Interface
The front end is the supplied single-page **Benue Weather** interface (sidebar workspace, live Open-Meteo weather, Leaflet map), served by Flask and wired to the Python model API:

| Section | Backend use |
|---|---|
| Dashboard / Forecast / Weather Map / Locations | Live weather from Open-Meteo (browser → Open-Meteo, internet needed) |
| **Quick Prediction** | `POST /api/predict` – rain probability, expected mm, risk level, confidence. Choose *Include recent rainfall* to enter last-1 h / 6 h / 24 h rain and use the rainfall-history model |
| **Model Performance** | `/api/metrics`, `/api/reference`, sample evaluation, **batch evaluation** (`/api/batch`, series, CSV download) |

Changes made to the supplied files: LGA spelling `Oturkpo` → `Otukpo` (dataset name; the API accepts both), optional rainfall inputs, richer prediction result, and a stray `</div>` fixed in the About section.

## Run it
```bash
./run.sh            # macOS / Linux   (run.bat on Windows)
# or manually
pip install -r requirements.txt
python app.py       # http://127.0.0.1:5000
```
Production: `python -m waitress --port=5000 app:app` or `docker build -t rainfall . && docker run -p 5000:5000 rainfall`.

Tests: `python -m unittest discover -s tests -v`

## Models
Two hurdle models (gradient-boosting classifier for P(rain) × gamma-loss regressor for the amount given rain), refit on 2020–2023.

| | Weather only | + Rainfall history |
|---|---|---|
| Rain detection ROC-AUC (2024 test) | 0.950 | 0.972 |
| Hourly R² | 0.32 | 0.53 |
| Hourly RMSE (mm) | 0.464 | 0.388 |

Held-out year 2024; all LGAs appear in every split.

## Batch file format
Columns: `datetime, lga, temperature_c, humidity_percent, wind_speed_kmh` (+ optional `rainfall_mm`). Strictly hourly per LGA, no gaps; the first 73 hours of each LGA are context only. A ready-made example is on the Batch page ("Download sample file").

## API
```
GET  /api/health | /api/lgas | /api/reference | /api/metrics | /api/climate
POST /api/predict   {"lga","temperature","humidity","wind_speed","temperature_1hr_ago","humidity_1hr_ago","date","hour","use_rain": auto|0|1, "rain_last_1h/6h/24h"}
POST /api/batch     multipart: file, use_rain=auto|0|1
GET  /api/batch/<id>/series?lga=Makurdi      GET /download/<id>.csv
```

## Project layout
```
app.py          Flask routes + JSON API          features.py   inference feature engineering (identical to training)
predictor.py    validation, model loading, inference     models/   trained models (joblib)
templates/index.html  static/css  static/js   the interface                           data/     reference stats, metrics, sample upload
tests/          unit/integration tests
```

Set `CORS_ORIGINS=*` (or an origin) if the interface is hosted on a different domain than the API (then set `window.BENUE_API_BASE_URL` before `index.js` loads).

## Limitations
- Outputs reflect the reanalysis-style data used for training, not rain-gauge measurements, and are not a substitute for official forecasts (e.g. NiMet).
- Quick prediction assumes steady conditions before the last hour; use batch mode with real history for reliable results.
- Models are trained on five years and 23 fixed locations; accuracy elsewhere is unknown.
- Batch results are held in memory for one hour; uploads are limited to 30 MB / 400,000 rows.
- Model files are pickled scikit-learn objects: only load model files you trust, and keep `scikit-learn==1.8.0`.
