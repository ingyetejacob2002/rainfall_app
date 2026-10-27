import io, sys, unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_upload.csv"
# payloads exactly as the web interface sends them
WET = dict(lga="Makurdi", temperature=25.5, humidity=91, wind_speed=11, temperature_1hr_ago=29.4, humidity_1hr_ago=74, date="2024-08-12", hour=18, use_rain="auto")
DRY = dict(lga="Makurdi", temperature=35, humidity=22, wind_speed=9, temperature_1hr_ago=34.5, humidity_1hr_ago=24, date="2024-01-12", hour=14, use_rain="auto")


class AppTests(unittest.TestCase):
    def setUp(self): self.c = app.test_client()

    def test_interface_and_static(self):
        r = self.c.get("/"); self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        for needle in ["Benue Weather", "predictionForm", "/static/css/style.css", "/static/js/index.js", "rain-field"]:
            self.assertIn(needle, html)
        for p in ["/static/css/style.css", "/static/js/index.js", "/sample.csv"]:
            r = self.c.get(p); self.assertEqual(r.status_code, 200, p); r.close()

    def test_contract_endpoints(self):
        self.assertEqual(self.c.get("/api/health").json["status"], "ok")
        names = self.c.get("/api/lgas").json
        self.assertEqual(len(names), 23); self.assertIn("Otukpo", names); self.assertTrue(all(isinstance(n, str) for n in names))
        m = self.c.get("/api/metrics").json
        self.assertTrue(0.3 < m["r2"] < 0.7 and m["rmse"] > 0 and m["mae"] > 0)
        ref = self.c.get("/api/reference").json
        self.assertIn("Weather-only model", ref); self.assertLessEqual(len(ref), 35)
        self.assertEqual(self.c.get("/nope").status_code, 404)

    def test_predict_wet_vs_dry(self):
        w, d = self.c.post("/api/predict", json=WET).json, self.c.post("/api/predict", json=DRY).json
        self.assertGreater(w["probability"], d["probability"] + 0.2); self.assertLess(d["probability"], 0.1)
        self.assertIsInstance(w["prediction"], float); self.assertIn("confidence", w); self.assertIn("prediction_label", w)

    def test_predict_aliases_and_rain_modes(self):
        r = self.c.post("/api/predict", json={**WET, "lga": "Oturkpo"}); self.assertEqual(r.status_code, 200)
        rain = {"rain_last_1h": 1.2, "rain_last_6h": 3.0, "rain_last_24h": 6.0}
        self.assertEqual(self.c.post("/api/predict", json={**WET, "use_rain": "1", **rain}).json["scenario"], "B_with_rain_history")
        self.assertEqual(self.c.post("/api/predict", json={**WET, "use_rain": "0", **rain}).json["scenario"], "A_weather_only")
        self.assertEqual(self.c.post("/api/predict", json={**WET, "use_rain": "1"}).status_code, 400)

    def test_predict_validation(self):
        for bad in [{**WET, "humidity": 140}, {**WET, "lga": "Atlantis"}, {**WET, "hour": None}, {**WET, "use_rain": "maybe"},
                    {**WET, "rain_last_1h": 5, "rain_last_6h": 3, "rain_last_24h": 9, "use_rain": "1"}]:
            self.assertEqual(self.c.post("/api/predict", json=bad).status_code, 400, bad)
        self.assertEqual(self.c.post("/api/predict", data="x", content_type="text/plain").status_code, 400)

    def _upload(self, df=None, **form):
        buf = io.BytesIO(SAMPLE.read_bytes() if df is None else df.to_csv(index=False).encode())
        return self.c.post("/api/batch", data={"file": (buf, "x.csv"), **form}, content_type="multipart/form-data")

    def test_batch_roundtrip(self):
        r = self._upload(); self.assertEqual(r.status_code, 200, r.json); j = r.json
        self.assertEqual(j["summary"]["rows_predicted"], 23 * (240 - 73)); self.assertIn("message", j)
        self.assertGreater(j["summary"]["evaluation"]["roc_auc"], 0.9)
        s = self.c.get(f"/api/batch/{j['id']}/series?lga=Gboko").json
        self.assertEqual(len(s["series"]), 167); self.assertEqual(set(s["series"][0]), {"time", "prediction", "probability", "actual"})
        self.assertIn("rmse", s["metrics"])
        self.assertEqual(self.c.get(f"/download/{j['id']}.csv").status_code, 200)
        self.assertEqual(self.c.get("/download/" + "0" * 32 + ".csv").status_code, 404)
        self.assertEqual(self.c.get(f"/api/batch/{j['id']}/series?lga=Nowhere").status_code, 404)
        self.assertEqual(self._upload(use_rain="0").json["summary"]["model"], "Weather only")

    def test_batch_errors(self):
        df = pd.read_csv(SAMPLE)
        self.assertEqual(self._upload(df.drop(columns=["humidity_percent"])).status_code, 400)
        r = self._upload(df.drop(df.index[500])); self.assertEqual(r.status_code, 400); self.assertIn("hourly", r.json["error"])
        self.assertEqual(self._upload(df.head(30)).status_code, 400)
        self.assertEqual(self._upload(df.assign(lga="Nowhere")).status_code, 400)
        self.assertEqual(self.c.post("/api/batch", data={}, content_type="multipart/form-data").status_code, 400)


if __name__ == "__main__":
    unittest.main()
