from http.server import BaseHTTPRequestHandler
import json, os
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = joblib.load(os.path.join(BASE, "model", "fraud_model.joblib"))
MODEL = ART["model"]
PREPROCESSOR = ART["preprocessor"]
FEATURES = ART["features"]
LOW, HIGH = float(ART["low"]), float(ART["high"])

def prepare_rows(rows):
    df = pd.DataFrame(rows)
    # Accept the app's readable features, or raw dataset rows containing
    # trans_date_trans_time and amt/category.
    if "trans_date_trans_time" in df.columns:
        dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        if "hour" not in df.columns:
            df["hour"] = dt.dt.hour
        if "day_of_week" not in df.columns:
            df["day_of_week"] = dt.dt.dayofweek
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    df = df[FEATURES].copy()
    for c in ["amt", "hour", "day_of_week"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["category"] = df["category"].fillna("unknown").astype(str)
    if df[["amt", "hour", "day_of_week"]].isna().any().any():
        raise ValueError("amt, hour and day_of_week must be numeric.")
    return df

def score(rows):
    df = prepare_rows(rows)
    encoded = PREPROCESSOR.transform(df)
    probabilities = MODEL.predict_proba(encoded)[:, 1]
    results = []
    for p in probabilities:
        p = float(p)
        if p >= HIGH:
            decision = "fraud"
        elif p <= LOW:
            decision = "genuine"
        else:
            decision = "review"
        results.append({"probability": round(p, 4), "decision": decision})
    return results

class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, {"status": "ok"})

    def do_GET(self):
        self._send(200, {"status": "ok", "low": LOW, "high": HIGH,
                         "features": FEATURES})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n))
            rows = payload.get("rows")
            if not isinstance(rows, list) or not 1 <= len(rows) <= 5000:
                raise ValueError("Send between 1 and 5000 rows.")
            self._send(200, {"results": score(rows), "low": LOW, "high": HIGH})
        except Exception as e:
            self._send(400, {"error": str(e)})
