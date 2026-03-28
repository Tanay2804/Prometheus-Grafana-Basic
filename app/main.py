import os
import time
import pickle
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app
from app.schema import HeartRequest
from app.metrics import (
    http_requests_total,
    active_requests,
    request_duration_seconds,
    predictions_total,
)
import random

app = FastAPI()

# Load model + scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "model", "rf_model.pkl")
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
feature_names = bundle["features"]

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start_time = time.time()

    active_requests.inc()
    response = await call_next(request)
    active_requests.dec()

    duration = time.time() - start_time
    request_duration_seconds.observe(duration)

    http_requests_total.labels(
        endpoint=request.url.path, status_code=response.status_code
    ).inc()

    return response


@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Prediction API!"}


def classify_risk(pred):
    mapping = {
        0: "No disease",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Very severe"
    }
    return mapping[pred]


# ---------------------------------------------------------------
def generate_realistic_sample():
    return [
        int(random.gauss(54, 9)),  # age
        random.uniform(0, 1),  # sex
        random.uniform(1, 4),  # cp
        int(random.gauss(130, 20)),  # trestbps
        int(random.gauss(245, 50)),  # chol
        random.uniform(0, 1),  # fbs
        random.uniform(0, 2),  # restecg
        int(random.gauss(150, 22)),  # thalach
        random.uniform(0, 1),  # exang = 1 sometimes
        round(random.gauss(2.5, 1.0), 2),  # oldpeak
        random.uniform(1, 3),  # slope
        random.uniform(0, 3),  # ca
        random.choice([3, 6, 7]),  # thal (valid values)
    ]


@app.get("/predict")
def predict():
    features = generate_realistic_sample()

    prediction = model.predict([features])[0]
    risk = classify_risk(prediction)

    predictions_total.labels(risk_level=risk).inc()

    return {"features": features, "prediction": int(prediction), "risk_level": risk}


# example post payload:
# curl -X POST "http://127.0.0.1:8000/predict" \
# -H "Content-Type: application/json" \
# -d '{"features":[63,1,3,145,233,1,0,150,0,2.3,0,0,1]}'
# {"prediction":0,"risk_level":"Low"}
# --------------------------------------------------------------
@app.post("/simulate")
def simulate():
    import random

    for _ in range(200):
        features = [random.uniform(0, 1) for _ in range(13)]
        pred = model.predict([features])[0]
        risk = classify_risk(pred)
        predictions_total.labels(risk_level=risk).inc()

    return {"status": "200 requests simulated"}
