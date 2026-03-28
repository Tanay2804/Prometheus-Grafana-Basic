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
    return "High" if pred == 1 else "Low"


@app.post("/predict")
def predict(data: HeartRequest):
    # Expect 13 features for Cleveland dataset

    prediction = model.predict([data.features])[0]
    risk = classify_risk(prediction)
    predictions_total.labels(risk_level=risk).inc()
    return {"prediction": int(prediction), "risk_level": risk}


# example post payload: 
# curl -X POST "http://127.0.0.1:8000/predict" \
# -H "Content-Type: application/json" \
# -d '{"features":[63,1,3,145,233,1,0,150,0,2.3,0,0,1]}'
# {"prediction":0,"risk_level":"Low"}

@app.post("/simulate")
def simulate():
    import random

    for _ in range(200):
        features = [random.uniform(0, 1) for _ in range(13)]
        pred = model.predict([features])[0]
        risk = classify_risk(pred)
        predictions_total.labels(risk_level=risk).inc()

    return {"status": "200 requests simulated"}
