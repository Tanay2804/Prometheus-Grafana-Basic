from prometheus_client import Counter, Gauge, Histogram


http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["endpoint", "status_code"],
)

active_requests = Gauge(
    "active_requests",
    "Number of active HTTP requests",
)

request_duration_seconds = Histogram(
    "request_duration_seconds",
    "HTTP request latency in seconds",
)

predictions_total = Counter(
    "predictions_total",
    "Total predictions by risk level",
    ["risk_level"],
)
