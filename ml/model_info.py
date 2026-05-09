import json

METRICS_PATH = "data/models/metrics.json"


def load_model_metrics() -> dict:
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)