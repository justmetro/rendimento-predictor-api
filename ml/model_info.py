import json
from pathlib import Path


METRICS_PATH = Path("data/models/metrics.json")
REAL_METRICS_PATH = Path("data/models/metrics_real.json")


def load_model_metrics() -> dict:
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_real_model_metrics() -> dict:
    with open(REAL_METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)