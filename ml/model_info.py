import json
from pathlib import Path


METRICS_PATH = Path("data/models/metrics.json")
REAL_METRICS_PATH = Path("data/models/metrics_real.json")
MODEL_COMPARISON_PATH = Path("data/models/model_comparison.json")


def load_json_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_model_metrics() -> dict:
    return load_json_file(METRICS_PATH)


def load_real_model_metrics() -> dict:
    return load_json_file(REAL_METRICS_PATH)


def load_model_comparison() -> dict:
    return load_json_file(MODEL_COMPARISON_PATH)