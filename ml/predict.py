import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = "data/models/rendimento_model_production.pkl"
METRICS_PATH = Path("data/models/metrics_production.json")
MODEL_NAME = "xgboost_pnad_real_production_v1"


def load_model():
    return joblib.load(MODEL_PATH)


def load_prediction_interval_residuals(metrics_path: Path = METRICS_PATH) -> dict:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    residuals = metrics.get("prediction_interval_residuals")

    if not residuals:
        rmse = float(metrics["rmse"])
        return {
            "lower_residual_p05": -rmse,
            "upper_residual_p95": rmse,
        }

    return residuals


def build_prediction_interval(prediction: float) -> dict:
    residuals = load_prediction_interval_residuals()
    intervalo_min = max(
        0,
        min(prediction, prediction + float(residuals["lower_residual_p05"])),
    )
    intervalo_max = max(prediction, prediction + float(residuals["upper_residual_p95"]))

    return {
        "min": round(intervalo_min, 2),
        "max": round(intervalo_max, 2),
    }


model = load_model()


def predict_rendimento(input_data: dict) -> float:
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return float(prediction)
