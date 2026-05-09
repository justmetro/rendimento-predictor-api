import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data/processed/pnad_real_processed.csv"
MODEL_PATH = "data/models/rendimento_model_real.pkl"
METRICS_PATH = "data/models/metrics_real.json"


def load_real_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {DATA_PATH}. "
            "Rode primeiro: python -m scripts.prepare_real_data"
        )

    return pd.read_csv(DATA_PATH)


def train_real_model() -> dict:
    df = load_real_data()

    features = ["idade", "sexo", "cor_raca", "anos_estudo", "setor", "regiao"]
    target = "rendimento_hora"

    X = df[features]
    y = df[target]

    numeric_features = ["idade", "anos_estudo"]
    categorical_features = ["sexo", "cor_raca", "setor", "regiao"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=8,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    if len(df) < 10:
        raise ValueError(
            "Dataset real processado tem poucas linhas para treino. "
            f"Linhas encontradas: {len(df)}. "
            "Use mais dados antes de treinar o modelo real."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    os.makedirs("data/models", exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    metrics = {
        "model_name": "random_forest_real_v1",
        "data_source": "pnad_real_processed",
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 3),
        "n_rows": len(df),
        "features": features,
        "target": target,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    print(f"Modelo real treinado e salvo em: {MODEL_PATH}")
    print(f"Métricas reais salvas em: {METRICS_PATH}")
    print(metrics)

    return metrics


if __name__ == "__main__":
    train_real_model()