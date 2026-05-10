import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DATA_PATH = Path("data/processed/pnad_real_processed.csv")
MODEL_PATH = Path("data/models/rendimento_model_production.pkl")
METRICS_PATH = Path("data/models/metrics_production.json")

FEATURES = ["idade", "sexo", "cor_raca", "anos_estudo", "setor", "regiao"]
TARGET = "rendimento_hora"
NUMERIC_FEATURES = ["idade", "anos_estudo"]
CATEGORICAL_FEATURES = ["sexo", "cor_raca", "setor", "regiao"]

MODEL_NAME = "xgboost_pnad_real_production_v1"
DATA_SOURCE = "pnad_real_processed"
PROMOTED_FROM = "model_comparison"
NOTE = "Modelo treinado com microdados reais da PNAD Contínua 2023 trimestre 1"


def load_real_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {data_path}. "
            "Gere primeiro data/processed/pnad_real_processed.csv."
        )

    return pd.read_csv(data_path)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    model = XGBRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=5,
        learning_rate=0.08,
        objective="reg:squarederror",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_metrics(
    rmse: float,
    mae: float,
    r2: float,
    n_rows: int,
) -> dict:
    return {
        "model_name": MODEL_NAME,
        "data_source": DATA_SOURCE,
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 3),
        "n_rows": n_rows,
        "features": FEATURES,
        "target": TARGET,
        "promoted_from": PROMOTED_FROM,
        "note": NOTE,
    }


def train_production_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    df = load_real_data(data_path)

    if len(df) < 30:
        raise ValueError(
            "Dataset real processado tem poucas linhas para treino de producao. "
            f"Linhas encontradas: {len(df)}."
        )

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metrics = build_metrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        n_rows=len(df),
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina modelo XGBoost de producao com PNAD real processada."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Caminho do CSV processado da PNAD real.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        metrics = train_production_model(data_path=args.data_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Metricas salvas em: {METRICS_PATH}")
    print(json.dumps(metrics, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
