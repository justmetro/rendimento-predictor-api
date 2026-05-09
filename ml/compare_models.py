import json
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DATA_PATH = "data/processed/pnad_real_processed.csv"
OUTPUT_PATH = "data/models/model_comparison.json"


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {DATA_PATH}. "
            "Rode primeiro: python -m scripts.prepare_real_data"
        )

    return pd.read_csv(DATA_PATH)


def build_pipeline(model):
    numeric_features = ["idade", "anos_estudo"]
    categorical_features = ["sexo", "cor_raca", "setor", "regiao"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
    )

    return {
        "model_name": name,
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 3),
        "cv_r2_mean": round(float(cv_scores.mean()), 3),
        "cv_r2_std": round(float(cv_scores.std()), 3),
    }


def compare_models() -> dict:
    df = load_data()

    if len(df) < 30:
        raise ValueError(
            "Dataset tem poucas linhas para comparação de modelos. "
            f"Linhas encontradas: {len(df)}."
        )

    features = ["idade", "sexo", "cor_raca", "anos_estudo", "setor", "regiao"]
    target = "rendimento_hora"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=8,
        ),
        "xgboost": XGBRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=5,
            learning_rate=0.08,
            objective="reg:squarederror",
        ),
    }

    results = []

    for name, model in models.items():
        pipeline = build_pipeline(model)

        metrics = evaluate_model(
            name=name,
            pipeline=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(metrics)

    best_model = min(results, key=lambda item: item["rmse"])

    comparison = {
        "data_source": "pnad_real_processed",
        "n_rows": len(df),
        "target": target,
        "features": features,
        "models": results,
        "best_model_by_rmse": best_model["model_name"],
    }

    os.makedirs("data/models", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4, ensure_ascii=False)

    print(f"Comparação de modelos salva em: {OUTPUT_PATH}")
    print(comparison)

    return comparison


if __name__ == "__main__":
    compare_models()