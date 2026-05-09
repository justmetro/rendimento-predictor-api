import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def generate_synthetic_data(n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    idade = np.random.randint(18, 66, n_rows)
    anos_estudo = np.random.randint(0, 21, n_rows)

    sexo = np.random.choice(["M", "F"], n_rows)
    cor_raca = np.random.choice(["Branca", "Preta", "Parda", "Amarela", "Indigena"], n_rows)
    setor = np.random.choice(["Servicos", "Industria", "Comercio", "Agricultura", "Construcao"], n_rows)
    regiao = np.random.choice(["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"], n_rows)

    rendimento_hora = (
        8
        + idade * 0.12
        + anos_estudo * 2.8
        + np.where(sexo == "M", 2.0, 0.0)
        + np.where(regiao == "Sudeste", 5.0, 0.0)
        + np.where(regiao == "Sul", 4.0, 0.0)
        + np.where(setor == "Servicos", 3.0, 0.0)
        + np.random.normal(0, 6, n_rows)
    )

    rendimento_hora = np.maximum(rendimento_hora, 5)

    return pd.DataFrame(
        {
            "idade": idade,
            "sexo": sexo,
            "cor_raca": cor_raca,
            "anos_estudo": anos_estudo,
            "setor": setor,
            "regiao": regiao,
            "rendimento_hora": rendimento_hora,
        }
    )


def train_model():
    df = generate_synthetic_data()

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
    joblib.dump(pipeline, "data/models/rendimento_model.pkl")

    metrics = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 3),
    }

    print("Modelo treinado e salvo em data/models/rendimento_model.pkl")
    print(metrics)

    return metrics


if __name__ == "__main__":
    train_model()