from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True
    assert response.json()["database_connected"] is True


def test_features_endpoint():
    response = client.get("/features")

    assert response.status_code == 200

    data = response.json()

    assert data["target"] == "rendimento_hora"
    assert "features" in data
    assert "idade" in data["features"]
    assert "sexo" in data["features"]


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "random_forest_sintetico_v1"
    assert "rmse" in data
    assert "mae" in data
    assert "r2" in data
    assert "features" in data
    assert data["target"] == "rendimento_hora"


def test_real_model_info_endpoint():
    response = client.get("/model-info/real")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "random_forest_real_v1"
    assert data["data_source"] == "pnad_real_processed"
    assert "rmse" in data
    assert "mae" in data
    assert "r2" in data
    assert "features" in data
    assert data["target"] == "rendimento_hora"


def test_production_model_info_endpoint():
    response = client.get("/model-info/production")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "xgboost_pnad_real_production_v1"
    assert data["data_source"] == "pnad_real_processed"
    assert data["target"] == "rendimento_hora"
    assert "rmse" in data
    assert "mae" in data
    assert "r2" in data
    assert "features" in data


def test_model_comparison_endpoint():
    response = client.get("/model-comparison")

    assert response.status_code == 200

    data = response.json()

    assert data["data_source"] == "pnad_real_processed"
    assert data["target"] == "rendimento_hora"
    assert "models" in data
    assert len(data["models"]) >= 3
    assert "best_model_by_rmse" in data

    model_names = [model["model_name"] for model in data["models"]]

    assert "linear_regression" in model_names
    assert "random_forest" in model_names
    assert "xgboost" in model_names


def test_feature_importance_endpoint():
    response = client.get("/feature-importance")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "random_forest_sintetico_v1"
    assert data["model_type"] == "production"
    assert "feature_importance" in data
    assert len(data["feature_importance"]) > 0
    assert "feature" in data["feature_importance"][0]
    assert "importance" in data["feature_importance"][0]


def test_real_feature_importance_endpoint():
    response = client.get("/feature-importance/real")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "random_forest_real_v1"
    assert data["model_type"] == "candidate"
    assert "feature_importance" in data
    assert len(data["feature_importance"]) > 0
    assert "feature" in data["feature_importance"][0]
    assert "importance" in data["feature_importance"][0]


def test_predict_valid_input():
    payload = {
        "idade": 35,
        "sexo": "M",
        "cor_raca": "Branca",
        "anos_estudo": 12,
        "setor": "Servicos",
        "regiao": "Sudeste"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "rendimento_hora_previsto" in data
    assert "intervalo_confianca" in data
    assert "features_usadas" in data
    assert data["modelo"] == "xgboost_pnad_real_production_v1"


def test_history_endpoint():
    payload = {
        "idade": 35,
        "sexo": "M",
        "cor_raca": "Branca",
        "anos_estudo": 12,
        "setor": "Servicos",
        "regiao": "Sudeste"
    }

    prediction_response = client.post("/predict", json=payload)
    assert prediction_response.status_code == 200

    response = client.get("/history?limit=1")

    assert response.status_code == 200

    data = response.json()

    assert data["total_returned"] == 1
    assert len(data["predictions"]) == 1

    prediction = data["predictions"][0]

    assert "id" in prediction
    assert prediction["idade"] == payload["idade"]
    assert prediction["sexo"] == payload["sexo"]
    assert prediction["cor_raca"] == payload["cor_raca"]
    assert prediction["anos_estudo"] == payload["anos_estudo"]
    assert prediction["setor"] == payload["setor"]
    assert prediction["regiao"] == payload["regiao"]
    assert "rendimento_hora_previsto" in prediction
    assert "intervalo_confianca" in prediction
    assert "min" in prediction["intervalo_confianca"]
    assert "max" in prediction["intervalo_confianca"]
    assert prediction["modelo"] == "xgboost_pnad_real_production_v1"
    assert "created_at" in prediction


def test_predict_invalid_age():
    payload = {
        "idade": 10,
        "sexo": "M",
        "cor_raca": "Branca",
        "anos_estudo": 12,
        "setor": "Servicos",
        "regiao": "Sudeste"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_sex():
    payload = {
        "idade": 35,
        "sexo": "X",
        "cor_raca": "Branca",
        "anos_estudo": 12,
        "setor": "Servicos",
        "regiao": "Sudeste"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_missing_field():
    payload = {
        "idade": 35,
        "sexo": "M",
        "cor_raca": "Branca",
        "anos_estudo": 12,
        "setor": "Servicos"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
