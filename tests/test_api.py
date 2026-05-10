from fastapi.testclient import TestClient

import api.routes as routes
import api.rate_limit as rate_limit
from database.database import get_db
from ml.predict import PredictionModelError
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
    assert isinstance(data["n_rows"], int)
    assert isinstance(data["features"], list)


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
    assert isinstance(data["n_rows"], int)
    assert isinstance(data["features"], list)


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
    assert isinstance(data["n_rows"], int)
    assert isinstance(data["features"], list)
    assert data["prediction_interval_method"] == "residual_percentile_5_95"
    assert "prediction_interval_residuals" in data
    assert "lower_residual_p05" in data["prediction_interval_residuals"]
    assert "upper_residual_p95" in data["prediction_interval_residuals"]
    assert data["promoted_from"] == "model_comparison"
    assert "note" in data


def test_model_comparison_endpoint():
    response = client.get("/model-comparison")

    assert response.status_code == 200

    data = response.json()

    assert data["data_source"] == "pnad_real_processed"
    assert data["target"] == "rendimento_hora"
    assert "models" in data
    assert len(data["models"]) >= 3
    assert "best_model_by_rmse" in data
    assert isinstance(data["n_rows"], int)
    assert isinstance(data["features"], list)

    model_names = [model["model_name"] for model in data["models"]]

    assert "linear_regression" in model_names
    assert "random_forest" in model_names
    assert "xgboost" in model_names
    assert {"model_name", "rmse", "mae", "r2", "cv_r2_mean", "cv_r2_std"}.issubset(
        data["models"][0]
    )


def test_model_info_openapi_response_model_contracts():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    expected_refs = {
        "/model-info": "#/components/schemas/ModelInfoOutput",
        "/model-info/real": "#/components/schemas/RealModelInfoOutput",
        "/model-info/production": "#/components/schemas/ProductionModelInfoOutput",
        "/model-comparison": "#/components/schemas/ModelComparisonOutput",
    }

    for path, expected_ref in expected_refs.items():
        response_schema = schema["paths"][path]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]

        assert response_schema == {"$ref": expected_ref}

    model_info = schema["components"]["schemas"]["ModelInfoOutput"]
    assert set(model_info["required"]) == {
        "model_name",
        "rmse",
        "mae",
        "r2",
        "n_rows",
        "features",
        "target",
    }

    comparison = schema["components"]["schemas"]["ModelComparisonOutput"]
    assert set(comparison["required"]) == {
        "data_source",
        "n_rows",
        "target",
        "features",
        "models",
        "best_model_by_rmse",
    }


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


def test_production_pnad_feature_importance_endpoint():
    response = client.get("/feature-importance/production")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "xgboost_pnad_real_production_v1"
    assert data["model_type"] == "production_pnad_real"
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
    assert isinstance(data["rendimento_hora_previsto"], (int, float))
    assert data["features_usadas"] == payload

    intervalo = data["intervalo_confianca"]

    assert "min" in intervalo
    assert "max" in intervalo
    assert isinstance(intervalo["min"], (int, float))
    assert isinstance(intervalo["max"], (int, float))
    assert intervalo["min"] <= data["rendimento_hora_previsto"] <= intervalo["max"]


def test_predict_openapi_response_model_contract():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()
    predict_schema = schema["paths"]["/predict"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]

    assert predict_schema == {"$ref": "#/components/schemas/PredictionOutput"}

    prediction_output = schema["components"]["schemas"]["PredictionOutput"]
    assert set(prediction_output["required"]) == {
        "rendimento_hora_previsto",
        "intervalo_confianca",
        "features_usadas",
        "modelo",
    }
    assert prediction_output["properties"]["rendimento_hora_previsto"]["type"] == "number"
    assert prediction_output["properties"]["intervalo_confianca"] == {
        "$ref": "#/components/schemas/PredictionInterval"
    }
    assert prediction_output["properties"]["features_usadas"] == {
        "$ref": "#/components/schemas/PredictionFeatures"
    }
    assert prediction_output["properties"]["modelo"]["type"] == "string"


def test_predict_rate_limit_returns_429(monkeypatch):
    rate_limit.clear_rate_limit_state()
    monkeypatch.setattr(rate_limit, "PREDICT_RATE_LIMIT", 1)
    monkeypatch.setattr(rate_limit, "PREDICT_RATE_WINDOW_SECONDS", 60)

    payload = {
        "idade": 35,
        "sexo": "M",
        "cor_raca": "Branca",
        "anos_estudo": 12,
        "setor": "Servicos",
        "regiao": "Sudeste"
    }

    try:
        first_response = client.post("/predict", json=payload)
        second_response = client.post("/predict", json=payload)
    finally:
        rate_limit.clear_rate_limit_state()

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert second_response.json()["detail"] == (
        "Limite de requisições excedido. Tente novamente em instantes."
    )


def test_predict_returns_response_when_history_save_fails():
    class FailingCommitDb:
        rollback_called = False

        def add(self, record):
            self.record = record

        def commit(self):
            from sqlalchemy.exc import SQLAlchemyError

            raise SQLAlchemyError("commit failed")

        def rollback(self):
            self.rollback_called = True

    fake_db = FailingCommitDb()
    previous_override = app.dependency_overrides.get(get_db)

    def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    payload = {
        "idade": 35,
        "sexo": "M",
        "cor_raca": "Branca",
        "anos_estudo": 12,
        "setor": "Servicos",
        "regiao": "Sudeste"
    }

    try:
        response = client.post("/predict", json=payload)
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = previous_override

    assert response.status_code == 200
    assert fake_db.rollback_called is True

    data = response.json()

    assert "rendimento_hora_previsto" in data
    assert "intervalo_confianca" in data
    assert data["modelo"] == "xgboost_pnad_real_production_v1"


def test_predict_returns_friendly_error_when_model_prediction_fails(monkeypatch):
    def raise_prediction_error(input_data):
        raise PredictionModelError("internal model failure")

    monkeypatch.setattr(routes, "predict_rendimento", raise_prediction_error)

    payload = {
        "idade": 35,
        "sexo": "M",
        "cor_raca": "Branca",
        "anos_estudo": 12,
        "setor": "Servicos",
        "regiao": "Sudeste"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Erro interno ao gerar predição."


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

    assert isinstance(data, dict)
    assert data["total_returned"] == 1
    assert isinstance(data["total_returned"], int)
    assert isinstance(data["predictions"], list)
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
    assert isinstance(prediction["rendimento_hora_previsto"], (int, float))
    assert "intervalo_confianca" in prediction
    assert "min" in prediction["intervalo_confianca"]
    assert "max" in prediction["intervalo_confianca"]
    assert isinstance(prediction["intervalo_confianca"]["min"], (int, float))
    assert isinstance(prediction["intervalo_confianca"]["max"], (int, float))
    assert (
        prediction["intervalo_confianca"]["min"]
        <= prediction["rendimento_hora_previsto"]
        <= prediction["intervalo_confianca"]["max"]
    )
    assert prediction["modelo"] == "xgboost_pnad_real_production_v1"
    assert "created_at" in prediction


def test_history_openapi_response_model_contract():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()
    history_schema = schema["paths"]["/history"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert history_schema == {"$ref": "#/components/schemas/HistoryOutput"}

    history_output = schema["components"]["schemas"]["HistoryOutput"]
    assert set(history_output["required"]) == {"total_returned", "predictions"}
    assert history_output["properties"]["total_returned"]["type"] == "integer"
    assert history_output["properties"]["predictions"]["type"] == "array"
    assert history_output["properties"]["predictions"]["items"] == {
        "$ref": "#/components/schemas/HistoryPredictionItem"
    }

    history_item = schema["components"]["schemas"]["HistoryPredictionItem"]
    assert set(history_item["required"]) == {
        "id",
        "idade",
        "sexo",
        "cor_raca",
        "anos_estudo",
        "setor",
        "regiao",
        "rendimento_hora_previsto",
        "intervalo_confianca",
        "modelo",
        "created_at",
    }


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
