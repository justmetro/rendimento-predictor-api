from fastapi.testclient import TestClient
import pytest

import api.routes as routes
import api.rate_limit as rate_limit
from core.config import (
    ANOS_ESTUDO_MAX,
    ANOS_ESTUDO_MIN,
    APP_NAME,
    APP_VERSION,
    COR_RACA_OPTIONS,
    IDADE_MAX,
    IDADE_MIN,
    REGIAO_OPTIONS,
    SETOR_OPTIONS,
    SEXO_OPTIONS,
)
from database.database import get_db
from ml.predict import PredictionModelError
from main import app

client = TestClient(app)


def valid_prediction_payload():
    return {
        "idade": 35,
        "sexo": "M",
        "cor_raca": "Branca",
        "anos_estudo": 12,
        "setor": "Servicos",
        "regiao": "Sudeste"
    }


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["database_connected"] is True
    assert data["app_name"] == APP_NAME
    assert data["model_name"] == "xgboost_pnad_real_production_v1"
    assert data["version"] == APP_VERSION


def test_health_openapi_response_model_contract():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()
    health_schema = schema["paths"]["/health"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert health_schema == {"$ref": "#/components/schemas/HealthOutput"}

    health_output = schema["components"]["schemas"]["HealthOutput"]
    assert set(health_output["required"]) == {
        "app_name",
        "status",
        "model_loaded",
        "database_connected",
        "model_name",
        "version",
    }
    assert health_output["properties"]["model_loaded"]["type"] == "boolean"
    assert health_output["properties"]["database_connected"]["type"] == "boolean"


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert data["app_name"] == APP_NAME
    assert data["status"] == "ok"
    assert data["model_name"] == "xgboost_pnad_real_production_v1"
    assert isinstance(data["total_predictions"], int)
    assert data["model_rmse"] == 5.86
    assert data["model_mae"] == 4.34
    assert data["model_r2"] == 0.333
    assert isinstance(data["model_rmse"], (int, float))
    assert isinstance(data["model_mae"], (int, float))
    assert isinstance(data["model_r2"], (int, float))


def test_metrics_returns_controlled_response_when_database_fails():
    class FailingQueryDb:
        rollback_called = False

        def query(self, model):
            from sqlalchemy.exc import SQLAlchemyError

            raise SQLAlchemyError("query failed")

        def rollback(self):
            self.rollback_called = True

    fake_db = FailingQueryDb()
    previous_override = app.dependency_overrides.get(get_db)

    def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    try:
        response = client.get("/metrics")
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_db, None)
        else:
            app.dependency_overrides[get_db] = previous_override

    assert response.status_code == 200
    assert fake_db.rollback_called is True
    assert response.json() == {
        "app_name": APP_NAME,
        "status": "degraded",
        "model_name": "xgboost_pnad_real_production_v1",
        "total_predictions": 0,
        "model_rmse": 5.86,
        "model_mae": 4.34,
        "model_r2": 0.333,
    }


def test_metrics_openapi_response_model_contract():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()
    metrics_schema = schema["paths"]["/metrics"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert metrics_schema == {"$ref": "#/components/schemas/MetricsOutput"}

    metrics_output = schema["components"]["schemas"]["MetricsOutput"]
    assert set(metrics_output["required"]) == {
        "app_name",
        "status",
        "model_name",
        "total_predictions",
        "model_rmse",
        "model_mae",
        "model_r2",
    }
    assert metrics_output["properties"]["total_predictions"]["type"] == "integer"
    assert metrics_output["properties"]["model_rmse"]["type"] == "number"
    assert metrics_output["properties"]["model_mae"]["type"] == "number"
    assert metrics_output["properties"]["model_r2"]["type"] == "number"


def test_metadata_endpoint():
    response = client.get("/metadata")

    assert response.status_code == 200

    data = response.json()

    assert data["app_name"] == APP_NAME
    assert data["version"] == APP_VERSION
    assert data["model_name"] == "xgboost_pnad_real_production_v1"
    assert data["prediction_endpoint"] == "/predict"
    assert data["numeric_constraints"]["idade"] == {"min": IDADE_MIN, "max": IDADE_MAX}
    assert data["numeric_constraints"]["anos_estudo"] == {
        "min": ANOS_ESTUDO_MIN,
        "max": ANOS_ESTUDO_MAX,
    }
    assert data["categorical_options"]["sexo"] == SEXO_OPTIONS
    assert data["categorical_options"]["cor_raca"] == COR_RACA_OPTIONS
    assert data["categorical_options"]["setor"] == SETOR_OPTIONS
    assert data["categorical_options"]["regiao"] == REGIAO_OPTIONS


def test_metadata_openapi_response_model_contract():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()
    metadata_schema = schema["paths"]["/metadata"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]

    assert metadata_schema == {"$ref": "#/components/schemas/MetadataOutput"}

    metadata_output = schema["components"]["schemas"]["MetadataOutput"]
    assert set(metadata_output["required"]) == {
        "app_name",
        "version",
        "model_name",
        "prediction_endpoint",
        "numeric_constraints",
        "categorical_options",
    }
    assert metadata_output["properties"]["numeric_constraints"]["type"] == "object"
    assert metadata_output["properties"]["categorical_options"]["type"] == "object"


def test_features_endpoint():
    response = client.get("/features")

    assert response.status_code == 200

    data = response.json()

    assert data["target"] == "rendimento_hora"
    assert "features" in data
    assert set(data["features"]) == {
        "idade",
        "sexo",
        "cor_raca",
        "anos_estudo",
        "setor",
        "regiao",
    }
    assert data["features"]["idade"] == {
        "type": "int",
        "min": IDADE_MIN,
        "max": IDADE_MAX,
    }
    assert data["features"]["anos_estudo"] == {
        "type": "int",
        "min": ANOS_ESTUDO_MIN,
        "max": ANOS_ESTUDO_MAX,
    }
    assert data["features"]["sexo"] == {
        "type": "category",
        "values": SEXO_OPTIONS,
    }
    assert data["features"]["cor_raca"] == {
        "type": "category",
        "values": COR_RACA_OPTIONS,
    }
    assert data["features"]["setor"] == {
        "type": "category",
        "values": SETOR_OPTIONS,
    }
    assert data["features"]["regiao"] == {
        "type": "category",
        "values": REGIAO_OPTIONS,
    }


def test_features_openapi_response_model_contract():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()
    features_schema = schema["paths"]["/features"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]

    assert "/features" in schema["paths"]
    assert "get" in schema["paths"]["/features"]
    assert features_schema == {"$ref": "#/components/schemas/FeaturesOutput"}
    assert "FeaturesOutput" in schema["components"]["schemas"]
    assert "FeaturesInfo" in schema["components"]["schemas"]
    assert "FeatureNumericInfo" in schema["components"]["schemas"]
    assert "FeatureCategoricalInfo" in schema["components"]["schemas"]

    features_output = schema["components"]["schemas"]["FeaturesOutput"]
    assert set(features_output["required"]) == {"target", "features"}


def test_features_and_metadata_are_consistent():
    features_response = client.get("/features")
    metadata_response = client.get("/metadata")

    assert features_response.status_code == 200
    assert metadata_response.status_code == 200

    features = features_response.json()["features"]
    metadata = metadata_response.json()
    numeric_constraints = metadata["numeric_constraints"]
    categorical_options = metadata["categorical_options"]

    assert features["idade"]["min"] == numeric_constraints["idade"]["min"]
    assert features["idade"]["max"] == numeric_constraints["idade"]["max"]
    assert features["anos_estudo"]["min"] == numeric_constraints["anos_estudo"]["min"]
    assert features["anos_estudo"]["max"] == numeric_constraints["anos_estudo"]["max"]
    assert features["sexo"]["values"] == categorical_options["sexo"]
    assert features["cor_raca"]["values"] == categorical_options["cor_raca"]
    assert features["setor"]["values"] == categorical_options["setor"]
    assert features["regiao"]["values"] == categorical_options["regiao"]


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


def test_predict_openapi_input_validation_contract():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()
    prediction_input = schema["components"]["schemas"]["PredictionInput"]
    properties = prediction_input["properties"]

    assert properties["idade"]["minimum"] == IDADE_MIN
    assert properties["idade"]["maximum"] == IDADE_MAX
    assert properties["anos_estudo"]["minimum"] == ANOS_ESTUDO_MIN
    assert properties["anos_estudo"]["maximum"] == ANOS_ESTUDO_MAX
    assert properties["sexo"]["enum"] == SEXO_OPTIONS
    assert properties["cor_raca"]["enum"] == COR_RACA_OPTIONS
    assert properties["setor"]["enum"] == SETOR_OPTIONS
    assert properties["regiao"]["enum"] == REGIAO_OPTIONS
    assert properties["idade"]["description"] == "Idade da pessoa em anos completos."
    assert properties["sexo"]["description"] == (
        "Sexo informado no formato categórico aceito pelo modelo."
    )
    assert properties["cor_raca"]["description"] == (
        "Categoria de cor ou raça aceita pelo modelo."
    )
    assert properties["anos_estudo"]["description"] == (
        "Quantidade de anos de estudo concluídos."
    )
    assert properties["setor"]["description"] == (
        "Setor de atividade profissional aceito pelo modelo."
    )
    assert properties["regiao"]["description"] == (
        "Região do Brasil associada ao registro."
    )
    assert properties["idade"]["examples"] == [35]
    assert properties["sexo"]["examples"] == ["M"]
    assert properties["cor_raca"]["examples"] == ["Branca"]
    assert properties["anos_estudo"]["examples"] == [12]
    assert properties["setor"]["examples"] == ["Servicos"]
    assert properties["regiao"]["examples"] == ["Sudeste"]


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


def test_history_endpoint_is_public_and_anonymized():
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
    assert "idade" not in prediction
    assert "sexo" not in prediction
    assert "cor_raca" not in prediction
    assert prediction["faixa_etaria"] == "35-44"
    assert prediction["anos_estudo"] == payload["anos_estudo"]
    assert prediction["escolaridade"] == "Medio completo"
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
        "faixa_etaria",
        "anos_estudo",
        "escolaridade",
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


def test_predict_age_above_max_returns_422():
    payload = {
        "idade": 101,
        "sexo": "M",
        "cor_raca": "Branca",
        "anos_estudo": 12,
        "setor": "Servicos",
        "regiao": "Sudeste"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_years_of_study_returns_422():
    payload = {
        "idade": 35,
        "sexo": "M",
        "cor_raca": "Branca",
        "anos_estudo": 21,
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


def test_predict_invalid_region_returns_422():
    payload = {
        "idade": 35,
        "sexo": "M",
        "cor_raca": "Branca",
        "anos_estudo": 12,
        "setor": "Servicos",
        "regiao": "Exterior"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "missing_field",
    ["idade", "sexo", "cor_raca", "anos_estudo", "setor", "regiao"],
)
def test_predict_missing_required_fields_return_422(missing_field):
    payload = valid_prediction_payload()
    payload.pop(missing_field)

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("idade", "trinta"),
        ("anos_estudo", "doze"),
    ],
)
def test_predict_invalid_numeric_field_types_return_422(field, invalid_value):
    payload = valid_prediction_payload()
    payload[field] = invalid_value

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
