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
    assert response.json()["model_loaded"] is False


def test_features_endpoint():
    response = client.get("/features")

    assert response.status_code == 200

    data = response.json()

    assert data["target"] == "rendimento_hora"
    assert "features" in data
    assert "idade" in data["features"]
    assert "sexo" in data["features"]


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
    assert data["modelo"] == "random_forest_sintetico_v1"


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