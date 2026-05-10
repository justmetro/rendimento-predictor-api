import requests

from frontend.metadata import (
    FALLBACK_METADATA,
    fetch_metadata,
    get_categorical_options,
    get_numeric_constraint,
)


def test_get_numeric_constraint_returns_metadata_values():
    metadata = {
        "numeric_constraints": {
            "idade": {
                "min": 20,
                "max": 90,
            },
        },
    }

    assert get_numeric_constraint(metadata, "idade") == (20, 90)


def test_get_numeric_constraint_uses_fallback_when_field_is_missing():
    metadata = {
        "numeric_constraints": {},
    }

    assert get_numeric_constraint(metadata, "idade") == (14, 100)


def test_get_numeric_constraint_uses_fallback_when_numeric_constraints_is_missing():
    assert get_numeric_constraint({}, "anos_estudo") == (0, 20)


def test_get_categorical_options_returns_metadata_values():
    metadata = {
        "categorical_options": {
            "sexo": ["F", "M"],
        },
    }

    assert get_categorical_options(metadata, "sexo") == ["F", "M"]


def test_get_categorical_options_uses_fallback_when_field_is_missing():
    metadata = {
        "categorical_options": {},
    }

    assert get_categorical_options(metadata, "regiao") == [
        "Norte",
        "Nordeste",
        "Centro-Oeste",
        "Sudeste",
        "Sul",
    ]


def test_get_categorical_options_uses_fallback_when_categorical_options_is_missing():
    assert get_categorical_options({}, "sexo") == ["M", "F"]


def test_get_categorical_options_uses_fallback_when_options_are_empty():
    metadata = {
        "categorical_options": {
            "setor": [],
        },
    }

    assert get_categorical_options(metadata, "setor") == [
        "Servicos",
        "Industria",
        "Comercio",
        "Agricultura",
        "Construcao",
    ]


def test_fetch_metadata_returns_fallback_when_api_call_fails(monkeypatch):
    def raise_request_error(url, timeout):
        raise requests.exceptions.RequestException("metadata unavailable")

    monkeypatch.setattr("frontend.metadata.requests.get", raise_request_error)

    assert fetch_metadata("http://api.example") == FALLBACK_METADATA
