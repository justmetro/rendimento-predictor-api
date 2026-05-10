import requests
import subprocess
import sys

from core.config import (
    ANOS_ESTUDO_MAX,
    ANOS_ESTUDO_MIN,
    IDADE_MAX,
    IDADE_MIN,
    REGIAO_OPTIONS,
    SETOR_OPTIONS,
    SEXO_OPTIONS,
)
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

    assert get_numeric_constraint(metadata, "idade") == (IDADE_MIN, IDADE_MAX)


def test_get_numeric_constraint_uses_fallback_when_numeric_constraints_is_missing():
    assert get_numeric_constraint({}, "anos_estudo") == (
        ANOS_ESTUDO_MIN,
        ANOS_ESTUDO_MAX,
    )


def test_get_numeric_constraint_uses_fallback_when_numeric_constraints_is_none():
    metadata = {
        "numeric_constraints": None,
    }

    assert get_numeric_constraint(metadata, "idade") == (IDADE_MIN, IDADE_MAX)


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

    assert get_categorical_options(metadata, "regiao") == REGIAO_OPTIONS


def test_get_categorical_options_uses_fallback_when_categorical_options_is_missing():
    assert get_categorical_options({}, "sexo") == SEXO_OPTIONS


def test_get_categorical_options_uses_fallback_when_categorical_options_is_none():
    metadata = {
        "categorical_options": None,
    }

    assert get_categorical_options(metadata, "sexo") == SEXO_OPTIONS


def test_get_categorical_options_uses_fallback_when_options_are_empty():
    metadata = {
        "categorical_options": {
            "setor": [],
        },
    }

    assert get_categorical_options(metadata, "setor") == SETOR_OPTIONS


def test_fetch_metadata_returns_fallback_when_api_call_fails(monkeypatch):
    def raise_request_error(url, timeout):
        raise requests.exceptions.RequestException("metadata unavailable")

    monkeypatch.setattr("frontend.metadata.requests.get", raise_request_error)

    assert fetch_metadata("http://api.example") == FALLBACK_METADATA


def test_metadata_can_be_imported_from_frontend_directory():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import metadata; assert metadata.FALLBACK_METADATA",
        ],
        cwd="frontend",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
