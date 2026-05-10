import requests

from core.config import (
    ANOS_ESTUDO_MAX,
    ANOS_ESTUDO_MIN,
    COR_RACA_OPTIONS,
    IDADE_MAX,
    IDADE_MIN,
    REGIAO_OPTIONS,
    SETOR_OPTIONS,
    SEXO_OPTIONS,
)


FALLBACK_METADATA = {
    "numeric_constraints": {
        "idade": {
            "min": IDADE_MIN,
            "max": IDADE_MAX,
        },
        "anos_estudo": {
            "min": ANOS_ESTUDO_MIN,
            "max": ANOS_ESTUDO_MAX,
        },
    },
    "categorical_options": {
        "sexo": SEXO_OPTIONS,
        "cor_raca": COR_RACA_OPTIONS,
        "setor": SETOR_OPTIONS,
        "regiao": REGIAO_OPTIONS,
    },
}


def fetch_metadata(api_url: str) -> dict:
    try:
        response = requests.get(
            f"{api_url}/metadata",
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return FALLBACK_METADATA


def get_numeric_constraint(metadata: dict, field: str) -> tuple[int, int]:
    fallback = FALLBACK_METADATA["numeric_constraints"][field]
    constraints = metadata.get("numeric_constraints", {}).get(field, {})

    return (
        int(constraints.get("min", fallback["min"])),
        int(constraints.get("max", fallback["max"])),
    )


def get_categorical_options(metadata: dict, field: str) -> list[str]:
    fallback = FALLBACK_METADATA["categorical_options"][field]
    options = metadata.get("categorical_options", {}).get(field)

    if not options:
        return fallback

    return list(options)
