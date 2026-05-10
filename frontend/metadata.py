import requests

try:
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
except ModuleNotFoundError:
    IDADE_MIN = 14
    IDADE_MAX = 100
    ANOS_ESTUDO_MIN = 0
    ANOS_ESTUDO_MAX = 20
    SEXO_OPTIONS = ["M", "F"]
    COR_RACA_OPTIONS = ["Branca", "Preta", "Parda", "Amarela", "Indigena"]
    SETOR_OPTIONS = ["Servicos", "Industria", "Comercio", "Agricultura", "Construcao"]
    REGIAO_OPTIONS = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


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
    numeric_constraints = metadata.get("numeric_constraints") or {}
    constraints = numeric_constraints.get(field) or {}

    return (
        int(constraints.get("min", fallback["min"])),
        int(constraints.get("max", fallback["max"])),
    )


def get_categorical_options(metadata: dict, field: str) -> list[str]:
    fallback = FALLBACK_METADATA["categorical_options"][field]
    categorical_options = metadata.get("categorical_options") or {}
    options = categorical_options.get(field)

    if not options:
        return fallback

    return list(options)
