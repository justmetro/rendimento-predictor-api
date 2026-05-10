import requests


FALLBACK_METADATA = {
    "numeric_constraints": {
        "idade": {
            "min": 14,
            "max": 100,
        },
        "anos_estudo": {
            "min": 0,
            "max": 20,
        },
    },
    "categorical_options": {
        "sexo": ["M", "F"],
        "cor_raca": ["Branca", "Preta", "Parda", "Amarela", "Indigena"],
        "setor": ["Servicos", "Industria", "Comercio", "Agricultura", "Construcao"],
        "regiao": ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
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
