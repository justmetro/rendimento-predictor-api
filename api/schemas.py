from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    idade: int = Field(..., ge=18, le=80)
    sexo: str = Field(..., pattern="^[MF]$")
    cor_raca: str
    anos_estudo: int = Field(..., ge=0, le=20)
    setor: str
    regiao: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "idade": 35,
                "sexo": "M",
                "cor_raca": "Branca",
                "anos_estudo": 12,
                "setor": "Servicos",
                "regiao": "Sudeste"
            }
        }
    }


class PredictionOutput(BaseModel):
    rendimento_hora_previsto: float
    intervalo_confianca: dict
    features_usadas: dict
    modelo: str