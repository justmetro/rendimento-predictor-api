from fastapi import APIRouter

from api.schemas import PredictionInput, PredictionOutput

router = APIRouter()


@router.get("/")
def root():
    return {
        "message": "Rendimento Predictor API",
        "status": "running",
        "docs": "/docs"
    }


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": False,
        "version": "0.1.0"
    }


@router.get("/features")
def get_features():
    return {
        "target": "rendimento_hora",
        "features": {
            "idade": {
                "type": "int",
                "min": 18,
                "max": 80
            },
            "sexo": {
                "type": "category",
                "values": ["M", "F"]
            },
            "cor_raca": {
                "type": "category",
                "values": ["Branca", "Preta", "Parda", "Amarela", "Indigena"]
            },
            "anos_estudo": {
                "type": "int",
                "min": 0,
                "max": 20
            },
            "setor": {
                "type": "category",
                "values": ["Servicos", "Industria", "Comercio", "Agricultura", "Construcao"]
            },
            "regiao": {
                "type": "category",
                "values": ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
            }
        }
    }


@router.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    rendimento_base = 10.0

    rendimento_previsto = (
        rendimento_base
        + data.idade * 0.15
        + data.anos_estudo * 2.5
    )

    return {
        "rendimento_hora_previsto": round(rendimento_previsto, 2),
        "intervalo_confianca": {
            "min": round(rendimento_previsto * 0.85, 2),
            "max": round(rendimento_previsto * 1.15, 2)
        },
        "features_usadas": data.model_dump(),
        "modelo": "baseline_temporario"
    }