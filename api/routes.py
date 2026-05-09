from fastapi import APIRouter

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
        "features": [
            "idade",
            "sexo",
            "cor_raca",
            "anos_estudo",
            "setor",
            "regiao"
        ]
    }