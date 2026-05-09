from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.schemas import PredictionInput, PredictionOutput
from database.database import check_database_connection, get_db
from database.models import PredictionRecord
from ml.feature_importance import (
    get_production_feature_importance,
    get_real_feature_importance,
)
from ml.model_info import (
    load_model_comparison,
    load_model_metrics,
    load_real_model_metrics,
)
from ml.predict import predict_rendimento

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
        "model_loaded": True,
        "database_connected": check_database_connection(),
        "version": "1.3.0"
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


@router.get("/model-info")
def get_model_info():
    return load_model_metrics()


@router.get("/model-info/real")
def get_real_model_info():
    return load_real_model_metrics()


@router.get("/model-comparison")
def get_model_comparison():
    return load_model_comparison()


@router.get("/feature-importance")
def get_feature_importance():
    return get_production_feature_importance()


@router.get("/feature-importance/real")
def get_real_feature_importance_endpoint():
    return get_real_feature_importance()


@router.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput, db: Session = Depends(get_db)):
    input_data = data.model_dump()

    rendimento_previsto = predict_rendimento(input_data)
    rendimento_previsto_rounded = round(rendimento_previsto, 2)
    intervalo_min = round(rendimento_previsto * 0.85, 2)
    intervalo_max = round(rendimento_previsto * 1.15, 2)
    modelo = "random_forest_sintetico_v1"

    prediction_record = PredictionRecord(
        idade=input_data["idade"],
        sexo=input_data["sexo"],
        cor_raca=input_data["cor_raca"],
        anos_estudo=input_data["anos_estudo"],
        setor=input_data["setor"],
        regiao=input_data["regiao"],
        rendimento_hora_previsto=rendimento_previsto_rounded,
        intervalo_min=intervalo_min,
        intervalo_max=intervalo_max,
        modelo=modelo,
    )

    db.add(prediction_record)
    db.commit()

    return {
        "rendimento_hora_previsto": rendimento_previsto_rounded,
        "intervalo_confianca": {
            "min": intervalo_min,
            "max": intervalo_max
        },
        "features_usadas": input_data,
        "modelo": modelo
    }


@router.get("/history")
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    records = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .limit(limit)
        .all()
    )

    predictions = [
        {
            "id": record.id,
            "idade": record.idade,
            "sexo": record.sexo,
            "cor_raca": record.cor_raca,
            "anos_estudo": record.anos_estudo,
            "setor": record.setor,
            "regiao": record.regiao,
            "rendimento_hora_previsto": record.rendimento_hora_previsto,
            "intervalo_confianca": {
                "min": record.intervalo_min,
                "max": record.intervalo_max,
            },
            "modelo": record.modelo,
            "created_at": record.created_at.isoformat() if record.created_at else None,
        }
        for record in records
    ]

    return {
        "total_returned": len(predictions),
        "predictions": predictions,
    }
