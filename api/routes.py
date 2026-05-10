from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.config import (
    ANOS_ESTUDO_MAX,
    ANOS_ESTUDO_MIN,
    APP_NAME,
    APP_VERSION,
    COR_RACA_OPTIONS,
    IDADE_MAX,
    IDADE_MIN,
    REGIAO_OPTIONS,
    SETOR_OPTIONS,
    SEXO_OPTIONS,
)
from api.schemas import (
    HealthOutput,
    HistoryOutput,
    MetadataOutput,
    MetricsOutput,
    ModelComparisonOutput,
    ModelInfoOutput,
    PredictionInput,
    PredictionOutput,
    ProductionModelInfoOutput,
    RealModelInfoOutput,
)
from api.rate_limit import enforce_predict_rate_limit
from database.database import check_database_connection, get_db
from database.models import PredictionRecord
from ml.feature_importance import (
    get_production_pnad_feature_importance,
    get_production_feature_importance,
    get_real_feature_importance,
)
from ml.model_info import (
    load_model_comparison,
    load_model_metrics,
    load_production_model_metrics,
    load_real_model_metrics,
)
from ml.predict import (
    MODEL_NAME,
    PredictionModelError,
    build_prediction_interval,
    predict_rendimento,
)

router = APIRouter()


@router.get("/")
def root():
    return {
        "message": APP_NAME,
        "status": "running",
        "docs": "/docs"
    }


@router.get("/health", response_model=HealthOutput)
def health_check():
    return {
        "app_name": APP_NAME,
        "status": "ok",
        "model_loaded": True,
        "database_connected": check_database_connection(),
        "model_name": MODEL_NAME,
        "version": APP_VERSION,
    }


@router.get("/metrics", response_model=MetricsOutput)
def get_metrics(db: Session = Depends(get_db)):
    status = "ok"
    production_metrics = load_production_model_metrics()

    try:
        total_predictions = db.query(PredictionRecord).count()
    except SQLAlchemyError:
        db.rollback()
        status = "degraded"
        total_predictions = 0

    return {
        "app_name": APP_NAME,
        "status": status,
        "model_name": MODEL_NAME,
        "total_predictions": total_predictions,
        "model_rmse": production_metrics["rmse"],
        "model_mae": production_metrics["mae"],
        "model_r2": production_metrics["r2"],
    }


@router.get("/metadata", response_model=MetadataOutput)
def get_metadata():
    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "model_name": MODEL_NAME,
        "prediction_endpoint": "/predict",
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


@router.get("/features")
def get_features():
    return {
        "target": "rendimento_hora",
        "features": {
            "idade": {
                "type": "int",
                "min": IDADE_MIN,
                "max": IDADE_MAX
            },
            "sexo": {
                "type": "category",
                "values": SEXO_OPTIONS
            },
            "cor_raca": {
                "type": "category",
                "values": COR_RACA_OPTIONS
            },
            "anos_estudo": {
                "type": "int",
                "min": ANOS_ESTUDO_MIN,
                "max": ANOS_ESTUDO_MAX
            },
            "setor": {
                "type": "category",
                "values": SETOR_OPTIONS
            },
            "regiao": {
                "type": "category",
                "values": REGIAO_OPTIONS
            }
        }
    }


@router.get("/model-info", response_model=ModelInfoOutput)
def get_model_info():
    return load_model_metrics()


@router.get("/model-info/real", response_model=RealModelInfoOutput)
def get_real_model_info():
    return load_real_model_metrics()


@router.get("/model-info/production", response_model=ProductionModelInfoOutput)
def get_production_model_info():
    return load_production_model_metrics()


@router.get("/model-comparison", response_model=ModelComparisonOutput)
def get_model_comparison():
    return load_model_comparison()


@router.get("/feature-importance")
def get_feature_importance():
    return get_production_feature_importance()


@router.get("/feature-importance/real")
def get_real_feature_importance_endpoint():
    return get_real_feature_importance()


@router.get("/feature-importance/production")
def get_production_pnad_feature_importance_endpoint():
    return get_production_pnad_feature_importance()


@router.post(
    "/predict",
    response_model=PredictionOutput,
    dependencies=[Depends(enforce_predict_rate_limit)],
)
def predict(data: PredictionInput, db: Session = Depends(get_db)):
    input_data = data.model_dump()

    try:
        rendimento_previsto = predict_rendimento(input_data)
        rendimento_previsto_rounded = round(rendimento_previsto, 2)
        intervalo = build_prediction_interval(rendimento_previsto)
    except PredictionModelError as exc:
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao gerar predição.",
        ) from exc

    intervalo_min = intervalo["min"]
    intervalo_max = intervalo["max"]
    modelo = MODEL_NAME

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

    try:
        db.add(prediction_record)
        db.commit()
    except SQLAlchemyError:
        db.rollback()

    return {
        "rendimento_hora_previsto": rendimento_previsto_rounded,
        "intervalo_confianca": intervalo,
        "features_usadas": input_data,
        "modelo": modelo
    }


@router.get("/history", response_model=HistoryOutput)
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
