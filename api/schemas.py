from pydantic import BaseModel, ConfigDict, Field


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


class PredictionInterval(BaseModel):
    min: float
    max: float


class PredictionFeatures(BaseModel):
    idade: int
    sexo: str
    cor_raca: str
    anos_estudo: int
    setor: str
    regiao: str


class PredictionOutput(BaseModel):
    rendimento_hora_previsto: float
    intervalo_confianca: PredictionInterval
    features_usadas: PredictionFeatures
    modelo: str


class HistoryPredictionItem(BaseModel):
    id: int
    idade: int
    sexo: str
    cor_raca: str
    anos_estudo: int
    setor: str
    regiao: str
    rendimento_hora_previsto: float
    intervalo_confianca: PredictionInterval
    modelo: str
    created_at: str | None


class HistoryOutput(BaseModel):
    total_returned: int
    predictions: list[HistoryPredictionItem]


class ModelInfoOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    rmse: float
    mae: float
    r2: float
    n_rows: int
    features: list[str]
    target: str


class RealModelInfoOutput(ModelInfoOutput):
    data_source: str


class PredictionIntervalResiduals(BaseModel):
    lower_residual_p05: float
    upper_residual_p95: float


class ProductionModelInfoOutput(RealModelInfoOutput):
    prediction_interval_method: str
    prediction_interval_residuals: PredictionIntervalResiduals
    promoted_from: str
    note: str


class ModelComparisonItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    rmse: float
    mae: float
    r2: float
    cv_r2_mean: float
    cv_r2_std: float


class ModelComparisonOutput(BaseModel):
    data_source: str
    n_rows: int
    target: str
    features: list[str]
    models: list[ModelComparisonItem]
    best_model_by_rmse: str
