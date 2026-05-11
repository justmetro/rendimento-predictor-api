from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.config import (
    ANOS_ESTUDO_MAX,
    ANOS_ESTUDO_MIN,
    IDADE_MAX,
    IDADE_MIN,
)


class PredictionInput(BaseModel):
    idade: int = Field(
        ...,
        ge=IDADE_MIN,
        le=IDADE_MAX,
        description="Idade da pessoa em anos completos.",
        examples=[35],
    )
    sexo: Literal["M", "F"] = Field(
        ...,
        description="Sexo informado no formato categórico aceito pelo modelo.",
        examples=["M"],
    )
    cor_raca: Literal["Branca", "Preta", "Parda", "Amarela", "Indigena"] = Field(
        ...,
        description="Categoria de cor ou raça aceita pelo modelo.",
        examples=["Branca"],
    )
    anos_estudo: int = Field(
        ...,
        ge=ANOS_ESTUDO_MIN,
        le=ANOS_ESTUDO_MAX,
        description="Quantidade de anos de estudo concluídos.",
        examples=[12],
    )
    setor: Literal["Servicos", "Industria", "Comercio", "Agricultura", "Construcao"] = Field(
        ...,
        description="Setor de atividade profissional aceito pelo modelo.",
        examples=["Servicos"],
    )
    regiao: Literal["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"] = Field(
        ...,
        description="Região do Brasil associada ao registro.",
        examples=["Sudeste"],
    )

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


class HealthOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    app_name: str
    status: str
    model_loaded: bool
    database_connected: bool
    model_name: str
    version: str


class FeatureNumericInfo(BaseModel):
    type: Literal["int"]
    min: int
    max: int


class FeatureCategoricalInfo(BaseModel):
    type: Literal["category"]
    values: list[str]


class FeaturesInfo(BaseModel):
    idade: FeatureNumericInfo
    sexo: FeatureCategoricalInfo
    cor_raca: FeatureCategoricalInfo
    anos_estudo: FeatureNumericInfo
    setor: FeatureCategoricalInfo
    regiao: FeatureCategoricalInfo


class FeaturesOutput(BaseModel):
    target: str
    features: FeaturesInfo


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
    faixa_etaria: str
    anos_estudo: int
    escolaridade: str
    setor: str
    regiao: str
    rendimento_hora_previsto: float
    intervalo_confianca: PredictionInterval
    modelo: str
    created_at: str | None


class HistoryOutput(BaseModel):
    total_returned: int
    predictions: list[HistoryPredictionItem]


class MetricsOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    app_name: str
    status: str
    model_name: str
    total_predictions: int
    model_rmse: float
    model_mae: float
    model_r2: float


class NumericConstraint(BaseModel):
    min: int
    max: int


class MetadataOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    app_name: str
    version: str
    model_name: str
    prediction_endpoint: str
    numeric_constraints: dict[str, NumericConstraint]
    categorical_options: dict[str, list[str]]


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
