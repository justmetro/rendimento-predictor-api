# API

Base URL de produção:

```txt
https://rendimento-predictor-api.onrender.com
```

Documentação interativa:

```txt
https://rendimento-predictor-api.onrender.com/docs
```

## GET /health

Retorna um health check informativo da aplicação.

Campos principais:

- `app_name`
- `status`
- `model_loaded`
- `database_connected`
- `model_name`
- `version`

## GET /metrics

Retorna métricas básicas de observabilidade.

Campos principais:

- `app_name`
- `status`
- `model_name`
- `total_predictions`
- `model_rmse`
- `model_mae`
- `model_r2`

`total_predictions` vem do banco de histórico. Se houver falha ao acessar o banco, a API retorna `status="degraded"`, `total_predictions=0` e faz rollback.

## GET /metadata

Retorna metadados para integração com clientes externos e frontend.

Campos principais:

- `app_name`
- `version`
- `model_name`
- `prediction_endpoint`
- `numeric_constraints`
- `categorical_options`

Exemplo parcial:

```json
{
  "prediction_endpoint": "/predict",
  "numeric_constraints": {
    "idade": {
      "min": 14,
      "max": 100
    },
    "anos_estudo": {
      "min": 0,
      "max": 20
    }
  },
  "categorical_options": {
    "sexo": ["M", "F"],
    "cor_raca": ["Branca", "Preta", "Parda", "Amarela", "Indigena"],
    "setor": ["Servicos", "Industria", "Comercio", "Agricultura", "Construcao"],
    "regiao": ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
  }
}
```

## GET /features

Retorna as variáveis aceitas pelo modelo no formato histórico do endpoint.

Campos principais:

- `idade`
- `sexo`
- `cor_raca`
- `anos_estudo`
- `setor`
- `regiao`

Os limites e categorias são alinhados com `core/config.py` e com o endpoint `/metadata`.

## POST /predict

Recebe dados sociodemográficos e retorna a previsão de rendimento por hora.

Exemplo de entrada:

```json
{
  "idade": 35,
  "sexo": "M",
  "cor_raca": "Branca",
  "anos_estudo": 12,
  "setor": "Servicos",
  "regiao": "Sudeste"
}
```

Exemplo de saída:

```json
{
  "rendimento_hora_previsto": 50.48,
  "intervalo_confianca": {
    "min": 42.9,
    "max": 58.05
  },
  "features_usadas": {
    "idade": 35,
    "sexo": "M",
    "cor_raca": "Branca",
    "anos_estudo": 12,
    "setor": "Servicos",
    "regiao": "Sudeste"
  },
  "modelo": "xgboost_pnad_real_production_v1"
}
```

Validações principais:

- `idade`: 14 a 100.
- `anos_estudo`: 0 a 20.
- Campos categóricos aceitam apenas categorias suportadas pelo modelo.
- Entradas inválidas retornam HTTP 422.
- O endpoint possui rate limiting de 30 requisições por minuto por IP.
- Excesso de requisições retorna HTTP 429.

## GET /history

Retorna as últimas predições salvas no banco SQLite.

Parâmetro opcional:

- `limit`: quantidade máxima de registros retornados. Valor padrão: 10.

Exemplo:

```http
GET /history?limit=10
```

Exemplo de saída:

```json
{
  "total_returned": 1,
  "predictions": [
    {
      "id": 1,
      "idade": 35,
      "sexo": "M",
      "cor_raca": "Branca",
      "anos_estudo": 12,
      "setor": "Servicos",
      "regiao": "Sudeste",
      "rendimento_hora_previsto": 50.48,
      "intervalo_confianca": {
        "min": 42.9,
        "max": 58.05
      },
      "modelo": "xgboost_pnad_real_production_v1",
      "created_at": "2026-05-09T12:00:00"
    }
  ]
}
```

## GET /model-info

Retorna métricas e informações do modelo legado sintético, mantido como baseline histórico.

## GET /model-info/real

Retorna métricas e informações do modelo candidato treinado pelo pipeline real.

## GET /model-info/production

Retorna métricas e informações do modelo XGBoost de produção treinado com microdados reais da PNAD Contínua 2023 T1.

Campos principais:

- `model_name`
- `rmse`
- `mae`
- `r2`
- `algorithm`
- `dataset`

## GET /model-comparison

Retorna a comparação entre modelos treinados no pipeline real.

Modelos comparados:

- Regressão Linear
- Random Forest
- XGBoost

A comparação inclui RMSE, MAE, R², média de R² em validação cruzada e desvio padrão da validação cruzada.

## GET /feature-importance

Retorna a importância das variáveis do modelo legado.

## GET /feature-importance/real

Retorna a importância das variáveis do modelo candidato do pipeline real.

## GET /feature-importance/production

Retorna a importância das variáveis do modelo de produção `xgboost_pnad_real_production_v1`.

Esse endpoint é usado pelo frontend para mostrar a importância de variáveis do mesmo modelo utilizado em `POST /predict`.
