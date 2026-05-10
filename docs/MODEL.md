# Modelo e Dados

## Fonte dos dados

O modelo de produção usa microdados reais da PNAD Contínua 2023 T1.

O pipeline parte do arquivo oficial de largura fixa e do layout SAS para extrair as variáveis usadas no treinamento.

## Parser PNAD

O parser real lê o arquivo de largura fixa usando as posições definidas no layout SAS.

Arquivos principais do pipeline:

```txt
scripts/download_pnad.py
scripts/extract_pnad_zip.py
scripts/inspect_pnad_input.py
scripts/parse_pnad_real.py
scripts/prepare_real_data.py
scripts/eda_real.py
ml/train_real.py
ml/compare_models.py
ml/train_production.py
```

Entradas esperadas:

```txt
data/raw/pnad/input_PNADC_trimestre1.txt
data/raw/pnad/extracted/PNADC_2023_trimestre1/PNADC_2023_trimestre1.txt
```

Artefatos gerados:

```txt
data/processed/pnad_real_processed.csv
data/processed/real_eda_summary.txt
data/processed/real_plots/
data/models/rendimento_model_real.pkl
data/models/metrics_real.json
data/models/model_comparison.json
data/models/rendimento_model_production.pkl
data/models/metrics_production.json
```

## Modelo em produção

```txt
Modelo: xgboost_pnad_real_production_v1
Algoritmo: XGBoost
Dados: PNAD Contínua 2023 T1
Linhas lidas pelo parser: 473.335
Linhas finais após limpeza: 175.132
RMSE: 5.86
MAE: 4.34
R²: 0.333
Método de intervalo: residual_percentile_5_95
```

Esse é o modelo carregado pelo endpoint `POST /predict`.

## Intervalo de predição

O campo `intervalo_confianca` mantém o formato da API e é estimado com os percentis 5 e 95 dos resíduos observados no conjunto de teste do modelo de produção.

Esse intervalo é empírico e baseado em resíduos. Ele não é uma abordagem de quantile regression nem bootstrap.

## Comparação de modelos

O projeto compara três abordagens sobre o dataset real processado da PNAD:

- Regressão Linear
- Random Forest
- XGBoost

Resultado atual:

```txt
Melhor modelo por RMSE: xgboost

linear_regression -> RMSE 6.16 | MAE 4.64 | R² 0.263
random_forest     -> RMSE 5.90 | MAE 4.37 | R² 0.325
xgboost           -> RMSE 5.86 | MAE 4.34 | R² 0.333
```

A comparação é gerada por:

```bash
python -m ml.compare_models
```

E salva em:

```txt
data/models/model_comparison.json
```

## Feature importance

O projeto expõe a importância das variáveis dos modelos legado, candidato e de produção.

Endpoints:

```http
GET /feature-importance
GET /feature-importance/real
GET /feature-importance/production
```

O frontend Streamlit usa a feature importance do modelo de produção para refletir o mesmo modelo usado por `POST /predict`.

## Modelo legado e dados sintéticos

Antes da integração com microdados reais da PNAD, o projeto usava um dataset sintético para validar o fluxo completo de machine learning e API.

O modelo legado é mantido como baseline histórico e referência de evolução do projeto.

Métricas do modelo legado:

```txt
RMSE: 6.42
MAE: 4.99
R²: 0.861
```

Dados sintéticos podem ser gerados por:

```bash
python -m scripts.generate_data
```

A EDA sintética pode ser gerada por:

```bash
python -m scripts.eda_synthetic
```
