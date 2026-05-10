# Changelog

Histórico das principais versões do projeto.

## v3.0

Release final de portfólio com API, frontend, modelo real PNAD, observabilidade, contratos OpenAPI, testes, CI/CD e deploy validado.

- API FastAPI publicada no Render.
- Frontend Streamlit publicado no Streamlit Cloud.
- Modelo XGBoost de produção treinado com microdados reais da PNAD Contínua 2023 T1.
- Parser real de arquivo PNAD em largura fixa usando layout SAS.
- Histórico de predições salvo em SQLite.
- Endpoints principais documentados em OpenAPI.
- Testes automatizados com pytest.
- Docker e GitHub Actions configurados.

## v2.9

Revisão leve de qualidade, redução de duplicação interna em metadados e fallback mais resiliente no frontend.

- Extração dos helpers internos `get_numeric_constraints()`, `get_categorical_options()` e `get_feature_definitions()`.
- `/metadata` e `/features` passaram a reutilizar os mesmos helpers.
- Fallback do frontend ficou mais robusto quando `numeric_constraints` ou `categorical_options` vêm como `None`.
- Novos testes cobrem esses cenários de fallback.

## v2.8

Consistência entre `/features`, `/metadata` e `core/config.py`, com contrato OpenAPI explícito para `/features`.

- `/features` passou a usar constantes centralizadas.
- Testes garantem alinhamento entre `/features`, `/metadata` e `core/config.py`.
- Adicionado `response_model=FeaturesOutput`.
- Criados schemas Pydantic para informações numéricas e categóricas.

## v2.7

Centralização de constantes em `core/config.py` e redução de duplicação entre API, frontend e testes.

- Centralização de `APP_NAME`, `APP_VERSION`, limites numéricos e opções categóricas.
- Backend, frontend e testes passaram a reutilizar as mesmas constantes.
- Limites e categorias usados por `/predict`, `/features`, `/metadata` e pelo frontend ficaram mais fáceis de manter.

## v2.6

Endpoint `/metadata` para integração, frontend consumindo metadados da API e testes de fallback.

- Criado `GET /metadata`.
- Exposição de `app_name`, `version`, `model_name`, `prediction_endpoint`, `numeric_constraints` e `categorical_options`.
- Frontend Streamlit passou a usar `/metadata` para configurar limites e categorias.
- Fallback local preserva o funcionamento do frontend caso `/metadata` falhe.
- Testes cobrem o endpoint, o contrato OpenAPI e os helpers do frontend.

## v2.5

Observabilidade básica com `/metrics`, health check informativo e métricas do modelo em produção.

- Criado `GET /metrics`.
- `total_predictions` vem do histórico salvo no banco.
- Falhas de banco em `/metrics` retornam `status="degraded"` e `total_predictions=0`.
- `/metrics` expõe `model_rmse`, `model_mae` e `model_r2`.
- `/health` passou a retornar informações de aplicação, modelo, banco e versão.

## v2.4

Validação robusta do input do `/predict`, documentação OpenAPI enriquecida e testes para payloads inválidos.

- `PredictionInput` valida limites e categorias aceitas.
- `idade`: 14 a 100.
- `anos_estudo`: 0 a 20.
- Categorias aceitas para `sexo`, `cor_raca`, `setor` e `regiao`.
- Campos do input receberam descrições e exemplos.
- Testes cobrem payload válido, campos ausentes, tipos inválidos, categorias inválidas e limites inválidos.

## v2.3

Contratos explícitos da API com Pydantic, documentação OpenAPI mais precisa e testes de contrato.

- `POST /predict` formalizado com `PredictionOutput`.
- `intervalo_confianca` e `features_usadas` tipados.
- `response_model` adicionado a endpoints de informações do modelo, comparação e histórico.
- Testes verificam contratos principais no OpenAPI.

## v2.2

Robustez em produção, rate limiting e feature importance do modelo de produção.

- Tratamento controlado de falhas no modelo e na predição.
- Falhas ao salvar histórico no banco não impedem a resposta da predição.
- Rollback em caso de erro de banco.
- Rate limiting no `POST /predict`: 30 requisições por minuto por IP.
- Excesso retorna HTTP 429.
- Criado `GET /feature-importance/production`.
- Frontend passou a usar feature importance do modelo de produção.

## v2.0 e v2.1

Evolução do pipeline real e consolidação do modelo de produção.

- Integração com microdados reais da PNAD Contínua 2023 T1.
- Parser de arquivo de largura fixa usando layout SAS.
- Treinamento e comparação entre Regressão Linear, Random Forest e XGBoost.
- Promoção do XGBoost real para modelo de produção.
- Exposição de métricas e informações do modelo real.
