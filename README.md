# Rendimento Predictor API

API e frontend para previsão de rendimento por hora com base em dados sociodemográficos, usando microdados reais da PNAD Contínua 2023 T1 e modelo XGBoost.

## Links

- Frontend Streamlit Cloud: https://rendimento-predictor-api.streamlit.app/
- Backend Render: https://rendimento-predictor-api.onrender.com
- Swagger/OpenAPI: https://rendimento-predictor-api.onrender.com/docs
- Health check: https://rendimento-predictor-api.onrender.com/health

## Visão geral

O projeto estima rendimento por hora a partir de idade, sexo, cor/raça, anos de estudo, setor e região. A solução combina uma API FastAPI, um frontend Streamlit e um modelo XGBoost treinado com microdados reais da PNAD Contínua 2023 trimestre 1.

A release `v3.0` consolida o projeto como versão final e estável de portfólio, com deploy validado no Render e no Streamlit Cloud, contratos OpenAPI/Pydantic, validação robusta, observabilidade, histórico de predições e testes automatizados.

## Demonstração funcional

1. O usuário informa idade, sexo, cor/raça, anos de estudo, setor e região.
2. A API valida a entrada com Pydantic.
3. O modelo de produção estima o rendimento por hora.
4. A aplicação retorna previsão, intervalo empírico, modelo usado e histórico salvo.

## Destaques técnicos

- API REST com FastAPI
- Frontend interativo com Streamlit
- XGBoost treinado com PNAD Contínua real
- Parser de arquivo PNAD em largura fixa usando layout SAS
- SQLite para histórico de predições
- Docker e GitHub Actions
- Deploy no Render e Streamlit Cloud
- Testes automatizados com pytest
- Contratos Pydantic/OpenAPI explícitos
- Validação robusta do `POST /predict`
- Rate limiting no `POST /predict`
- Observabilidade com `/health` e `/metrics`
- Integração frontend/API via `/metadata` com fallback local

## Stack

| Camada | Tecnologias |
| --- | --- |
| API | Python, FastAPI, Pydantic |
| Frontend | Streamlit |
| Machine Learning | scikit-learn, XGBoost, Pandas, NumPy |
| Banco | SQLAlchemy, SQLite |
| Testes | pytest, pytest-cov |
| Deploy/DevOps | Docker, GitHub Actions, Render, Streamlit Cloud |

## Modelo em produção

- Dataset: PNAD Contínua 2023 T1
- Linhas lidas pelo parser: 473.335
- Linhas finais após limpeza: 175.132
- Modelo: `xgboost_pnad_real_production_v1`
- RMSE: 5.86
- MAE: 4.34
- R²: 0.333
- Intervalo de predição: percentis 5 e 95 dos resíduos do conjunto de teste

## Principais endpoints

- `GET /health`
- `GET /metrics`
- `GET /metadata`
- `GET /features`
- `POST /predict`
- `GET /history`
- `GET /model-info/production`
- `GET /feature-importance/production`

Documentação completa dos endpoints: [docs/API.md](docs/API.md).

## Como rodar localmente

Crie e ative o ambiente virtual:

```bash
python -m venv venv
venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Rode a API:

```bash
uvicorn main:app --reload
```

Rode o frontend:

```bash
streamlit run frontend/app_streamlit.py
```

Rode os testes:

```bash
pytest
```

## Testes

A release `v3.0` conta com 73 testes automatizados cobrindo API, contratos OpenAPI, validação de entrada, rate limiting, histórico, metadata, fallback do frontend, parser PNAD e treinamento do modelo de produção.

## Documentação adicional

- [docs/API.md](docs/API.md): endpoints, contratos e exemplos
- [docs/EXAMPLES.md](docs/EXAMPLES.md): exemplos simples em cURL, Python e JavaScript
- [docs/MODEL.md](docs/MODEL.md): dados, parser, treinamento, métricas e modelo
- [docs/CHANGELOG.md](docs/CHANGELOG.md): histórico de versões
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): desenvolvimento, testes, Docker, CI/CD e manutenção

## Status

Projeto finalizado em `v3.0` como versão estável de portfólio.
