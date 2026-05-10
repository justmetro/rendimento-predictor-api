# Desenvolvimento

## Estrutura do projeto

```txt
api/                 Rotas, schemas e rate limiting
core/                Constantes centralizadas
database/            Configuração SQLAlchemy, init_db e modelos ORM
frontend/            Aplicação Streamlit e helpers de metadata
ml/                  Predição, treino, comparação, métricas e feature importance
scripts/             Download, extração, parser e EDA da PNAD
tests/               Testes automatizados
docs/                Documentação técnica
```

## Configuração centralizada

As principais constantes ficam em:

```txt
core/config.py
```

Esse módulo centraliza:

- `APP_NAME`
- `APP_VERSION`
- limites de `idade`
- limites de `anos_estudo`
- opções de `sexo`
- opções de `cor_raca`
- opções de `setor`
- opções de `regiao`

As constantes são reutilizadas por API, schemas, frontend e testes.

## SQLite e histórico

O projeto usa SQLite local para registrar predições realizadas pela API.

- Banco local: `data/predictions.db`
- Banco de testes: `data/test_predictions.db`
- Endpoint de consulta: `GET /history`

A tabela é criada automaticamente ao iniciar a API ou manualmente com:

```bash
python -m database.init_db
```

Cada chamada válida para `POST /predict` salva uma predição. Se houver erro ao salvar o histórico, a transação é revertida e a API ainda retorna a predição calculada.

## Observabilidade

Endpoints principais:

```http
GET /health
GET /metrics
```

`/health` retorna informações de status, banco, modelo e versão.

`/metrics` retorna contagem de predições salvas e métricas do modelo de produção. Em falha de banco, responde de forma controlada com `status="degraded"` e `total_predictions=0`.

## Metadata e fallback do frontend

O frontend Streamlit consome:

```http
GET /metadata
```

Esse endpoint fornece limites e categorias usados no formulário de predição. Se `/metadata` falhar ou vier parcialmente indisponível, o frontend usa fallback local com os mesmos valores esperados pelo backend.

Os helpers ficam em:

```txt
frontend/metadata.py
```

## Validação e rate limiting

`POST /predict` valida:

- `idade`: 14 a 100
- `anos_estudo`: 0 a 20
- `sexo`: categorias suportadas
- `cor_raca`: categorias suportadas
- `setor`: categorias suportadas
- `regiao`: categorias suportadas

Entradas inválidas retornam HTTP 422.

O endpoint possui rate limiting simples em memória:

```txt
30 requisições por minuto por IP
```

Excesso retorna HTTP 429.

## Docker

Construir a imagem:

```bash
docker build -t rendimento-predictor-api .
```

Executar o container:

```bash
docker run -p 8000:8000 rendimento-predictor-api
```

A documentação local fica disponível em:

```txt
http://127.0.0.1:8000/docs
```

## GitHub Actions

O projeto usa GitHub Actions para rodar testes automaticamente a cada push na branch principal.

O workflow valida instalação de dependências e execução da suíte com pytest.

## Testes

Rodar a suíte completa:

```bash
pytest
```

Rodar com cobertura:

```bash
pytest --cov=.
```

A suíte cobre API, contratos OpenAPI, validação de entrada, histórico, metadata, fallback do frontend, parser PNAD e treinamento.

## Deploy

Backend/API:

```txt
https://rendimento-predictor-api.onrender.com
```

Frontend Streamlit:

```txt
https://rendimento-predictor-api.streamlit.app/
```

No Streamlit Cloud, o frontend deve apontar para a API publicada:

```toml
API_URL = "https://rendimento-predictor-api.onrender.com"
APP_ENV = "production"
```
