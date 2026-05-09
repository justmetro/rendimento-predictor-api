# Rendimento Predictor API

API REST desenvolvida com FastAPI para predizer rendimento por hora com base em características sociodemográficas e profissionais.

Este projeto combina Machine Learning, backend, testes automatizados, EDA, visualização de dados, Docker, CI/CD, deploy de API e frontend, simulando um fluxo completo de desenvolvimento de uma aplicação preditiva.

## Aplicação Web

Frontend publicado no Streamlit Cloud:

```txt
https://rendimento-predictor-api.streamlit.app/
```

A interface permite:

- preencher dados para predição;
- consultar o modelo em produção;
- consultar o modelo candidato do pipeline real;
- visualizar métricas dos modelos;
- visualizar importância das variáveis;
- comparar Regressão Linear, Random Forest e XGBoost;
- visualizar histórico de predições;
- visualizar informações gerais do projeto.

## Deploy da API

A API está disponível publicamente em:

```txt
https://rendimento-predictor-api.onrender.com/
```

Documentação interativa Swagger:

```txt
https://rendimento-predictor-api.onrender.com/docs
```

Health check:

```txt
https://rendimento-predictor-api.onrender.com/health
```

Informações do modelo em produção:

```txt
https://rendimento-predictor-api.onrender.com/model-info
```

Informações do modelo candidato do pipeline real:

```txt
https://rendimento-predictor-api.onrender.com/model-info/real
```

Comparação de modelos:

```txt
https://rendimento-predictor-api.onrender.com/model-comparison
```

Importância das variáveis do modelo em produção:

```txt
https://rendimento-predictor-api.onrender.com/feature-importance
```

Importância das variáveis do modelo candidato:

```txt
https://rendimento-predictor-api.onrender.com/feature-importance/real
```

Histórico de predições:

```txt
https://rendimento-predictor-api.onrender.com/history
```

## Objetivo

Construir uma aplicação capaz de receber dados como idade, sexo, cor/raça, anos de estudo, setor e região, e retornar uma estimativa de rendimento por hora.

## Tecnologias utilizadas

- Python
- FastAPI
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Joblib
- Pytest
- GitHub Actions
- Docker
- Render
- Streamlit Cloud

## Endpoints

### Health check

```http
GET /health
```

Retorna o status da API.

### Features

```http
GET /features
```

Retorna as variáveis aceitas pelo modelo.

### Informações do modelo em produção

```http
GET /model-info
```

Retorna métricas e informações do modelo atualmente usado no endpoint `/predict`.

### Informações do modelo candidato

```http
GET /model-info/real
```

Retorna métricas e informações do modelo treinado pelo pipeline de dados reais padronizados.

Atualmente, esse modelo candidato ainda utiliza um arquivo no formato real padronizado para validar o pipeline. A próxima etapa é substituir esse arquivo por dados reais de fato da PNAD/IBGE.

### Comparação de modelos

```http
GET /model-comparison
```

Retorna a comparação entre os modelos treinados no pipeline:

- Regressão Linear
- Random Forest
- XGBoost

A comparação inclui RMSE, MAE, R², média de R² em validação cruzada e desvio padrão da validação cruzada.

### Importância das variáveis do modelo em produção

```http
GET /feature-importance
```

Retorna as variáveis mais importantes para o modelo atualmente usado pela API.

### Importância das variáveis do modelo candidato

```http
GET /feature-importance/real
```

Retorna as variáveis mais importantes para o modelo candidato treinado pelo pipeline real.

### Predição

```http
POST /predict
```

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
  "modelo": "random_forest_sintetico_v1"
}
```

### Histórico de predições

```http
GET /history
```

Retorna as últimas predições salvas no banco SQLite. O endpoint aceita o parâmetro opcional `limit`, com valor padrão `10`.

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
      "modelo": "random_forest_sintetico_v1",
      "created_at": "2026-05-09T12:00:00"
    }
  ]
}
```

## Banco de dados e histórico

O projeto usa SQLite local para registrar as predições realizadas pela API.

- Arquivo local do banco: `data/predictions.db`
- Esse arquivo é ignorado pelo Git.
- Cada chamada para `POST /predict` salva uma nova predição.
- O endpoint `GET /history` consulta as últimas predições salvas.
- Não há Alembic nesta etapa; a tabela é criada automaticamente ao iniciar a API ou manualmente com `python -m database.init_db`.

## Modelo atual

O modelo atual em produção é um baseline treinado com dados sintéticos, usado para validar o fluxo completo:

```txt
dados → EDA → treino → comparação de modelos → modelo salvo → API → frontend → testes → CI/CD → deploy
```

Métricas atuais do modelo em produção:

```txt
RMSE: 6.42
MAE: 4.99
R²: 0.861
```

## Comparação de modelos

O projeto compara três abordagens de modelagem:

```txt
Regressão Linear
Random Forest
XGBoost
```

Resultado atual da comparação:

```txt
Melhor modelo por RMSE: linear_regression

linear_regression → RMSE 5.76 | MAE 4.57 | R² 0.888
random_forest     → RMSE 6.42 | MAE 4.99 | R² 0.861
xgboost           → RMSE 6.51 | MAE 5.14 | R² 0.857
```

Como o dataset atual ainda é sintético e foi gerado com uma relação aproximadamente linear, a Regressão Linear apresenta melhor desempenho nesta etapa.

A comparação é gerada por:

```bash
python -m ml.compare_models
```

E salva em:

```txt
data/models/model_comparison.json
```

## Pipeline de dados reais

A partir da versão 1.1, o projeto também possui um pipeline separado para dados reais padronizados.

Arquivos principais:

```txt
scripts/prepare_real_data.py
scripts/eda_real.py
ml/train_real.py
ml/compare_models.py
```

Esse pipeline lê um CSV em:

```txt
data/raw/pnad_real.csv
```

E gera:

```txt
data/processed/pnad_real_processed.csv
data/processed/real_eda_summary.txt
data/processed/real_plots/
data/models/rendimento_model_real.pkl
data/models/metrics_real.json
data/models/model_comparison.json
```

O objetivo é separar claramente:

```txt
modelo em produção → usado pela API em /predict
modelo candidato → treinado pelo pipeline real e exposto em /model-info/real
comparação → modelos avaliados e expostos em /model-comparison
```

## Importância das variáveis

O projeto expõe a importância das variáveis dos modelos Random Forest.

Para o modelo em produção:

```bash
GET /feature-importance
```

Para o modelo candidato:

```bash
GET /feature-importance/real
```

Esses endpoints ajudam a entender quais variáveis mais influenciam a predição, como idade, anos de estudo, região, setor, sexo e cor/raça.

## Dados sintéticos e EDA

Antes da integração com dados reais do IBGE/PNAD, o projeto utiliza um dataset sintético para validar o fluxo completo de Machine Learning e API.

O dataset sintético é gerado por:

```bash
python -m scripts.generate_data
```

Esse comando cria o arquivo:

```txt
data/processed/synthetic_rendimento.csv
```

A análise exploratória inicial pode ser gerada com:

```bash
python -m scripts.eda_synthetic
```

Esse comando cria o relatório:

```txt
data/processed/eda_summary.txt
```

A EDA inclui:

- shape do dataset
- tipos das variáveis
- valores ausentes
- estatísticas descritivas
- distribuição por sexo
- distribuição por região
- média de rendimento por região
- média de rendimento por setor
- gráficos exploratórios em PNG

## Visualizações da EDA

### Distribuição do rendimento por hora

![Distribuição do rendimento por hora](data/processed/plots/target_distribution.png)

### Média de rendimento por região

![Média de rendimento por região](data/processed/plots/rendimento_by_regiao.png)

### Média de rendimento por setor

![Média de rendimento por setor](data/processed/plots/rendimento_by_setor.png)

### Idade vs rendimento por hora

![Idade vs rendimento por hora](data/processed/plots/idade_vs_rendimento.png)

## Como rodar localmente

Clone o repositório:

```bash
git clone https://github.com/justmetro/rendimento-predictor-api.git
cd rendimento-predictor-api
```

Crie e ative o ambiente virtual:

```bash
python -m venv venv
venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Gere os dados sintéticos:

```bash
python -m scripts.generate_data
```

Gere a EDA inicial:

```bash
python -m scripts.eda_synthetic
```

Treine o modelo sintético:

```bash
python ml/train.py
```

Prepare os dados reais padronizados:

```bash
python -m scripts.prepare_real_data
```

Gere a EDA dos dados reais padronizados:

```bash
python -m scripts.eda_real
```

Treine o modelo candidato do pipeline real:

```bash
python -m ml.train_real
```

Compare os modelos:

```bash
python -m ml.compare_models
```

Inicialize o banco SQLite, se quiser criar a tabela manualmente:

```bash
python -m database.init_db
```

O banco também é criado automaticamente ao iniciar a API.

Rode a API:

```bash
uvicorn main:app --reload
```

Acesse a documentação automática:

```txt
http://127.0.0.1:8000/docs
```

## Como rodar o frontend localmente

Rode:

```bash
streamlit run frontend/app_streamlit.py
```

Ou:

```bash
python -m streamlit run frontend/app_streamlit.py
```

Acesse:

```txt
http://localhost:8501
```

Por padrão, o frontend local usa a API em:

```txt
http://localhost:8000
```

Em produção, configure a URL da API por variável de ambiente ou secret do Streamlit Cloud:

```txt
API_URL = "https://rendimento-predictor-api.onrender.com"
```

## Como rodar com Docker

Construa a imagem:

```bash
docker build -t rendimento-predictor-api .
```

Rode o container:

```bash
docker run -p 8000:8000 rendimento-predictor-api
```

Acesse:

```txt
http://127.0.0.1:8000/docs
```

## Testes

Para rodar os testes:

```bash
pytest
```

Os testes usam um banco SQLite isolado em `data/test_predictions.db`. Esse arquivo é ignorado pelo Git.

Para rodar com cobertura:

```bash
pytest --cov=.
```

## CI/CD

O projeto usa GitHub Actions para rodar os testes automaticamente a cada push na branch `main`.

## Deploy

- Backend/API: Render
- Frontend: Streamlit Cloud

No Streamlit Cloud, configure os secrets para apontar o frontend para a API publicada no Render:

```toml
API_URL = "https://rendimento-predictor-api.onrender.com"
APP_ENV = "production"
```

## Próximos passos

- Substituir o arquivo padronizado de exemplo por dados públicos reais do IBGE/PNAD
- Adicionar análise exploratória dos dados reais completos
- Melhorar feature engineering
- Promover o melhor modelo candidato para produção
- Adicionar banco de dados para salvar predições
- Melhorar a interface web
