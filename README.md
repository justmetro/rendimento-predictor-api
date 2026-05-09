# Rendimento Predictor API

API REST desenvolvida com FastAPI para predizer rendimento por hora com base em características sociodemográficas e profissionais.

Este projeto combina Machine Learning, backend, testes automatizados, EDA, visualização de dados, Docker, CI/CD, deploy de API e frontend, simulando um fluxo completo de desenvolvimento de uma aplicação preditiva.

## Aplicação Web

Frontend publicado no Streamlit Cloud:

```txt
https://rendimento-predictor-api.streamlit.app/
```

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

## Objetivo

Construir uma aplicação capaz de receber dados como idade, sexo, cor/raça, anos de estudo, setor e região, e retornar uma estimativa de rendimento por hora.

## Tecnologias utilizadas

- Python
- FastAPI
- Streamlit
- Scikit-learn
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

## Modelo atual

O modelo atual em produção é um baseline treinado com dados sintéticos, usado para validar o fluxo completo:

```txt
dados → EDA → treino → modelo salvo → API → frontend → testes → CI/CD → deploy
```

Métricas atuais do modelo em produção:

```txt
RMSE: 6.42
MAE: 4.99
R²: 0.861
```

## Pipeline de dados reais

A partir da versão 1.1, o projeto também possui um pipeline separado para dados reais padronizados.

Arquivos principais:

```txt
scripts/prepare_real_data.py
scripts/eda_real.py
ml/train_real.py
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
```

O objetivo é separar claramente:

```txt
modelo em produção → usado pela API em /predict
modelo candidato → treinado pelo pipeline real e exposto em /model-info/real
```

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

Para rodar com cobertura:

```bash
pytest --cov=.
```

## CI/CD

O projeto usa GitHub Actions para rodar os testes automaticamente a cada push na branch `main`.

## Deploy

- Backend/API: Render
- Frontend: Streamlit Cloud

## Próximos passos

- Substituir o arquivo padronizado de exemplo por dados públicos reais do IBGE/PNAD
- Adicionar análise exploratória dos dados reais completos
- Melhorar feature engineering
- Comparar modelos: Regressão Linear, Random Forest e XGBoost
- Promover o melhor modelo candidato para produção
- Adicionar banco de dados para salvar predições
- Melhorar a interface web