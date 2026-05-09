# Rendimento Predictor API

API REST desenvolvida com FastAPI para predizer rendimento por hora com base em características sociodemográficas e profissionais.

Este projeto combina Machine Learning, backend, testes automatizados e CI/CD, simulando um fluxo completo de desenvolvimento de uma API preditiva.

## Objetivo

Construir uma API capaz de receber dados como idade, sexo, cor/raça, anos de estudo, setor e região, e retornar uma estimativa de rendimento por hora.

## Tecnologias utilizadas

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Pytest
- GitHub Actions

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

### Informações do modelo

```http
GET /model-info
```

Retorna métricas e informações do modelo treinado.

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

O modelo atual é um baseline treinado com dados sintéticos, usado para validar o fluxo completo:

```txt
dados → treino → modelo salvo → API → predição → testes → CI/CD
```

Métricas atuais:

```txt
RMSE: 6.42
MAE: 4.99
R²: 0.861
```

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

Treine o modelo:

```bash
python ml/train.py
```

Rode a API:

```bash
uvicorn main:app --reload
```

Acesse a documentação automática:

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

## Próximos passos

- Substituir dados sintéticos por dados públicos reais do IBGE/PNAD
- Adicionar análise exploratória dos dados
- Melhorar feature engineering
- Comparar modelos: Regressão Linear, Random Forest e XGBoost
- Adicionar banco de dados para salvar predições
- Fazer deploy da APIa