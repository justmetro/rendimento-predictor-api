# Exemplos de uso

Base URL de produção:

```txt
https://rendimento-predictor-api.onrender.com
```

## cURL

```bash
curl -X POST "https://rendimento-predictor-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "idade": 35,
    "sexo": "M",
    "cor_raca": "Branca",
    "anos_estudo": 12,
    "setor": "Servicos",
    "regiao": "Sudeste"
  }'
```

## Python

```python
import requests

url = "https://rendimento-predictor-api.onrender.com/predict"

payload = {
    "idade": 35,
    "sexo": "M",
    "cor_raca": "Branca",
    "anos_estudo": 12,
    "setor": "Servicos",
    "regiao": "Sudeste",
}

response = requests.post(url, json=payload, timeout=30)
response.raise_for_status()

print(response.json())
```

## JavaScript

```javascript
const url = "https://rendimento-predictor-api.onrender.com/predict";

const payload = {
  idade: 35,
  sexo: "M",
  cor_raca: "Branca",
  anos_estudo: 12,
  setor: "Servicos",
  regiao: "Sudeste",
};

const response = await fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(payload),
});

if (!response.ok) {
  throw new Error(`Erro na API: ${response.status}`);
}

const result = await response.json();
console.log(result);
```
