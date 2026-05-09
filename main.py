from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Rendimento Predictor API",
    description="API para predizer rendimento do trabalho usando Machine Learning.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Rendimento Predictor API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": False,
        "version": "0.1.0"
    }


@app.get("/features")
def get_features():
    return {
        "target": "rendimento_hora",
        "features": [
            "idade",
            "sexo",
            "cor_raca",
            "anos_estudo",
            "setor",
            "regiao"
        ]
    }