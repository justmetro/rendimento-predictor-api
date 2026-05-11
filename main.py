from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from database.init_db import init_db


init_db()

app = FastAPI(
    title="Rendimento Predictor API",
    description="API para predizer rendimento do trabalho usando Machine Learning.",
    version="0.1.0"
)

ALLOWED_ORIGINS = [
    "https://rendimento-predictor-api.streamlit.app",
    "https://rendimento-predictor-api.onrender.com",
    "http://localhost:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
