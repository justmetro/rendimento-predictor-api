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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
