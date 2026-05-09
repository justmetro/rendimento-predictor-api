import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/predictions.db")
database_url = make_url(DATABASE_URL)
is_sqlite = database_url.get_backend_name() == "sqlite"


if is_sqlite and database_url.database and database_url.database != ":memory:":
    Path(database_url.database).parent.mkdir(parents=True, exist_ok=True)


connect_args = {"check_same_thread": False} if is_sqlite else {}


engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def check_database_connection() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return True
    except Exception:
        return False


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
