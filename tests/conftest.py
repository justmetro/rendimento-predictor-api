from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.database import Base, get_db
from database.models import PredictionRecord
from main import app


TEST_DATABASE_PATH = Path("data/test_predictions.db")
TEST_DATABASE_URL = "sqlite:///./data/test_predictions.db"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)


def override_get_db():
    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()


def prepare_test_database() -> None:
    TEST_DATABASE_PATH.parent.mkdir(exist_ok=True)

    if TEST_DATABASE_PATH.exists():
        TEST_DATABASE_PATH.unlink()

    Base.metadata.create_all(bind=test_engine)


def cleanup_test_database() -> None:
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()

    if TEST_DATABASE_PATH.exists():
        TEST_DATABASE_PATH.unlink()


prepare_test_database()
app.dependency_overrides[get_db] = override_get_db


def pytest_sessionfinish(session, exitstatus):
    app.dependency_overrides.pop(get_db, None)
    cleanup_test_database()
