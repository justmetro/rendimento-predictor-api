from database.database import Base, engine
from database.models import PredictionRecord


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Banco de dados inicializado com sucesso.")