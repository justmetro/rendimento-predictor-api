from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String

from database.database import Base


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    idade = Column(Integer, nullable=False)
    sexo = Column(String, nullable=False)
    cor_raca = Column(String, nullable=False)
    anos_estudo = Column(Integer, nullable=False)
    setor = Column(String, nullable=False)
    regiao = Column(String, nullable=False)

    rendimento_hora_previsto = Column(Float, nullable=False)
    intervalo_min = Column(Float, nullable=False)
    intervalo_max = Column(Float, nullable=False)

    modelo = Column(String, nullable=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
