"""Modelo PredicaoIA — laudo da CNN-LSTM + mapa SHAP no disco."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.exame import Exame


class PredicaoIA(Base):
    __tablename__ = "predicoes_ia"
    __table_args__ = (
        CheckConstraint(
            "resultado_score >= 0 AND resultado_score <= 1",
            name="ck_predicoes_ia_score_range",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    id_exame: Mapped[int] = mapped_column(
        ForeignKey("exames.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    resultado_score: Mapped[float] = mapped_column(Float, nullable=False)
    mapa_shap_path: Mapped[str] = mapped_column(String(500), nullable=False)
    data_analise: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    exame: Mapped[Exame] = relationship("Exame", back_populates="predicoes")

    def __repr__(self) -> str:
        return f"<PredicaoIA id={self.id} score={self.resultado_score}>"
