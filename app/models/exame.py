"""Modelo Exame — prontuário do sinal EEG (.edf no disco)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

STATUS_EXAME_PENDENTE = "pendente"
STATUS_EXAME_CONCLUIDO = "concluido"

if TYPE_CHECKING:
    from app.models.paciente import Paciente
    from app.models.predicao_ia import PredicaoIA


class Exame(Base):
    __tablename__ = "exames"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    id_paciente: Mapped[int] = mapped_column(
        ForeignKey("pacientes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    data_upload: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    taxa_amostragem: Mapped[float] = mapped_column(Float, nullable=False)
    canais_eeg: Mapped[str | None] = mapped_column(Text, nullable=True)
    arquivo_path: Mapped[str] = mapped_column(String(500), nullable=False)
    laudo_texto: Mapped[str | None] = mapped_column(Text, nullable=True)
    status_exame: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=STATUS_EXAME_PENDENTE,
        server_default=STATUS_EXAME_PENDENTE,
    )

    paciente: Mapped[Paciente] = relationship("Paciente", back_populates="exames")
    predicoes: Mapped[list[PredicaoIA]] = relationship(
        "PredicaoIA",
        back_populates="exame",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Exame id={self.id} arquivo_path={self.arquivo_path!r}>"
