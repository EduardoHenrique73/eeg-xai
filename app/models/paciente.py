"""Modelo Paciente — sujeito do exame neurológico."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Date, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.base import TimestampMixin

if TYPE_CHECKING:
    from app.models.exame import Exame
    from app.models.usuario import Usuario


class Paciente(Base, TimestampMixin):
    __tablename__ = "pacientes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nome: Mapped[str] = mapped_column(String(200), nullable=False)
    data_nascimento: Mapped[date] = mapped_column(Date, nullable=False)
    sexo: Mapped[str] = mapped_column(String(1), nullable=False)
    cpf: Mapped[Optional[str]] = mapped_column(String(11), unique=True, nullable=True)
    telefone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    observacoes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    id_usuario: Mapped[int] = mapped_column(
        ForeignKey("usuarios.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    medico: Mapped[Usuario] = relationship("Usuario", back_populates="pacientes")
    exames: Mapped[list[Exame]] = relationship(
        "Exame",
        back_populates="paciente",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Paciente id={self.id} nome={self.nome!r}>"
