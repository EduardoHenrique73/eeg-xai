"""Schemas Pydantic — pacientes clínicos."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class PacienteCreate(BaseModel):
    nome: str = Field(..., min_length=2, max_length=200)
    data_nascimento: date
    sexo: str = Field(..., min_length=1, max_length=1, pattern=r"^[MF]$")
    cpf: str | None = Field(default=None, max_length=11)
    telefone: str | None = Field(default=None, max_length=20)
    observacoes: str | None = None
    id_usuario: int | None = Field(
        default=None,
        description="Médico responsável; padrão id=1 em desenvolvimento.",
    )


class PacienteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    nome: str
    data_nascimento: date
    sexo: str
    cpf: str | None
    telefone: str | None
    observacoes: str | None
    id_usuario: int
    created_at: datetime | None = None
    updated_at: datetime | None = None
