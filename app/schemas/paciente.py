"""Schemas Pydantic — pacientes clínicos."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _validar_cpf(valor: str | None) -> str | None:
    if valor is None:
        return None
    cpf = "".join(caractere for caractere in valor if caractere.isdigit())
    if len(cpf) != 11 or cpf == cpf[0] * 11:
        raise ValueError("CPF invalido.")
    for posicao in (9, 10):
        soma = sum(int(cpf[indice]) * (posicao + 1 - indice) for indice in range(posicao))
        digito = (soma * 10) % 11
        if digito == 10:
            digito = 0
        if digito != int(cpf[posicao]):
            raise ValueError("CPF invalido.")
    return cpf


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

    _cpf_valido = field_validator("cpf")(_validar_cpf)


class PacienteUpdate(BaseModel):
    nome: str | None = Field(default=None, min_length=2, max_length=200)
    data_nascimento: date | None = None
    sexo: str | None = Field(default=None, min_length=1, max_length=1, pattern=r"^[MF]$")
    cpf: str | None = Field(default=None, max_length=11)
    telefone: str | None = Field(default=None, max_length=20)
    observacoes: str | None = None
    id_usuario: int | None = None

    _cpf_valido = field_validator("cpf")(_validar_cpf)


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
