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


def _validar_telefone(valor: str | None) -> str | None:
    if valor is None:
        return None
    telefone = "".join(caractere for caractere in valor if caractere.isdigit())
    if len(telefone) not in (10, 11):
        raise ValueError("Telefone deve conter DDD e 10 ou 11 digitos.")
    return telefone


def _validar_nome(valor: str | None) -> str | None:
    if valor is None:
        return None
    nome = " ".join(valor.strip().split())
    if any(not (caractere.isalpha() or caractere in " '-") for caractere in nome):
        raise ValueError("Nome deve conter apenas letras, espacos, hifen ou apostrofo.")
    return nome


def _validar_data_nascimento(valor: date | None) -> date | None:
    if valor is not None and valor > date.today():
        raise ValueError("Data de nascimento nao pode estar no futuro.")
    return valor


class PacienteCreate(BaseModel):
    nome: str = Field(..., min_length=2, max_length=200)
    data_nascimento: date
    sexo: str = Field(..., min_length=1, max_length=1, pattern=r"^[MF]$")
    cpf: str | None = Field(default=None, max_length=14)
    telefone: str | None = Field(default=None, max_length=15)
    observacoes: str | None = None
    id_usuario: int | None = Field(
        default=None,
        description="Médico responsável; padrão id=1 em desenvolvimento.",
    )

    _cpf_valido = field_validator("cpf")(_validar_cpf)
    _telefone_valido = field_validator("telefone")(_validar_telefone)
    _nome_valido = field_validator("nome")(_validar_nome)
    _data_nascimento_valida = field_validator("data_nascimento")(_validar_data_nascimento)


class PacienteUpdate(BaseModel):
    nome: str | None = Field(default=None, min_length=2, max_length=200)
    data_nascimento: date | None = None
    sexo: str | None = Field(default=None, min_length=1, max_length=1, pattern=r"^[MF]$")
    cpf: str | None = Field(default=None, max_length=14)
    telefone: str | None = Field(default=None, max_length=15)
    observacoes: str | None = None
    id_usuario: int | None = None

    _cpf_valido = field_validator("cpf")(_validar_cpf)
    _telefone_valido = field_validator("telefone")(_validar_telefone)
    _nome_valido = field_validator("nome")(_validar_nome)
    _data_nascimento_valida = field_validator("data_nascimento")(_validar_data_nascimento)


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
