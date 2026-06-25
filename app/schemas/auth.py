"""Schemas Pydantic — autenticação."""

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    email: EmailStr
    senha: str = Field(..., min_length=1)


class MedicoAuthResponse(BaseModel):
    id: int
    nome: str
    email: EmailStr
    crm: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    medico: MedicoAuthResponse


class RecuperarSenhaRequest(BaseModel):
    email: EmailStr


class RecuperarSenhaResponse(BaseModel):
    message: str = (
        "Se o e-mail estiver cadastrado, você receberá instruções de recuperação em instantes."
    )
