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
    threshold_confianca: float = Field(default=0.5, ge=0.0, le=1.0)
    montagem_padrao: list[str] = Field(default_factory=list)
    exibir_shap: bool = True


class MedicoConfigUpdate(BaseModel):
    nome: str = Field(..., min_length=2, max_length=200)
    email: EmailStr
    crm: str = Field(..., min_length=3, max_length=20)
    threshold_confianca: float = Field(..., ge=0.0, le=1.0)
    montagem_padrao: list[str] = Field(default_factory=list)
    exibir_shap: bool = True


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
