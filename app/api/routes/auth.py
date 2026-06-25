"""Rotas de autenticação médica (JWT)."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, verify_password
from app.database import get_db
from app.models import Usuario
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    MedicoAuthResponse,
    RecuperarSenhaRequest,
    RecuperarSenhaResponse,
)

router = APIRouter(prefix="/api/auth", tags=["Autenticação"])


@router.post("/login", response_model=LoginResponse, summary="Login do médico")
async def login(
    credenciais: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> LoginResponse:
    result = await db.execute(
        select(Usuario).where(Usuario.email == credenciais.email.lower())
    )
    usuario = result.scalar_one_or_none()

    if usuario is None or not verify_password(credenciais.senha, usuario.senha_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-mail ou senha inválidos.",
        )

    token = create_access_token(
        subject=str(usuario.id),
        extra_claims={"email": usuario.email},
    )

    return LoginResponse(
        access_token=token,
        medico=MedicoAuthResponse(
            id=usuario.id,
            nome=usuario.nome,
            email=usuario.email,
            crm=usuario.crm,
        ),
    )


@router.post(
    "/recuperar-senha",
    response_model=RecuperarSenhaResponse,
    summary="Solicitar recuperação de senha (simulado)",
)
async def recuperar_senha(
    payload: RecuperarSenhaRequest,
    db: AsyncSession = Depends(get_db),
) -> RecuperarSenhaResponse:
    """Simula envio de instruções — não expõe se o e-mail existe (boa prática)."""
    await db.execute(select(Usuario).where(Usuario.email == payload.email.lower()))
    return RecuperarSenhaResponse()
