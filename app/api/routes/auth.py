"""Rotas de autenticacao e configuracoes do medico."""

import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.core.security import create_access_token, verify_password
from app.database import get_db
from app.models import Usuario
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    MedicoAuthResponse,
    MedicoConfigUpdate,
    RecuperarSenhaRequest,
    RecuperarSenhaResponse,
)

router = APIRouter(prefix="/api/auth", tags=["Autenticacao"])


def _montagem_para_lista(valor: str | None) -> list[str]:
    if not valor:
        return []
    try:
        dados = json.loads(valor)
    except json.JSONDecodeError:
        return []
    if not isinstance(dados, list):
        return []
    return [str(canal) for canal in dados if str(canal).strip()]


def _usuario_para_auth(usuario: Usuario) -> MedicoAuthResponse:
    return MedicoAuthResponse(
        id=usuario.id,
        nome=usuario.nome,
        email=usuario.email,
        crm=usuario.crm,
        threshold_confianca=usuario.threshold_confianca,
        montagem_padrao=_montagem_para_lista(usuario.montagem_padrao),
        exibir_shap=usuario.exibir_shap,
    )


@router.post("/login", response_model=LoginResponse, summary="Login do medico")
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
            detail="E-mail ou senha invalidos.",
        )

    token = create_access_token(
        subject=str(usuario.id),
        extra_claims={"email": usuario.email},
    )

    return LoginResponse(access_token=token, medico=_usuario_para_auth(usuario))


@router.get("/me", response_model=MedicoAuthResponse, summary="Perfil e preferencias do medico")
async def obter_me(
    usuario: Usuario = Depends(get_current_user),
) -> MedicoAuthResponse:
    return _usuario_para_auth(usuario)


@router.patch(
    "/me",
    response_model=MedicoAuthResponse,
    summary="Atualizar perfil e preferencias do medico",
)
async def atualizar_me(
    payload: MedicoConfigUpdate,
    usuario: Usuario = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MedicoAuthResponse:
    usuario_db = await db.get(Usuario, usuario.id)
    if usuario_db is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario nao encontrado.",
        )

    email = payload.email.lower()
    existente = await db.scalar(select(Usuario).where(Usuario.email == email))
    if existente is not None and existente.id != usuario_db.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="E-mail ja cadastrado para outro medico.",
        )

    usuario_db.nome = payload.nome.strip()
    usuario_db.email = email
    usuario_db.crm = payload.crm.strip()
    usuario_db.threshold_confianca = payload.threshold_confianca
    usuario_db.montagem_padrao = json.dumps(payload.montagem_padrao)
    usuario_db.exibir_shap = payload.exibir_shap

    await db.flush()
    await db.refresh(usuario_db)
    return _usuario_para_auth(usuario_db)


@router.post(
    "/recuperar-senha",
    response_model=RecuperarSenhaResponse,
    summary="Solicitar recuperacao de senha (simulado)",
)
async def recuperar_senha(
    payload: RecuperarSenhaRequest,
    db: AsyncSession = Depends(get_db),
) -> RecuperarSenhaResponse:
    await db.execute(select(Usuario).where(Usuario.email == payload.email.lower()))
    return RecuperarSenhaResponse()
