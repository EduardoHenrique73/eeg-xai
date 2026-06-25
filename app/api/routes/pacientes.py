"""Rotas de gestao de pacientes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.deps import get_current_user
from app.database import get_db
from app.models import Paciente, Usuario
from app.schemas.paciente import PacienteCreate, PacienteResponse, PacienteUpdate

router = APIRouter(
    prefix="/api/pacientes",
    tags=["Pacientes"],
    dependencies=[Depends(get_current_user)],
)

USUARIO_PADRAO_DEV_ID = 1


async def _resolver_id_usuario(
    payload: PacienteCreate,
    db: AsyncSession,
    settings: Settings,
) -> int:
    usuario_id = payload.id_usuario

    if usuario_id is None:
        if settings.is_development:
            usuario_id = USUARIO_PADRAO_DEV_ID
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="id_usuario e obrigatorio fora do ambiente de desenvolvimento.",
            )

    medico = await db.get(Usuario, usuario_id)
    if medico is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuario (medico) com id {usuario_id} nao encontrado.",
        )

    return usuario_id


def _is_unique_violation(exc: Exception) -> bool:
    mensagem = str(exc).lower()
    return "unique constraint failed" in mensagem or "unique constraint" in mensagem


@router.get("", response_model=list[PacienteResponse], summary="Listar pacientes")
async def listar_pacientes(
    db: AsyncSession = Depends(get_db),
) -> list[PacienteResponse]:
    result = await db.execute(select(Paciente).order_by(Paciente.nome.asc()))
    return list(result.scalars().all())


@router.get(
    "/{paciente_id}",
    response_model=PacienteResponse,
    summary="Obter paciente por ID",
)
async def obter_paciente(
    paciente_id: int,
    db: AsyncSession = Depends(get_db),
) -> PacienteResponse:
    paciente = await db.get(Paciente, paciente_id)
    if paciente is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paciente com id {paciente_id} nao encontrado.",
        )
    return paciente


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=PacienteResponse,
    summary="Cadastrar novo paciente",
)
async def criar_paciente(
    payload: PacienteCreate,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> PacienteResponse:
    usuario_id = await _resolver_id_usuario(payload, db, settings)

    paciente = Paciente(
        nome=payload.nome,
        data_nascimento=payload.data_nascimento,
        sexo=payload.sexo,
        cpf=payload.cpf,
        telefone=payload.telefone,
        observacoes=payload.observacoes,
        id_usuario=usuario_id,
    )
    db.add(paciente)

    try:
        await db.flush()
        await db.refresh(paciente)
    except Exception as exc:
        await db.rollback()
        if _is_unique_violation(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="CPF ja cadastrado para outro paciente.",
            ) from exc
        raise

    return paciente


@router.patch(
    "/{paciente_id}",
    response_model=PacienteResponse,
    summary="Atualizar paciente",
)
async def atualizar_paciente(
    paciente_id: int,
    payload: PacienteUpdate,
    db: AsyncSession = Depends(get_db),
) -> PacienteResponse:
    paciente = await db.get(Paciente, paciente_id)
    if paciente is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paciente com id {paciente_id} nao encontrado.",
        )

    dados = payload.model_dump(exclude_unset=True)
    usuario_id = dados.get("id_usuario")
    if usuario_id is not None:
        medico = await db.get(Usuario, usuario_id)
        if medico is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Usuario (medico) com id {usuario_id} nao encontrado.",
            )

    for campo, valor in dados.items():
        setattr(paciente, campo, valor)

    try:
        await db.flush()
        await db.refresh(paciente)
    except Exception as exc:
        await db.rollback()
        if _is_unique_violation(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="CPF ja cadastrado para outro paciente.",
            ) from exc
        raise

    return paciente


@router.delete(
    "/{paciente_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Excluir paciente",
)
async def excluir_paciente(
    paciente_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    paciente = await db.get(Paciente, paciente_id)
    if paciente is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paciente com id {paciente_id} nao encontrado.",
        )

    await db.delete(paciente)
    await db.flush()
