"""Rotas de gestao de pacientes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db
from app.models import Paciente, Usuario
from app.schemas.paciente import PacienteCreate, PacienteResponse, PacienteUpdate

router = APIRouter(
    prefix="/api/pacientes",
    tags=["Pacientes"],
    dependencies=[Depends(get_current_user)],
)

async def _obter_paciente_do_usuario(
    paciente_id: int,
    usuario: Usuario,
    db: AsyncSession,
) -> Paciente:
    result = await db.execute(
        select(Paciente).where(
            Paciente.id == paciente_id,
            Paciente.id_usuario == usuario.id,
        )
    )
    paciente = result.scalar_one_or_none()
    if paciente is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paciente com id {paciente_id} nao encontrado.",
        )
    return paciente


def _is_unique_violation(exc: Exception) -> bool:
    mensagem = str(exc).lower()
    return "unique constraint failed" in mensagem or "unique constraint" in mensagem


@router.get("", response_model=list[PacienteResponse], summary="Listar pacientes")
async def listar_pacientes(
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(get_current_user),
) -> list[PacienteResponse]:
    result = await db.execute(
        select(Paciente)
        .where(Paciente.id_usuario == usuario.id)
        .order_by(Paciente.nome.asc())
    )
    return list(result.scalars().all())


@router.get(
    "/{paciente_id}",
    response_model=PacienteResponse,
    summary="Obter paciente por ID",
)
async def obter_paciente(
    paciente_id: int,
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(get_current_user),
) -> PacienteResponse:
    return await _obter_paciente_do_usuario(paciente_id, usuario, db)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=PacienteResponse,
    summary="Cadastrar novo paciente",
)
async def criar_paciente(
    payload: PacienteCreate,
    db: AsyncSession = Depends(get_db),
    usuario: Usuario = Depends(get_current_user),
) -> PacienteResponse:
    if payload.id_usuario is not None and payload.id_usuario != usuario.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Nao e permitido cadastrar paciente para outro medico.",
        )

    paciente = Paciente(
        nome=payload.nome,
        data_nascimento=payload.data_nascimento,
        sexo=payload.sexo,
        cpf=payload.cpf,
        telefone=payload.telefone,
        observacoes=payload.observacoes,
        id_usuario=usuario.id,
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
    usuario: Usuario = Depends(get_current_user),
) -> PacienteResponse:
    paciente = await _obter_paciente_do_usuario(paciente_id, usuario, db)

    dados = payload.model_dump(exclude_unset=True)
    usuario_id = dados.get("id_usuario")
    if usuario_id is not None and usuario_id != usuario.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Nao e permitido transferir paciente para outro medico.",
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
    usuario: Usuario = Depends(get_current_user),
) -> None:
    paciente = await _obter_paciente_do_usuario(paciente_id, usuario, db)

    await db.delete(paciente)
    await db.flush()
