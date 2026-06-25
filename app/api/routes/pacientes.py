"""Rotas de gestão de pacientes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.deps import get_current_user
from app.database import get_db
from app.models import Paciente, Usuario
from app.schemas.paciente import PacienteCreate, PacienteResponse

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
                detail="id_usuario é obrigatório fora do ambiente de desenvolvimento.",
            )

    medico = await db.get(Usuario, usuario_id)
    if medico is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuário (médico) com id {usuario_id} não encontrado.",
        )

    return usuario_id


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
            detail=f"Paciente com id {paciente_id} não encontrado.",
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
        if "UNIQUE constraint failed" in str(exc) or "unique constraint" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="CPF já cadastrado para outro paciente.",
            ) from exc
        raise

    return paciente
