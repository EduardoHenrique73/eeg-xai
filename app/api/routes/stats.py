"""Rota de estatísticas gerais para o Dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user
from app.database import get_db
from app.models import Exame, Paciente
from app.models.exame import STATUS_EXAME_CONCLUIDO

router = APIRouter(prefix="/api/stats", tags=["Dashboard"])


class DashboardStats(BaseModel):
    total_pacientes: int
    exames_pendentes: int
    laudos_emitidos: int


@router.get(
    "",
    response_model=DashboardStats,
    summary="Estatísticas gerais do dashboard",
    dependencies=[Depends(get_current_user)],
)
async def obter_stats(db: AsyncSession = Depends(get_db)) -> DashboardStats:
    total_pacientes = await db.scalar(select(func.count()).select_from(Paciente)) or 0

    exames_pendentes = (
        await db.scalar(
            select(func.count())
            .select_from(Exame)
            .where(Exame.status_exame != STATUS_EXAME_CONCLUIDO)
        )
    ) or 0

    laudos_emitidos = (
        await db.scalar(
            select(func.count())
            .select_from(Exame)
            .where(Exame.status_exame == STATUS_EXAME_CONCLUIDO)
        )
    ) or 0

    return DashboardStats(
        total_pacientes=total_pacientes,
        exames_pendentes=exames_pendentes,
        laudos_emitidos=laudos_emitidos,
    )
