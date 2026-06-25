"""Rotinas de inicialização para ambiente de desenvolvimento."""

from __future__ import annotations

from datetime import date

from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.security import hash_password

SENHA_DEV_PADRAO = "senha123"


def _habilitar_sqlite_fk(engine: AsyncEngine) -> None:
    if not engine.url.drivername.startswith("sqlite"):
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _pragma_fk(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


async def criar_tabelas(engine: AsyncEngine) -> None:
    _habilitar_sqlite_fk(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _migrar_colunas_exame_sqlite(engine)


async def _migrar_colunas_exame_sqlite(engine: AsyncEngine) -> None:
    """Adiciona colunas de laudo em bancos SQLite já existentes (dev local)."""
    if not engine.url.drivername.startswith("sqlite"):
        return

    async with engine.begin() as conn:
        result = await conn.execute(text("PRAGMA table_info(exames)"))
        colunas = {row[1] for row in result.fetchall()}

        if "laudo_texto" not in colunas:
            try:
                await conn.execute(text("ALTER TABLE exames ADD COLUMN laudo_texto TEXT"))
            except Exception as exc:
                if "duplicate column" not in str(exc).lower():
                    raise

        if "status_exame" not in colunas:
            try:
                await conn.execute(
                    text(
                        "ALTER TABLE exames ADD COLUMN status_exame VARCHAR(20) "
                        "NOT NULL DEFAULT 'pendente'"
                    )
                )
            except Exception as exc:
                if "duplicate column" not in str(exc).lower():
                    raise

from app.database import AsyncSessionLocal, Base
from app.models import Paciente, Usuario


async def semear_dados_desenvolvimento() -> None:
    """Garante médico e paciente mockados para o frontend (id=1)."""
    async with AsyncSessionLocal() as session:
        medico = await session.scalar(
            select(Usuario).where(Usuario.email == "ana.silva@hospital.com")
        )
        if medico is None:
            medico = Usuario(
                nome="Dr. Ana Silva",
                crm="123456-SP",
                email="ana.silva@hospital.com",
                senha_hash=hash_password(SENHA_DEV_PADRAO),
            )
            session.add(medico)
            await session.flush()
        elif "placeholder" in medico.senha_hash:
            medico.senha_hash = hash_password(SENHA_DEV_PADRAO)

        paciente = await session.scalar(select(Paciente).where(Paciente.id == 1))
        if paciente is None:
            existente = await session.scalar(
                select(Paciente).where(Paciente.cpf == "12345678901")
            )
            if existente is None:
                session.add(
                    Paciente(
                        nome="João da Costa",
                        data_nascimento=date(1985, 3, 15),
                        sexo="M",
                        cpf="12345678901",
                        telefone="11999998888",
                        observacoes="Histórico familiar de epilepsia",
                        id_usuario=medico.id,
                    )
                )

        await session.commit()
