"""Fixtures compartilhadas para testes (TDD)."""

from datetime import date

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.security import hash_password
from app.database import Base
from app.models import Exame, Paciente, PredicaoIA, Usuario


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def async_engine():
    """Engine SQLite em memória para testes isolados."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine) -> AsyncSession:
    """Sessão assíncrona com rollback automático por teste."""
    session_factory = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def usuario_medico(db_session: AsyncSession) -> Usuario:
    medico = Usuario(
        nome="Dr. Ana Silva",
        crm="123456-SP",
        email="ana.silva@hospital.com",
        senha_hash=hash_password("senha-teste"),
    )
    db_session.add(medico)
    await db_session.commit()
    await db_session.refresh(medico)
    return medico


@pytest_asyncio.fixture
async def paciente(db_session: AsyncSession, usuario_medico: Usuario) -> Paciente:
    subject = Paciente(
        nome="João da Costa",
        data_nascimento=date(1985, 3, 15),
        sexo="M",
        cpf="12345678901",
        telefone="11999998888",
        observacoes="Histórico familiar de epilepsia",
        id_usuario=usuario_medico.id,
    )
    db_session.add(subject)
    await db_session.commit()
    await db_session.refresh(subject)
    return subject


@pytest_asyncio.fixture
async def exame(db_session: AsyncSession, paciente: Paciente) -> Exame:
    registro = Exame(
        id_paciente=paciente.id,
        taxa_amostragem=256.0,
        arquivo_path="C:/storage/edf/exame_001.edf",
    )
    db_session.add(registro)
    await db_session.commit()
    await db_session.refresh(registro)
    return registro
