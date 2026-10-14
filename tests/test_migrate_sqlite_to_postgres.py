from datetime import date

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from app.database import Base
from app.models import Usuario, Paciente
from scripts.migrate_sqlite_to_postgres import copy_records


@pytest.mark.asyncio
async def test_copy_records_keeps_source_and_rejects_nonempty_target(tmp_path):
    source = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'source.db').as_posix()}")
    target = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'target.db').as_posix()}")
    try:
        for engine in (source, target):
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        async with source.begin() as conn:
            await conn.execute(Usuario.__table__.insert().values(
                id=10, nome="Teste", crm="1", email="test@example.org", senha_hash="hash"
            ))
            await conn.execute(Paciente.__table__.insert().values(
                id=20, nome="Paciente", data_nascimento=date(2000, 1, 1), sexo="F", id_usuario=10
            ))
        assert await copy_records(source, target) == {
            "usuarios": 1, "pacientes": 1, "exames": 0, "predicoes_ia": 0
        }
        with pytest.raises(ValueError, match="Destino nao vazio"):
            await copy_records(source, target)
        async with source.connect() as conn:
            assert len((await conn.execute(Usuario.__table__.select())).all()) == 1
    finally:
        await source.dispose()
        await target.dispose()
