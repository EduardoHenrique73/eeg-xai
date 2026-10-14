"""Copy clinical records to an empty PostgreSQL database without altering SQLite.

Set EEG_XAI_POSTGRES_URL to a postgresql+asyncpg URL. By default the SQLite
source is read from the current DATABASE_URL in .env. No EDF/model files move.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import sys

from sqlalchemy import DateTime, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.database import Base  # noqa: E402
import app.models  # noqa: E402,F401


TABLES = tuple(Base.metadata.tables[name] for name in (
    "usuarios", "pacientes", "exames", "predicoes_ia"
))


async def copy_records(source: AsyncEngine, target: AsyncEngine) -> dict[str, int]:
    """Copy in FK order; abort if target already contains clinical records."""
    async with source.connect() as source_conn, target.begin() as target_conn:
        for table in TABLES:
            existing = await target_conn.scalar(select(func.count()).select_from(table))
            if existing:
                raise ValueError(f"Destino nao vazio: {table.name}. Nenhum registro copiado.")

        counts: dict[str, int] = {}
        for table in TABLES:
            rows = (await source_conn.execute(select(table))).mappings().all()
            if rows:
                values = []
                for row in rows:
                    record = dict(row)
                    for column in table.columns:
                        value = record[column.name]
                        if isinstance(column.type, DateTime) and value is not None:
                            record[column.name] = _as_utc(value)
                    values.append(record)
                await target_conn.execute(table.insert(), values)
            counts[table.name] = len(rows)

        if target.dialect.name == "postgresql":
            for table in TABLES:
                max_id = await target_conn.scalar(select(func.max(table.c.id)))
                if max_id is not None:
                    await target_conn.execute(
                        text("SELECT setval(pg_get_serial_sequence(:table, 'id'), :value, true)"),
                        {"table": table.name, "value": max_id},
                    )

        for table in TABLES:
            copied = await target_conn.scalar(select(func.count()).select_from(table))
            if copied != counts[table.name]:
                raise RuntimeError(f"Contagem divergente em {table.name}")
        return counts


async def migrate(source_url: str, target_url: str) -> dict[str, int]:
    source_kind = make_url(source_url).drivername
    target_kind = make_url(target_url).drivername
    if not source_kind.startswith("sqlite"):
        raise ValueError("A origem deve ser SQLite; este comando nunca apaga a origem.")
    if target_kind != "postgresql+asyncpg":
        raise ValueError("O destino deve usar postgresql+asyncpg.")

    source = create_async_engine(source_url)
    target = create_async_engine(target_url)
    try:
        async with target.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        return await copy_records(source, target)
    finally:
        await source.dispose()
        await target.dispose()


def _comparable(value: object) -> object:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value
    return value


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


async def repair_timestamps(source_url: str, target_url: str) -> dict[str, int]:
    """Repair only datetime fields, after proving other fields match source."""
    source = create_async_engine(source_url)
    target = create_async_engine(target_url)
    try:
        async with source.connect() as src, target.begin() as dst:
            changes: dict[str, int] = {}
            for table in TABLES:
                left = (await src.execute(select(table).order_by(table.c.id))).mappings().all()
                right = (await dst.execute(select(table).order_by(table.c.id))).mappings().all()
                if len(left) != len(right):
                    raise RuntimeError(f"Contagem divergente em {table.name}")
                timestamp_names = {
                    column.name for column in table.columns
                    if isinstance(column.type, DateTime)
                }
                pending = []
                for original, copied in zip(left, right):
                    if original["id"] != copied["id"] or any(
                        original[key] != copied[key]
                        for key in original if key not in timestamp_names
                    ):
                        raise RuntimeError(f"Campos nao temporais divergentes em {table.name}")
                    timestamps = {
                        key: _as_utc(original[key]) for key in timestamp_names
                        if original[key] is not None
                        and _comparable(original[key]) != _comparable(copied[key])
                    }
                    if timestamps:
                        pending.append((original["id"], timestamps))
                for identifier, timestamps in pending:
                    await dst.execute(table.update().where(table.c.id == identifier).values(**timestamps))
                changes[table.name] = len(pending)
            return changes
    finally:
        await source.dispose()
        await target.dispose()


async def verify_records(source_url: str, target_url: str) -> dict[str, int]:
    source = create_async_engine(source_url)
    target = create_async_engine(target_url)
    try:
        async with source.connect() as src, target.connect() as dst:
            counts = {}
            for table in TABLES:
                original = (await src.execute(select(table).order_by(table.c.id))).mappings().all()
                copied = (await dst.execute(select(table).order_by(table.c.id))).mappings().all()
                if len(original) != len(copied):
                    raise RuntimeError(f"Contagem divergente em {table.name}")
                for left, right in zip(original, copied):
                    different = [
                        key for key in left
                        if _comparable(left[key]) != _comparable(right[key])
                    ]
                    if different:
                        raise RuntimeError(
                            f"Dados divergentes em {table.name}, id={left['id']}, colunas={different}"
                        )
                counts[table.name] = len(original)
            return counts
    finally:
        await source.dispose()
        await target.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", help="URL SQLite; padrao: DATABASE_URL atual")
    parser.add_argument("--check-only", action="store_true", help="Verifica destino sem copiar registros")
    parser.add_argument("--verify-only", action="store_true", help="Compara os registros dos dois bancos sem alterar dados")
    parser.add_argument("--repair-timestamps", action="store_true", help="Corrige apenas datas da copia existente, apos validar outros campos")
    parser.add_argument(
        "--target-env-file", type=Path, default=ROOT / ".env.postgres.local",
        help="Arquivo local ignorado pelo Git com EEG_XAI_POSTGRES_URL",
    )
    args = parser.parse_args()
    target_url = os.environ.get("EEG_XAI_POSTGRES_URL")
    if not target_url and args.target_env_file.is_file():
        target_url = dotenv_values(args.target_env_file).get("EEG_XAI_POSTGRES_URL")
    if not target_url:
        parser.error(
            "Defina EEG_XAI_POSTGRES_URL no ambiente ou em .env.postgres.local "
            "(arquivo ignorado pelo Git)."
        )
    source_url = args.source_url or get_settings().database_url
    if not make_url(source_url).drivername.startswith("sqlite"):
        backup_env = ROOT / ".env.sqlite.backup"
        source_url = dotenv_values(backup_env).get("DATABASE_URL") if backup_env.is_file() else None
        if not source_url or not make_url(source_url).drivername.startswith("sqlite"):
            parser.error("Informe --source-url SQLite ou preserve .env.sqlite.backup.")
    if args.check_only:
        async def check() -> tuple[str, str, list[str]]:
            engine = create_async_engine(target_url)
            try:
                async with engine.connect() as conn:
                    version = await conn.scalar(text("SHOW server_version"))
                    data_dir = await conn.scalar(text("SHOW data_directory"))
                    rows = await conn.execute(text(
                        "SELECT tablename FROM pg_tables "
                        "WHERE schemaname = 'public' ORDER BY tablename"
                    ))
                    return str(version), str(data_dir), [row[0] for row in rows]
            finally:
                await engine.dispose()

        version, data_dir, tables = asyncio.run(check())
        print(f"Conexao PostgreSQL OK. Versao: {version}; data_directory: {data_dir}")
        print(f"Tabelas public: {tables}")
        return
    if args.verify_only:
        counts = asyncio.run(verify_records(source_url, target_url))
        print(f"Registros identicos por tabela: {counts}")
        return
    if args.repair_timestamps:
        changes = asyncio.run(repair_timestamps(source_url, target_url))
        print(f"Registros com datas corrigidas: {changes}")
        return
    sqlite_path = Path(make_url(source_url).database or "")
    if not sqlite_path.is_file():
        parser.error(f"Banco SQLite de origem nao encontrado: {sqlite_path}")
    backup_dir = ROOT / "transfer_data"
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / f"sqlite_before_postgres_{datetime.now():%Y%m%d_%H%M%S}.db"
    with sqlite3.connect(sqlite_path) as original, sqlite3.connect(backup) as snapshot:
        original.backup(snapshot)
    print(f"Backup SQLite preservado em: {backup}")
    counts = asyncio.run(migrate(source_url, target_url))
    print(f"Registros copiados e conferidos: {counts}")
    print("SQLite de origem preservado. Altere DATABASE_URL apenas apos validar PostgreSQL.")


if __name__ == "__main__":
    main()
