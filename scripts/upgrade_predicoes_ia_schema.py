"""Adiciona colunas novas em `predicoes_ia` para ambientes locais existentes."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

from sqlalchemy import inspect, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import engine


async def main() -> None:
    async with engine.begin() as conn:
        def _listar_colunas(sync_conn):
            return {col["name"] for col in inspect(sync_conn).get_columns("predicoes_ia")}

        colunas = await conn.run_sync(_listar_colunas)
        if "detalhes_json" not in colunas:
            await conn.execute(text("ALTER TABLE predicoes_ia ADD COLUMN detalhes_json TEXT"))
            print("Coluna adicionada: predicoes_ia.detalhes_json")
        else:
            print("Schema ja atualizado: predicoes_ia.detalhes_json existe.")


if __name__ == "__main__":
    asyncio.run(main())
