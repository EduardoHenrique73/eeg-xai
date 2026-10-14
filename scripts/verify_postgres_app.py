"""Smoke-test the FastAPI app against the local PostgreSQL configuration."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

from dotenv import dotenv_values
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.engine import make_url


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

url = get_settings().database_url
expected = dotenv_values(ROOT / ".env.postgres.local").get("EEG_XAI_POSTGRES_URL")
if not expected or url != expected or make_url(url).drivername != "postgresql+asyncpg":
    raise SystemExit("DATABASE_URL deve apontar para o PostgreSQL local configurado")

from app.database import AsyncSessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Exame, Paciente, PredicaoIA, Usuario  # noqa: E402


async def main() -> None:
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            response.raise_for_status()
            assert response.json()["status"] == "ok"

        async with AsyncSessionLocal() as session:
            counts = {
                model.__tablename__: await session.scalar(select(func.count()).select_from(model))
                for model in (Usuario, Paciente, Exame, PredicaoIA)
            }
            assert counts == {"usuarios": 1, "pacientes": 1, "exames": 14, "predicoes_ia": 14}
            print(f"FastAPI + PostgreSQL OK: {counts}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
