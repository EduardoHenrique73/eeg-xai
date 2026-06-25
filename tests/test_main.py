"""Testes da aplicação FastAPI (smoke tests)."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_check_retorna_ok():
  transport = ASGITransport(app=app)
  async with AsyncClient(transport=transport, base_url="http://test") as client:
      response = await client.get("/health")

  assert response.status_code == 200
  payload = response.json()
  assert payload["status"] == "ok"
  assert payload["service"] == "eeg-xai"


@pytest.mark.asyncio
async def test_root_retorna_informacoes_da_api():
  transport = ASGITransport(app=app)
  async with AsyncClient(transport=transport, base_url="http://test") as client:
      response = await client.get("/")

  assert response.status_code == 200
  assert "version" in response.json()
