"""Testes das rotas de pacientes."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.deps import get_current_user
from app.database import get_db
from app.main import app


@pytest_asyncio.fixture
async def client_pacientes(async_engine, usuario_medico):
    session_factory = async_sessionmaker(async_engine, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def override_current_user():
        return usuario_medico

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_current_user
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_listar_pacientes_retorna_lista(client_pacientes, paciente):
    response = await client_pacientes.get("/api/pacientes")
    assert response.status_code == 200
    dados = response.json()
    assert len(dados) >= 1
    assert dados[0]["nome"] == paciente.nome


@pytest.mark.asyncio
async def test_obter_paciente_por_id(client_pacientes, paciente):
    response = await client_pacientes.get(f"/api/pacientes/{paciente.id}")
    assert response.status_code == 200
    assert response.json()["cpf"] == paciente.cpf


@pytest.mark.asyncio
async def test_obter_paciente_inexistente_retorna_404(client_pacientes):
    response = await client_pacientes.get("/api/pacientes/99999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_criar_paciente_vincula_usuario_padrao_dev(
    client_pacientes,
    usuario_medico,
):
    response = await client_pacientes.post(
        "/api/pacientes",
        json={
            "nome": "Maria Souza",
            "data_nascimento": "1990-07-20",
            "sexo": "F",
            "cpf": "98765432100",
            "telefone": "11988887777",
        },
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["nome"] == "Maria Souza"
    assert payload["id_usuario"] == usuario_medico.id
