"""Testes das rotas de autenticação."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.security import hash_password
from app.database import get_db
from app.main import app
from app.models import Usuario


@pytest_asyncio.fixture
async def client_auth(async_engine):
    session_factory = async_sessionmaker(async_engine, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def medico_com_senha(db_session) -> Usuario:
    medico = Usuario(
        nome="Dr. Teste Auth",
        crm="999999-SP",
        email="medico.auth@test.com",
        senha_hash=hash_password("senha-forte"),
    )
    db_session.add(medico)
    await db_session.commit()
    await db_session.refresh(medico)
    return medico


@pytest.mark.asyncio
async def test_login_credenciais_validas_retorna_jwt(client_auth, medico_com_senha):
    response = await client_auth.post(
        "/api/auth/login",
        json={"email": medico_com_senha.email, "senha": "senha-forte"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["access_token"]
    assert payload["medico"]["id"] == medico_com_senha.id
    assert payload["medico"]["nome"] == medico_com_senha.nome


@pytest.mark.asyncio
async def test_login_credenciais_invalidas_retorna_401(client_auth, medico_com_senha):
    response = await client_auth.post(
        "/api/auth/login",
        json={"email": medico_com_senha.email, "senha": "errada"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_recuperar_senha_retorna_200(client_auth):
    response = await client_auth.post(
        "/api/auth/recuperar-senha",
        json={"email": "inexistente@test.com"},
    )
    assert response.status_code == 200
    assert "instruções" in response.json()["message"].lower()


@pytest.mark.asyncio
async def test_rota_protegida_sem_token_retorna_401(client_auth):
    response = await client_auth.get("/api/pacientes")
    assert response.status_code == 401
