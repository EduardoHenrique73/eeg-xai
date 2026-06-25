"""Testes das rotas de autenticação."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.deps import get_current_user
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
    assert payload["medico"]["threshold_confianca"] == pytest.approx(0.5)
    assert payload["medico"]["montagem_padrao"] == []
    assert payload["medico"]["exibir_shap"] is True


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
async def test_atualizar_perfil_e_preferencias_medico(client_auth, medico_com_senha):
    async def override_current_user():
        return medico_com_senha

    app.dependency_overrides[get_current_user] = override_current_user

    response = await client_auth.patch(
        "/api/auth/me",
        json={
            "nome": "Dra. Config",
            "email": "config@test.com",
            "crm": "123456-GO",
            "threshold_confianca": 0.72,
            "montagem_padrao": ["FP1-F7", "F7-T7"],
            "exibir_shap": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["nome"] == "Dra. Config"
    assert payload["email"] == "config@test.com"
    assert payload["crm"] == "123456-GO"
    assert payload["threshold_confianca"] == pytest.approx(0.72)
    assert payload["montagem_padrao"] == ["FP1-F7", "F7-T7"]
    assert payload["exibir_shap"] is False
    app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.asyncio
async def test_rota_protegida_sem_token_retorna_401(client_auth):
    response = await client_auth.get("/api/pacientes")
    assert response.status_code == 401
