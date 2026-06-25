"""Testes das rotas REST (ingestão clínica)."""

from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import get_settings
from app.core.deps import get_current_user
from app.database import get_db
from app.main import app
from app.models import Exame, PredicaoIA


@pytest.fixture
def edf_storage_tmp(tmp_path, monkeypatch):
    """Diretório temporário para arquivos .edf e SHAP durante os testes."""
    storage = tmp_path / "edf"
    shap = tmp_path / "shap"
    storage.mkdir(parents=True, exist_ok=True)
    shap.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    monkeypatch.setattr(settings, "edf_storage_path", storage)
    monkeypatch.setattr(settings, "shap_storage_path", shap)
    monkeypatch.setattr(settings, "storage_root", tmp_path)
    return storage


@pytest_asyncio.fixture
async def client(async_engine, edf_storage_tmp, usuario_medico):
    """Cliente HTTP com banco e storage de teste injetados."""
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
    mock_processar = AsyncMock()

    with patch("app.api.routes.exames.processar_exame_ia", mock_processar):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            yield http_client, mock_processar

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_extensao_invalida_retorna_415(client, paciente):
    http_client, _ = client
    response = await http_client.post(
        "/api/exames/upload",
        data={"paciente_id": str(paciente.id), "taxa_amostragem": "256"},
        files={"arquivo": ("sinal.txt", BytesIO(b"dados invalidos"), "text/plain")},
    )

    assert response.status_code == 415
    assert "edf" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_edf_valido_retorna_202_e_persiste(client, paciente, db_session, edf_storage_tmp):
    http_client, mock_processar = client
    paciente_id = paciente.id
    conteudo = b"0       EDF+DUMMY"

    response = await http_client.post(
        "/api/exames/upload",
        data={"paciente_id": str(paciente_id), "taxa_amostragem": "512"},
        files={"arquivo": ("exame_clinico.edf", BytesIO(conteudo), "application/octet-stream")},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["message"] == (
        "Exame recebido com sucesso. Selecione os canais e solicite a análise IA."
    )
    assert "exame_id" in payload
    assert "arquivo_path" in payload

    db_session.expire_all()
    result = await db_session.execute(
        select(Exame).where(Exame.id == payload["exame_id"])
    )
    exame = result.scalar_one()

    assert exame.id_paciente == paciente_id
    assert exame.taxa_amostragem == 512.0
    assert exame.arquivo_path == payload["arquivo_path"]

    nome_arquivo = exame.arquivo_path.replace("\\", "/").split("/")[-1]
    arquivo_salvo = edf_storage_tmp / nome_arquivo
    assert arquivo_salvo.exists()
    assert arquivo_salvo.read_bytes() == conteudo

    mock_processar.assert_not_awaited()


@pytest.mark.asyncio
async def test_solicitar_analise_ia_retorna_202(client, exame, db_session, tmp_path):
    http_client, mock_processar = client
    arquivo = tmp_path / "sinal.edf"
    arquivo.write_bytes(b"EDF")
    exame.arquivo_path = str(arquivo.resolve())
    await db_session.commit()

    with patch(
        "app.api.routes.exames.listar_canais_eeg_edf",
        return_value=["FP1", "F7", "O1"],
    ):
        response = await http_client.post(
            f"/api/exames/{exame.id}/analise",
            json={"canais_selecionados": ["FP1", "F7"]},
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["exame_id"] == exame.id
    assert payload["status"] == "em_processamento"
    assert payload["canais_processados"] == ["FP1", "F7"]
    mock_processar.assert_awaited_once_with(
        exame.id,
        canais_selecionados=["FP1", "F7"],
    )


@pytest.mark.asyncio
async def test_solicitar_analise_sem_canais_retorna_422(client, exame, db_session, tmp_path):
    http_client, _ = client
    arquivo = tmp_path / "sinal.edf"
    arquivo.write_bytes(b"EDF")
    exame.arquivo_path = str(arquivo.resolve())
    await db_session.commit()

    response = await http_client.post(
        f"/api/exames/{exame.id}/analise",
        json={"canais_selecionados": []},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_solicitar_analise_canal_invalido_retorna_422(client, exame, db_session, tmp_path):
    http_client, _ = client
    arquivo = tmp_path / "sinal.edf"
    arquivo.write_bytes(b"EDF")
    exame.arquivo_path = str(arquivo.resolve())
    await db_session.commit()

    with patch(
        "app.api.routes.exames.listar_canais_eeg_edf",
        return_value=["FP1"],
    ):
        response = await http_client.post(
            f"/api/exames/{exame.id}/analise",
            json={"canais_selecionados": ["CANAL_INEXISTENTE"]},
        )

    assert response.status_code == 422
    assert "inválidos" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_paciente_inexistente_retorna_404(client):
    http_client, _ = client
    response = await http_client.post(
        "/api/exames/upload",
        data={"paciente_id": "99999", "taxa_amostragem": "256"},
        files={"arquivo": ("exame.edf", BytesIO(b"EDF"), "application/octet-stream")},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_diagnostico_exame_inexistente_retorna_404(client):
    http_client, _ = client
    response = await http_client.get("/api/exames/99999/diagnostico")

    assert response.status_code == 404
    assert "não encontrado" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_diagnostico_em_processamento_retorna_206(client, exame):
    http_client, _ = client
    response = await http_client.get(f"/api/exames/{exame.id}/diagnostico")

    assert response.status_code == 206
    payload = response.json()
    assert payload["status"] == "em_processamento"
    assert payload["exame_id"] == exame.id
    assert payload["id_paciente"] == exame.id_paciente
    assert payload["taxa_amostragem"] == exame.taxa_amostragem
    assert "data_upload" in payload
    assert "resultado_score" not in payload
    assert "mapa_shap_url" not in payload


@pytest.mark.asyncio
async def test_diagnostico_concluido_retorna_200_com_url_shap(
    client,
    exame,
    db_session,
    edf_storage_tmp,
):
    settings = get_settings()
    mapa_interno = settings.shap_storage_path / f"exame_{exame.id}_shap.png"
    mapa_interno.write_bytes(b"PNG-FAKE")

    predicao = PredicaoIA(
        id_exame=exame.id,
        resultado_score=0.82,
        mapa_shap_path=str(mapa_interno.resolve()),
    )
    db_session.add(predicao)
    await db_session.commit()

    http_client, _ = client
    response = await http_client.get(f"/api/exames/{exame.id}/diagnostico")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "concluido"
    assert payload["exame_id"] == exame.id
    assert payload["resultado_score"] == pytest.approx(0.82)
    assert payload["classificacao_clinica"] == "Crise Epiléptica"
    assert payload["mapa_shap_url"] == f"/static/shap/exame_{exame.id}_shap.png"
    assert payload["mapa_shap_url"].startswith("/static/")
    assert payload["mapa_shap_url"].endswith("_shap.png")
    assert "data_analise" in payload


@pytest.mark.asyncio
async def test_diagnostico_score_baixo_classificacao_normal(client, exame, db_session):
    predicao = PredicaoIA(
        id_exame=exame.id,
        resultado_score=0.3,
        mapa_shap_path=str(get_settings().shap_storage_path / f"exame_{exame.id}_shap.png"),
    )
    db_session.add(predicao)
    await db_session.commit()

    http_client, _ = client
    response = await http_client.get(f"/api/exames/{exame.id}/diagnostico")

    assert response.status_code == 200
    assert response.json()["classificacao_clinica"] == "Atividade Normal"


@pytest.mark.asyncio
async def test_obter_sinais_exame_retorna_pontos_downsampled(
    client,
    exame,
    db_session,
    tmp_path,
):
    arquivo = tmp_path / "sinal.edf"
    arquivo.write_bytes(b"EDF")
    exame.arquivo_path = str(arquivo.resolve())
    await db_session.commit()

    pontos_fake = [
        {"tempo": 0.0, "amplitude": 1.5},
        {"tempo": 0.1, "amplitude": -2.0},
    ]
    dados_fake = {
        "pontos": pontos_fake,
        "taxa_amostragem_hz": 256.0,
        "n_canais_eeg": 3,
        "canais_eeg": ["FP1", "F7", "O1"],
        "n_pontos_original": 5000,
        "n_pontos_retornados": 2,
    }

    http_client, _ = client
    with patch(
        "app.api.routes.exames.extrair_sinais_para_visualizacao",
        return_value=dados_fake,
    ):
        response = await http_client.get(f"/api/exames/{exame.id}/sinais")

    assert response.status_code == 200
    payload = response.json()
    assert payload["exame_id"] == exame.id
    assert len(payload["pontos"]) == 2
    assert payload["pontos"][0]["amplitude"] == 1.5
    assert payload["n_pontos_original"] == 5000
    assert payload["canais_eeg"] == ["FP1", "F7", "O1"]


@pytest.mark.asyncio
async def test_obter_sinais_exame_inexistente_retorna_404(client):
    http_client, _ = client
    response = await http_client.get("/api/exames/99999/sinais")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_salvar_laudo_exame_atualiza_status_e_texto(client, exame, db_session):
    http_client, _ = client
    response = await http_client.patch(
        f"/api/exames/{exame.id}/laudo",
        json={"laudo_texto": "Paciente apresenta padrão compatível com atividade interictal."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status_exame"] == "concluido"
    assert "interictal" in payload["laudo_texto"]

    db_session.expire_all()
    await db_session.refresh(exame)
    assert exame.status_exame == "concluido"
    assert exame.laudo_texto is not None


@pytest.mark.asyncio
async def test_salvar_laudo_exame_ja_emitido_retorna_409(client, exame, db_session):
    exame.laudo_texto = "Laudo anterior."
    exame.status_exame = "concluido"
    await db_session.commit()

    http_client, _ = client
    response = await http_client.patch(
        f"/api/exames/{exame.id}/laudo",
        json={"laudo_texto": "Tentativa de alteração."},
    )

    assert response.status_code == 409

