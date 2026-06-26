"""Testes do pipeline assíncrono de processamento IA."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.ai_engine.feature_extractor import extrair_features_de_valores
from app.ai_engine.inference import criar_modelo_cnn_lstm_dummy, limpar_cache_modelos
from app.config import get_settings
from app.models import Exame, PredicaoIA
from app.services.exame_pipeline import processar_exame_ia


@pytest.fixture
def features_fake() -> dict:
    return extrair_features_de_valores(np.linspace(-1.0, 1.0, 50))


@pytest.fixture
def modelo_dummy_path(tmp_path):
    limpar_cache_modelos()
    caminho = tmp_path / "modelo_pipeline.keras"
    criar_modelo_cnn_lstm_dummy().save(caminho)
    yield caminho
    limpar_cache_modelos()


@pytest_asyncio.fixture
async def exame_com_arquivo(db_session, paciente, tmp_path) -> Exame:
    """Exame com arquivo .edf mínimo no disco (conteúdo irrelevante — IA mockada)."""
    arquivo = tmp_path / "exame_pipeline.edf"
    arquivo.write_bytes(b"0       EDF+MOCK")

    exame = Exame(
        id_paciente=paciente.id,
        taxa_amostragem=256.0,
        arquivo_path=str(arquivo.resolve()),
    )
    db_session.add(exame)
    await db_session.commit()
    await db_session.refresh(exame)
    return exame


@pytest.mark.asyncio
async def test_processar_exame_ia_salva_predicao_com_mapa_shap(
    async_engine,
    exame_com_arquivo,
    features_fake,
    modelo_dummy_path,
    tmp_path,
    monkeypatch,
):
    """Pipeline persiste PredicaoIA com resultado_score e caminho SHAP real."""
    session_factory = async_sessionmaker(async_engine, expire_on_commit=False)
    settings = get_settings()
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    monkeypatch.setattr(settings, "keras_model_path", modelo_dummy_path)
    monkeypatch.setattr(settings, "shap_storage_path", shap_dir)
    monkeypatch.setattr(settings, "app_env", "production")

    mapa_esperado = str((shap_dir / f"exame_{exame_com_arquivo.id}_shap.png").resolve())

    mock_extrair = (
        lambda arquivo_path, max_duration_seconds=None, canais_selecionados=None, **kwargs: features_fake
    )
    mock_inferir = AsyncMock(return_value=0.73)
    mock_shap = lambda modelo, vetor, nomes, exame_id, **kwargs: mapa_esperado

    with patch("app.services.exame_pipeline.AsyncSessionLocal", session_factory):
        with patch("app.services.exame_pipeline.gerar_mapa_shap", side_effect=mock_shap):
            await processar_exame_ia(
                exame_com_arquivo.id,
                extrair_features=mock_extrair,
                inferir=mock_inferir,
            )

    async with session_factory() as session:
        result = await session.execute(
            select(PredicaoIA).where(PredicaoIA.id_exame == exame_com_arquivo.id)
        )
        predicao = result.scalar_one()

    assert predicao.resultado_score == pytest.approx(0.73)
    assert predicao.mapa_shap_path == mapa_esperado
    assert predicao.mapa_shap_path.endswith("_shap.png")
    assert predicao.detalhes_json is not None
    mock_inferir.assert_awaited_once()


@pytest.mark.asyncio
async def test_processar_exame_inexistente_nao_levanta_erro(async_engine, features_fake):
    session_factory = async_sessionmaker(async_engine, expire_on_commit=False)
    mock_extrair = (
        lambda arquivo_path, max_duration_seconds=None, canais_selecionados=None, **kwargs: features_fake
    )

    with patch("app.services.exame_pipeline.AsyncSessionLocal", session_factory):
        await processar_exame_ia(
            99999,
            extrair_features=mock_extrair,
            inferir=AsyncMock(return_value=0.5),
            gerar_shap=lambda *args, **kwargs: "/tmp/shap.png",
        )
