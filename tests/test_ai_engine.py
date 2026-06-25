"""
Testes de integração do motor de IA com exames EDF reais (dataset_amostra/).
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from app.ai_engine.feature_extractor import FEATURE_NAMES, extrair_features_de_valores, extrair_features_edf
from app.ai_engine.inference import (
    criar_modelo_cnn_lstm_dummy,
    limpar_cache_modelos,
    preparar_tensor_entrada,
    realizar_inferencia,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset_amostra"
EDF_CHB01_03 = DATASET_DIR / "chb01_03.edf"
EDF_CHB01_01 = DATASET_DIR / "chb01_01.edf"


@pytest.fixture
def features_mock() -> dict:
    """Dicionário de 19 features simulando saída do feature_extractor."""
    sinal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    return extrair_features_de_valores(sinal)


@pytest.fixture
def modelo_dummy_path(tmp_path: Path) -> Path:
    """Salva modelo CNN-LSTM dummy em disco para testes de inferência."""
    limpar_cache_modelos()
    caminho = tmp_path / "modelo_dummy.keras"
    model = criar_modelo_cnn_lstm_dummy()
    model.save(caminho)
    yield caminho
    limpar_cache_modelos()


def _resolver_edf_real() -> Path:
    """Resolve o arquivo EDF real disponível para testes de integração."""
    if EDF_CHB01_03.exists():
        return EDF_CHB01_03.resolve()
    if EDF_CHB01_01.exists():
        return EDF_CHB01_01.resolve()
    pytest.skip(
        "Coloque um arquivo .edf em dataset_amostra/ (ex.: chb01_03.edf) para rodar este teste."
    )


class TestIntegracaoEdfReal:
    """Integração com gravações clínicas reais (CHB-MIT)."""

    def test_extrair_features_edf_arquivo_real(self) -> None:
        caminho = _resolver_edf_real()

        tracemalloc.start()
        try:
            features = extrair_features_edf(
                str(caminho),
                max_duration_seconds=120.0,
            )
        finally:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        assert isinstance(features, dict)

        for nome in FEATURE_NAMES:
            assert nome in features
            assert isinstance(features[nome], (int, float))

        assert isinstance(features["entropia_shannon"], float)
        assert 0.0 <= features["entropia_shannon"] <= 1.0
        assert features["limiar"] == pytest.approx(features["media_valores"])
        assert features["total_amostras"] == features["total_amostras_brutas"]
        assert features["total_amostras"] > 0
        assert features["total_padroes"] > 0
        assert features["n_canais_eeg"] >= 1
        assert features["arquivo_path"] == str(caminho)
        assert features["taxa_amostragem"] > 0
        assert len(features["feature_vector"]) == 19

        # Pico de memória razoável (< 512 MB) para janela de 2 minutos
        assert peak < 512 * 1024 * 1024, (
            f"Pico de memória excessivo: {peak / 1024 / 1024:.1f} MB"
        )

    def test_chb01_03_quando_disponivel(self) -> None:
        if not EDF_CHB01_03.exists():
            pytest.skip("dataset_amostra/chb01_03.edf ainda não foi adicionado ao projeto.")

        features = extrair_features_edf(str(EDF_CHB01_03.resolve()), max_duration_seconds=120.0)

        assert features["entropia_shannon"] > 0.0
        assert features["desvio_padrao"] >= 0.0
        assert features["variancia"] >= 0.0


class TestInferenciaCnnLstm:
    """Testes do módulo inference.py com modelo Keras dummy."""

    def test_preparar_tensor_formato_3d(self, features_mock: dict) -> None:
        tensor = preparar_tensor_entrada(features_mock)
        assert tensor.shape == (1, 19, 1)
        assert tensor.dtype == np.float32

    @pytest.mark.asyncio
    async def test_realizar_inferencia_retorna_float_valido(
        self,
        features_mock: dict,
        modelo_dummy_path: Path,
    ) -> None:
        score = await realizar_inferencia(features_mock, str(modelo_dummy_path))

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_cache_modelo_reutiliza_mesma_instancia(
        self,
        features_mock: dict,
        modelo_dummy_path: Path,
    ) -> None:
        from app.ai_engine.inference import _carregar_modelo_cached

        limpar_cache_modelos()
        caminho = str(modelo_dummy_path)

        primeiro = _carregar_modelo_cached(caminho)
        segundo = _carregar_modelo_cached(caminho)

        assert primeiro is segundo

        score = await realizar_inferencia(features_mock, caminho)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_modelo_inexistente_levanta_erro(self, features_mock: dict) -> None:
        limpar_cache_modelos()
        with pytest.raises(FileNotFoundError):
            await realizar_inferencia(features_mock, "/caminho/modelo_inexistente.keras")
