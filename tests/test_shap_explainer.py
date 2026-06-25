"""Testes do módulo SHAP (XAI)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ai_engine.feature_extractor import FEATURE_NAMES, extrair_features_de_valores
from app.ai_engine.inference import criar_modelo_cnn_lstm_dummy
from app.ai_engine.shap_explainer import gerar_mapa_shap


@pytest.fixture
def feature_vector() -> np.ndarray:
    return extrair_features_de_valores(np.linspace(-2.0, 2.0, 40))["feature_vector"]


@pytest.fixture
def modelo_dummy():
    return criar_modelo_cnn_lstm_dummy()


def test_gerar_mapa_shap_cria_png_com_kernel_explainer_mock(
    modelo_dummy,
    feature_vector,
    tmp_path: Path,
) -> None:
    """Garante que o PNG é criado fisicamente na pasta correta."""
    shap_fake = np.random.randn(len(FEATURE_NAMES)).astype(np.float32)
    explainer_mock = MagicMock()
    explainer_mock.shap_values.return_value = shap_fake
    explainer_mock.expected_value = 0.5

    with patch("app.ai_engine.shap_explainer.shap.KernelExplainer", return_value=explainer_mock):
        caminho = gerar_mapa_shap(
            modelo_dummy,
            np.asarray(feature_vector, dtype=np.float32),
            FEATURE_NAMES,
            exame_id=42,
            output_dir=tmp_path,
            nsamples=10,
        )

    arquivo = Path(caminho)
    assert arquivo.exists()
    assert arquivo.parent == tmp_path.resolve()
    assert arquivo.name == "exame_42_shap.png"
    assert arquivo.stat().st_size > 0
    explainer_mock.shap_values.assert_called_once()


def test_gerar_mapa_shap_vetor_tamanho_invalido(modelo_dummy, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="feature_vector deve ter"):
        gerar_mapa_shap(
            modelo_dummy,
            np.zeros(5, dtype=np.float32),
            FEATURE_NAMES,
            exame_id=1,
            output_dir=tmp_path,
        )
