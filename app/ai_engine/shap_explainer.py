"""
Gerador de explicações SHAP — Explainable AI (XAI) clínico.

Calcula o impacto das 19 features na predição CNN-LSTM e persiste o mapa em disco.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_NSAMPLES = 100


def _predicao_modelo(modelo: Any, features_2d: np.ndarray) -> np.ndarray:
    """
    Função de predição para o SHAP — entrada (n_amostras, n_features).

    O modelo Keras espera (batch, n_features, 1).
    """
    tensor = features_2d.astype(np.float32).reshape(-1, features_2d.shape[1], 1)
    saida = modelo.predict(tensor, verbose=0)
    return np.asarray(saida).reshape(-1)


def gerar_mapa_shap(
    modelo: Any,
    feature_vector: np.ndarray,
    nomes_features: list[str],
    exame_id: int,
    *,
    output_dir: Path | None = None,
    nsamples: int = DEFAULT_NSAMPLES,
) -> str:
    """
    Calcula SHAP values e salva gráfico de impacto das features.

    Args:
        modelo: Modelo Keras CNN-LSTM carregado.
        feature_vector: Vetor 1D com 19 features (já escalonadas, se aplicável).
        nomes_features: Nomes legíveis das features (ordem do vetor).
        exame_id: ID do exame para nomear o arquivo.
        output_dir: Diretório de saída (padrão: settings.shap_storage_path).
        nsamples: Amostras do KernelExplainer (trade-off velocidade/precisão).

    Returns:
        Caminho absoluto do PNG salvo em storage/shap/.
    """
    settings = get_settings()
    destino_dir = output_dir or settings.shap_storage_path
    destino_dir.mkdir(parents=True, exist_ok=True)
    caminho_png = (destino_dir / f"exame_{exame_id}_shap.png").resolve()

    vetor = np.asarray(feature_vector, dtype=np.float32).ravel()
    if vetor.shape[0] != len(nomes_features):
        raise ValueError(
            f"feature_vector deve ter {len(nomes_features)} elementos, "
            f"recebido {vetor.shape[0]}"
        )

    amostra = vetor.reshape(1, -1)
    background = np.zeros((1, vetor.shape[0]), dtype=np.float32)

    logger.info("Calculando SHAP para exame %s (nsamples=%d)", exame_id, nsamples)
    explainer = shap.KernelExplainer(
        lambda data: _predicao_modelo(modelo, data),
        background,
    )
    shap_values = explainer.shap_values(amostra, nsamples=nsamples)

    valores_shap = np.asarray(shap_values)
    if valores_shap.ndim > 1:
        valores_shap = valores_shap[0] if valores_shap.shape[0] == 1 else valores_shap.flatten()[: len(nomes_features)]
    valores_shap = valores_shap.ravel()[: len(nomes_features)]

    _salvar_barplot_impacto(
        nomes_features=nomes_features,
        valores_shap=valores_shap,
        feature_vector=vetor,
        caminho_png=caminho_png,
        exame_id=exame_id,
    )

    logger.info("Mapa SHAP salvo em %s", caminho_png)
    return str(caminho_png)


def _salvar_barplot_impacto(
    *,
    nomes_features: list[str],
    valores_shap: np.ndarray,
    feature_vector: np.ndarray,
    caminho_png: Path,
    exame_id: int,
) -> None:
    """Gera barplot horizontal de impacto por feature (thread-safe, backend Agg)."""
    indices = np.argsort(np.abs(valores_shap))
    nomes_ordenados = [nomes_features[i] for i in indices]
    valores_ordenados = valores_shap[indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    cores = ["#d62728" if valor > 0 else "#1f77b4" for valor in valores_ordenados]
    ax.barh(nomes_ordenados, valores_ordenados, color=cores, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Impacto SHAP na probabilidade de crise")
    ax.set_title(f"Explicabilidade (XAI) — Exame #{exame_id}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


class SHAPExplainer:
    """Facade orientada a objetos para geração de mapas SHAP."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir

    def generate_heatmap(
        self,
        modelo: Any,
        feature_vector: np.ndarray,
        nomes_features: list[str],
        exame_id: int,
    ) -> Path:
        caminho = gerar_mapa_shap(
            modelo,
            feature_vector,
            nomes_features,
            exame_id,
            output_dir=self.output_dir,
        )
        return Path(caminho)
