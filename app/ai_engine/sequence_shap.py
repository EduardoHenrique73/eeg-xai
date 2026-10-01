"""Explicabilidade temporal e por canal para o CNN-LSTM sequencial."""

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


def gerar_mapa_shap_sequencial(
    modelo: Any,
    background: np.ndarray,
    amostra: np.ndarray,
    *,
    exame_id: int,
    canais: list[str],
    inicio_seconds: float,
    fim_seconds: float,
    output_dir: Path | None = None,
) -> str:
    """Gera heatmap Gradient SHAP agregado por passo temporal e canal."""
    if amostra.ndim != 3 or amostra.shape[0] != 1:
        raise ValueError("A amostra SHAP deve ter shape (1, sequencia, features).")
    if background.ndim != 3 or background.shape[1:] != amostra.shape[1:]:
        raise ValueError("Background SHAP incompativel com a amostra.")

    explainer = shap.GradientExplainer(modelo, background)
    valores = explainer.shap_values(amostra)
    if isinstance(valores, list):
        valores = valores[0]
    matriz = np.asarray(valores, dtype=np.float32)
    while matriz.ndim > 3 and matriz.shape[-1] == 1:
        matriz = matriz[..., 0]
    matriz = np.squeeze(matriz, axis=0)
    if matriz.ndim != 2:
        raise ValueError(f"Saida SHAP inesperada: {matriz.shape}")

    n_passos, n_features = matriz.shape
    if canais and n_features % len(canais) == 0:
        por_canal = np.abs(matriz).reshape(n_passos, len(canais), -1).mean(axis=2).T
        rotulos = canais
    else:
        por_canal = np.abs(matriz).mean(axis=1, keepdims=True).T
        rotulos = ["Sinal agregado"]

    destino = output_dir or get_settings().shap_storage_path
    destino.mkdir(parents=True, exist_ok=True)
    caminho = (destino / f"exame_{exame_id}_sequence_shap.png").resolve()

    fig_height = max(3.5, min(9.0, 1.8 + 0.25 * len(rotulos)))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    imagem = ax.imshow(por_canal, aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_yticks(np.arange(len(rotulos)))
    ax.set_yticklabels(rotulos, fontsize=8)
    ax.set_xticks(np.arange(n_passos))
    ax.set_xticklabels([str(i + 1) for i in range(n_passos)])
    ax.set_xlabel("Janela dentro da sequencia")
    ax.set_ylabel("Canal EEG")
    ax.set_title(
        f"Gradient SHAP do trecho de pico ({inicio_seconds:.1f}s a {fim_seconds:.1f}s)"
    )
    fig.colorbar(imagem, ax=ax, label="|valor SHAP| medio")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Mapa SHAP sequencial salvo em %s", caminho)
    return str(caminho)
