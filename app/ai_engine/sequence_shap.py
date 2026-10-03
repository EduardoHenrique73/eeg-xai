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


def _resumir_por_canal(
    matriz: np.ndarray,
    canais: list[str],
    janelas: list[dict[str, float]],
    canais_omitidos: list[str],
) -> tuple[np.ndarray, dict[str, Any] | None]:
    n_passos, n_features = matriz.shape
    positivos = np.maximum(matriz, 0)
    if not canais or len(janelas) != n_passos or n_features % len(canais):
        return positivos.mean(axis=1, keepdims=True).T, None

    por_canal = positivos.reshape(n_passos, len(canais), -1).mean(axis=2).T
    indices_visiveis = [indice for indice, canal in enumerate(canais) if canal not in canais_omitidos]
    if not indices_visiveis:
        return por_canal, None
    atribuicoes: dict[tuple[str, float, float], float] = {}
    for indice_canal, canal in enumerate(canais):
        if canal in canais_omitidos:
            continue
        for indice_passo, janela in enumerate(janelas):
            inicio = float(janela["start_seconds"])
            fim = float(janela["end_seconds"])
            if not np.isfinite(inicio) or not np.isfinite(fim) or fim <= inicio:
                raise ValueError("Janela temporal SHAP invalida.")
            chave = (canal, inicio, fim)
            atribuicoes[chave] = atribuicoes.get(chave, 0.0) + float(
                por_canal[indice_canal, indice_passo]
            )

    maior = max(atribuicoes.values(), default=0.0)
    if maior <= 0:
        return por_canal, None
    celulas = [
        {
            "canal": canal,
            "start_seconds": inicio,
            "end_seconds": fim,
            "intensity": valor / maior,
        }
        for (canal, inicio, fim), valor in atribuicoes.items()
    ]

    return por_canal, {
        "scope": "peak_sequence",
        "basis": "window_features",
        "cells": celulas,
    }


def gerar_mapa_shap_sequencial(
    modelo: Any,
    background: np.ndarray,
    amostra: np.ndarray,
    *,
    exame_id: int,
    canais: list[str],
    janelas: list[dict[str, float]],
    canais_omitidos: list[str] | None = None,
    inicio_seconds: float,
    fim_seconds: float,
    output_dir: Path | None = None,
) -> tuple[str, dict[str, Any] | None]:
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

    n_passos = matriz.shape[0]
    por_canal, overlay = _resumir_por_canal(matriz, canais, janelas, canais_omitidos or [])
    if canais and por_canal.shape[0] == len(canais):
        rotulos = canais
    else:
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
    ax.set_xticklabels(
        [f"{janela['start_seconds']:.0f}-{janela['end_seconds']:.0f}" for janela in janelas]
        if len(janelas) == n_passos else [str(i + 1) for i in range(n_passos)],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Intervalo da janela (s)")
    ax.set_ylabel("Canal EEG")
    ax.set_title(
        f"Gradient SHAP da sequencia de pico (alvo: {inicio_seconds:.1f}s a {fim_seconds:.1f}s)"
    )
    fig.colorbar(imagem, ax=ax, label="SHAP positivo medio")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Mapa SHAP sequencial salvo em %s", caminho)
    return str(caminho), overlay
