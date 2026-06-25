"""Extração de sinais EEG para visualização no frontend (downsampling)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.ai_engine.feature_extractor import carregar_sinal_edf, listar_canais_eeg_edf

MAX_PONTOS_VISUALIZACAO = 1500


def extrair_sinais_para_visualizacao(
    arquivo_path: str | Path,
    *,
    max_pontos: int = MAX_PONTOS_VISUALIZACAO,
    max_duration_seconds: float | None = None,
) -> dict[str, object]:
    """
    Lê o .edf, média dos canais EEG e reduz para ~1500 pontos (Recharts).

    Returns:
        Dict com lista `pontos` [{tempo, amplitude}, ...] e metadados.
    """
    if max_pontos < 2:
        raise ValueError("max_pontos deve ser >= 2")

    sinal, taxa_hz, n_canais = carregar_sinal_edf(
        arquivo_path,
        max_duration_seconds=max_duration_seconds,
    )
    canais_eeg = listar_canais_eeg_edf(arquivo_path)
    # MNE retorna Volts (SI); converter para µV para exibição clínica
    sinal_uv = sinal * 1e6
    n_original = int(sinal_uv.shape[0])

    if n_original > max_pontos:
        indices = np.linspace(0, n_original - 1, max_pontos, dtype=int)
        amostras = sinal_uv[indices]
        tempos = indices.astype(np.float64) / taxa_hz
    else:
        amostras = sinal_uv
        tempos = np.arange(n_original, dtype=np.float64) / taxa_hz

    pontos = [
        {
            "tempo": round(float(t), 4),
            "amplitude": round(float(a), 2),
        }
        for t, a in zip(tempos, amostras, strict=True)
    ]

    return {
        "pontos": pontos,
        "taxa_amostragem_hz": taxa_hz,
        "n_canais_eeg": n_canais,
        "canais_eeg": canais_eeg,
        "n_pontos_original": n_original,
        "n_pontos_retornados": len(pontos),
    }
