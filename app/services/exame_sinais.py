"""Extração de sinais EEG para visualização no frontend (downsampling)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.ai_engine.feature_extractor import carregar_sinais_edf

MAX_PONTOS_VISUALIZACAO = 1500


def extrair_sinais_para_visualizacao(
    arquivo_path: str | Path,
    *,
    max_pontos: int = MAX_PONTOS_VISUALIZACAO,
    max_duration_seconds: float | None = None,
    canais_selecionados: list[str] | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> dict[str, object]:
    """
    Le o .edf preservando os canais EEG e reduz cada serie para o frontend.

    Returns:
        Dict com lista `pontos` [{tempo, amplitude}, ...] e metadados.
    """
    if max_pontos < 2:
        raise ValueError("max_pontos deve ser >= 2")
    if (start_seconds is None) != (end_seconds is None):
        raise ValueError("Informe inicio e fim do recorte temporal.")
    if start_seconds is not None and end_seconds is not None:
        if (
            not np.isfinite(start_seconds)
            or not np.isfinite(end_seconds)
            or start_seconds < 0
            or end_seconds <= start_seconds
            or end_seconds - start_seconds > 120
        ):
            raise ValueError("Recorte temporal deve ter inicio >= 0 e duracao de ate 120 segundos.")

    sinais, taxa_hz, canais_eeg = carregar_sinais_edf(
        arquivo_path,
        max_duration_seconds=max_duration_seconds,
        canais_selecionados=canais_selecionados,
    )
    # MNE retorna Volts (SI); converter para µV para exibição clínica
    indice_inicio = int(np.floor(start_seconds * taxa_hz)) if start_seconds is not None else 0
    indice_fim = int(np.ceil(end_seconds * taxa_hz)) if end_seconds is not None else sinais.shape[1]
    indice_fim = min(indice_fim, sinais.shape[1])
    if indice_fim - indice_inicio < 2:
        raise ValueError("Recorte temporal fora do exame ou curto demais.")
    sinais_uv = sinais[:, indice_inicio:indice_fim] * 1e6
    n_original = int(sinais_uv.shape[1])

    if n_original > max_pontos:
        indices = np.linspace(0, n_original - 1, max_pontos, dtype=int)
        amostras = sinais_uv[:, indices]
        tempos = (indices.astype(np.float64) + indice_inicio) / taxa_hz
    else:
        amostras = sinais_uv
        tempos = (np.arange(n_original, dtype=np.float64) + indice_inicio) / taxa_hz

    series = [
        {
            "canal": canal,
            "pontos": [
                {
                    "tempo": round(float(t), 4),
                    "amplitude": round(float(a), 2),
                }
                for t, a in zip(tempos, amostras[indice_canal], strict=True)
            ],
        }
        for indice_canal, canal in enumerate(canais_eeg)
    ]
    sinal_medio = np.mean(amostras, axis=0)
    pontos = [
        {
            "tempo": round(float(t), 4),
            "amplitude": round(float(a), 2),
        }
        for t, a in zip(tempos, sinal_medio, strict=True)
    ]

    return {
        "pontos": pontos,
        "series": series,
        "taxa_amostragem_hz": taxa_hz,
        "n_canais_eeg": len(canais_eeg),
        "canais_eeg": canais_eeg,
        "n_pontos_original": n_original,
        "n_pontos_retornados": len(pontos),
    }
