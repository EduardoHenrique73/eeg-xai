"""Extração de sinais EEG para visualização no frontend (downsampling)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mne

from app.ai_engine.feature_extractor import carregar_sinais_edf, selecionar_canais_eeg_validos

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

    if start_seconds is not None and end_seconds is not None:
        raw = mne.io.read_raw_edf(Path(arquivo_path), preload=False, verbose=False)
        canais_eeg = selecionar_canais_eeg_validos(raw, canais_selecionados)
        raw.pick(canais_eeg)
        taxa_hz = float(raw.info["sfreq"])
        limite = raw.n_times
        if max_duration_seconds is not None and max_duration_seconds > 0:
            limite = min(limite, int(max_duration_seconds * taxa_hz))
        indice_inicio = int(np.floor(start_seconds * taxa_hz))
        indice_fim = min(int(np.ceil(end_seconds * taxa_hz)), limite)
        if indice_fim - indice_inicio < 2:
            raise ValueError("Recorte temporal fora do exame ou curto demais.")
        sinais = np.asarray(raw.get_data(start=indice_inicio, stop=indice_fim), dtype=float)
    else:
        sinais, taxa_hz, canais_eeg = carregar_sinais_edf(
            arquivo_path,
            max_duration_seconds=max_duration_seconds,
            canais_selecionados=canais_selecionados,
        )
        indice_inicio = 0
        indice_fim = sinais.shape[1]
    if indice_fim - indice_inicio < 2:
        raise ValueError("Recorte temporal fora do exame ou curto demais.")
    # MNE retorna Volts (SI); converter para µV para exibição clínica.
    sinais_uv = sinais * 1e6
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
