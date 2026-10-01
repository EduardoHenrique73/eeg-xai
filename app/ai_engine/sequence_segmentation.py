"""Utilitarios para segmentacao temporal com sequencias sobrepostas."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from app.ai_engine.training import SeizureInterval


def _sobreposicao(
    inicio: float,
    fim: float,
    intervalos: list[SeizureInterval],
) -> float:
    return float(
        sum(
            max(0.0, min(fim, intervalo.end_seconds) - max(inicio, intervalo.start_seconds))
            for intervalo in intervalos
        )
    )


def construir_alvos_temporais(
    metas_sequencias: list[dict[str, Any]],
    intervalos_por_arquivo: dict[str, list[SeizureInterval]],
    *,
    sequence_length: int,
    window_seconds: float,
    step_seconds: float,
    min_ictal_overlap_ratio: float,
) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    """Reconstrui um rotulo e metadados para cada passo da sequencia."""
    alvos: list[list[int]] = []
    metas_temporais: list[list[dict[str, Any]]] = []
    for meta in metas_sequencias:
        arquivo = str(meta["arquivo"])
        inicio_contexto = float(meta["context_start_seconds"])
        intervalos = intervalos_por_arquivo.get(arquivo, [])
        labels_bloco: list[int] = []
        metas_bloco: list[dict[str, Any]] = []
        for passo in range(sequence_length):
            inicio = inicio_contexto + passo * step_seconds
            fim = inicio + window_seconds
            overlap = _sobreposicao(inicio, fim, intervalos)
            ratio = overlap / window_seconds if window_seconds > 0 else 0.0
            label = int(ratio >= min_ictal_overlap_ratio)
            contexto = "ictal" if label else ("boundary" if overlap > 0 else "interictal")
            evento_idx: int | None = None
            if label and intervalos:
                sobreposicoes = [
                    max(0.0, min(fim, item.end_seconds) - max(inicio, item.start_seconds))
                    for item in intervalos
                ]
                evento_idx = int(np.argmax(sobreposicoes))
            labels_bloco.append(label)
            metas_bloco.append(
                {
                    "arquivo": arquivo,
                    "paciente": meta.get("paciente"),
                    "start_seconds": float(inicio),
                    "end_seconds": float(fim),
                    "label": label,
                    "context": contexto,
                    "ictal_overlap_ratio": float(ratio),
                    "event_key": f"{arquivo}:{evento_idx}" if evento_idx is not None else None,
                }
            )
        alvos.append(labels_bloco)
        metas_temporais.append(metas_bloco)
    return np.asarray(alvos, dtype=np.float32)[..., np.newaxis], metas_temporais


def agregar_predicoes_temporais(
    scores: np.ndarray,
    alvos: np.ndarray,
    metas_temporais: list[list[dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Calcula a media das predicoes para janelas repetidas por sobreposicao."""
    scores = np.asarray(scores, dtype=np.float32)
    alvos = np.asarray(alvos, dtype=np.float32)
    if scores.shape != alvos.shape:
        raise ValueError("scores e alvos temporais devem ter a mesma forma.")

    acumulado: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    rotulos: dict[tuple[str, float, float], int] = {}
    metadados: dict[tuple[str, float, float], dict[str, Any]] = {}
    for seq_idx, metas_bloco in enumerate(metas_temporais):
        for passo, meta in enumerate(metas_bloco):
            chave = (
                str(meta["arquivo"]),
                float(meta["start_seconds"]),
                float(meta["end_seconds"]),
            )
            label = int(alvos[seq_idx, passo, 0])
            if chave in rotulos and rotulos[chave] != label:
                raise ValueError("Rotulos divergentes para a mesma janela temporal.")
            rotulos[chave] = label
            metadados[chave] = meta
            acumulado[chave].append(float(scores[seq_idx, passo, 0]))

    chaves = sorted(acumulado, key=lambda item: (item[0], item[1], item[2]))
    return (
        np.asarray([rotulos[chave] for chave in chaves], dtype=np.int64),
        np.asarray([np.mean(acumulado[chave]) for chave in chaves], dtype=np.float32),
        [metadados[chave] for chave in chaves],
    )


def pesos_temporais(
    alvos: np.ndarray,
    metas_temporais: list[list[dict[str, Any]]],
    *,
    boundary_weight: float,
    class_weight: dict[int, float] | None,
    balance_events: bool = False,
) -> np.ndarray:
    """Gera pesos por passo para bordas e desbalanceamento de classes."""
    pesos = np.ones(alvos.shape[:2], dtype=np.float32)
    for seq_idx, metas_bloco in enumerate(metas_temporais):
        for passo, meta in enumerate(metas_bloco):
            if meta.get("context") == "boundary":
                pesos[seq_idx, passo] *= boundary_weight
            if class_weight is not None:
                pesos[seq_idx, passo] *= float(class_weight[int(alvos[seq_idx, passo, 0])])
    if balance_events:
        contagens: dict[str, int] = defaultdict(int)
        for seq_idx, metas_bloco in enumerate(metas_temporais):
            for passo, meta in enumerate(metas_bloco):
                chave = meta.get("event_key")
                if int(alvos[seq_idx, passo, 0]) == 1 and chave:
                    contagens[str(chave)] += 1
        total = sum(contagens.values())
        if contagens and total > 0:
            peso_medio = total / len(contagens)
            for seq_idx, metas_bloco in enumerate(metas_temporais):
                for passo, meta in enumerate(metas_bloco):
                    chave = meta.get("event_key")
                    if int(alvos[seq_idx, passo, 0]) == 1 and chave:
                        pesos[seq_idx, passo] *= float(peso_medio / contagens[str(chave)])
    return pesos
