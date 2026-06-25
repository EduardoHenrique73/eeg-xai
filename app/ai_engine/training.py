"""Pipeline simples de janelamento, data augmentation e k-fold.

Este modulo e propositalmente leve: ele cria janelas temporais sobre sinais EEG,
rotula cada janela por sobreposicao com intervalos de crise e avalia as 19
features ja usadas pelo sistema com validacao cruzada estratificada.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import mne
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.ai_engine.feature_extractor import FEATURE_NAMES, extrair_features_de_valores


@dataclass(frozen=True)
class SeizureInterval:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class WindowSpec:
    start_seconds: float
    end_seconds: float
    label: int


def parse_chbmit_summary(summary_path: str | Path) -> dict[str, list[SeizureInterval]]:
    """Extrai intervalos de crise do arquivo `chbXX-summary.txt` do CHB-MIT."""
    texto = Path(summary_path).read_text(encoding="utf-8", errors="ignore")
    resultado: dict[str, list[SeizureInterval]] = {}
    arquivo_atual: str | None = None
    inicio_atual: float | None = None

    for linha in texto.splitlines():
        nome = re.search(r"File Name:\s*(\S+)", linha)
        if nome:
            arquivo_atual = nome.group(1)
            resultado.setdefault(arquivo_atual, [])
            inicio_atual = None
            continue

        inicio = re.search(r"Seizure Start Time:\s*(\d+(?:\.\d+)?)\s*seconds", linha)
        if inicio and arquivo_atual:
            inicio_atual = float(inicio.group(1))
            continue

        fim = re.search(r"Seizure End Time:\s*(\d+(?:\.\d+)?)\s*seconds", linha)
        if fim and arquivo_atual and inicio_atual is not None:
            resultado[arquivo_atual].append(
                SeizureInterval(
                    start_seconds=inicio_atual,
                    end_seconds=float(fim.group(1)),
                )
            )
            inicio_atual = None

    return resultado


def window_overlaps_seizure(
    start_seconds: float,
    end_seconds: float,
    intervals: list[SeizureInterval],
) -> bool:
    """Retorna True quando a janela cruza qualquer intervalo de crise."""
    return any(
        start_seconds < interval.end_seconds and end_seconds > interval.start_seconds
        for interval in intervals
    )


def gerar_janelas_temporais(
    duration_seconds: float,
    intervals: list[SeizureInterval],
    *,
    window_seconds: float = 10.0,
    step_seconds: float = 5.0,
) -> list[WindowSpec]:
    """
    Gera janelas sobrepostas. Esse janelamento e a forma inicial de data augmentation.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds deve ser maior que zero.")
    if step_seconds <= 0:
        raise ValueError("step_seconds deve ser maior que zero.")
    if duration_seconds < window_seconds:
        return []

    specs: list[WindowSpec] = []
    start = 0.0
    while start + window_seconds <= duration_seconds:
        end = start + window_seconds
        specs.append(
            WindowSpec(
                start_seconds=start,
                end_seconds=end,
                label=1 if window_overlaps_seizure(start, end, intervals) else 0,
            )
        )
        start += step_seconds
    return specs


def _limitar_janelas_por_classe(
    specs: list[WindowSpec],
    max_windows_per_class: int | None,
) -> list[WindowSpec]:
    if max_windows_per_class is None:
        return specs

    selecionadas: list[WindowSpec] = []
    for label in (0, 1):
        classe = [spec for spec in specs if spec.label == label]
        if len(classe) <= max_windows_per_class:
            selecionadas.extend(classe)
            continue
        indices = np.linspace(0, len(classe) - 1, max_windows_per_class, dtype=int)
        selecionadas.extend(classe[int(i)] for i in indices)

    return sorted(selecionadas, key=lambda spec: spec.start_seconds)


def extrair_dataset_janelado_de_sinal(
    signal: np.ndarray,
    sfreq: float,
    intervals: list[SeizureInterval],
    *,
    window_seconds: float = 10.0,
    step_seconds: float = 5.0,
    max_windows_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extrai matriz X/y de features a partir de um sinal 1D ja carregado."""
    sinal = np.asarray(signal, dtype=float).ravel()
    duration_seconds = sinal.size / sfreq
    specs = gerar_janelas_temporais(
        duration_seconds,
        intervals,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    specs = _limitar_janelas_por_classe(specs, max_windows_per_class)

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    metadados: list[dict[str, Any]] = []

    for spec in specs:
        inicio = int(round(spec.start_seconds * sfreq))
        fim = int(round(spec.end_seconds * sfreq))
        janela = sinal[inicio:fim]
        if janela.size < 4:
            continue

        features = extrair_features_de_valores(janela)
        x_rows.append([float(features[nome]) for nome in FEATURE_NAMES])
        y_rows.append(spec.label)
        metadados.append(
            {
                "start_seconds": spec.start_seconds,
                "end_seconds": spec.end_seconds,
                "label": spec.label,
            }
        )

    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64), metadados


def extrair_dataset_janelado_edf(
    edf_path: str | Path,
    intervals: list[SeizureInterval],
    *,
    window_seconds: float = 10.0,
    step_seconds: float = 5.0,
    max_windows_per_class: int | None = 40,
    canais_selecionados: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Carrega um EDF, calcula sinal medio dos canais e extrai features por janela."""
    path = Path(edf_path)
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    if len(raw.ch_names) == 0:
        raise ValueError(f"Nenhum canal EEG encontrado em {path}")

    if canais_selecionados:
        invalidos = [canal for canal in canais_selecionados if canal not in raw.ch_names]
        if invalidos:
            raise ValueError(f"Canais EEG invalidos: {', '.join(invalidos)}")
        raw.pick(canais_selecionados)

    signal = np.mean(raw.get_data(), axis=0)
    sfreq = float(raw.info["sfreq"])
    x, y, metadados = extrair_dataset_janelado_de_sinal(
        signal,
        sfreq,
        intervals,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        max_windows_per_class=max_windows_per_class,
    )
    for item in metadados:
        item["arquivo"] = path.name
    return x, y, metadados


def avaliar_kfold_features(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Avalia as features com Logistic Regression e k-fold estratificado."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("K-fold requer ao menos duas classes: normal e crise.")

    splits = min(n_splits, int(np.min(counts)))
    if splits < 2:
        raise ValueError("Cada classe precisa ter ao menos duas janelas para k-fold.")

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []

    for train_idx, test_idx in cv.split(x, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])

        fold_metrics.append(
            {
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "precision": float(precision_score(y[test_idx], pred, zero_division=0)),
                "recall": float(recall_score(y[test_idx], pred, zero_division=0)),
                "f1": float(f1_score(y[test_idx], pred, zero_division=0)),
            }
        )

    media = {
        metric: float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in fold_metrics[0]
    }

    return {
        "n_samples": int(len(y)),
        "n_features": int(x.shape[1]) if x.ndim == 2 else 0,
        "n_splits": int(splits),
        "class_counts": {str(int(cls)): int(count) for cls, count in zip(classes, counts)},
        "folds": fold_metrics,
        "mean": media,
    }
