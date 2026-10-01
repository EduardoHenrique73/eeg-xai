"""Segundo estagio para confirmar candidatos temporais de crise."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut


FEATURE_NAMES = (
    "duration_log",
    "n_windows_log",
    "score_mean",
    "score_max",
    "score_median",
    "score_p75",
    "score_std",
    "score_variance",
    "score_p90",
    "score_min",
    "score_auc_normalized",
    "count_above_010_log",
    "count_above_020_log",
    "count_above_030_log",
    "count_above_050_log",
    "fraction_above_010",
    "fraction_above_020",
    "fraction_above_030",
    "fraction_above_050",
    "evidence_score",
    "contrast_to_file_median",
    "contrast_to_file_p90",
    "context_before_mean",
    "context_after_mean",
    "contrast_to_local_context",
    "peak_contrast_to_local_context",
    "coverage_ratio",
    "run_density_per_minute",
    "longest_positive_block_ratio",
    "gap_count_log",
    "mean_gap_windows_log",
    "duration_above_010_ratio",
    "duration_above_020_ratio",
    "duration_above_030_ratio",
    "duration_above_050_ratio",
    "max_score_rise",
    "max_score_drop",
    "initial_trend",
    "final_trend",
    "peak_to_mean",
)

RANKING_POLICIES = ("confirmation", "duration", "continuity", "combined")


def _longest_true_block(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _gap_lengths(mask: np.ndarray) -> list[int]:
    mask = np.asarray(mask, dtype=bool)
    positive = np.flatnonzero(mask)
    if positive.size < 2:
        return []
    interior = ~mask[positive[0] : positive[-1] + 1]
    lengths: list[int] = []
    current = 0
    for value in interior:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _edge_trend(scores: np.ndarray, *, initial: bool) -> float:
    if scores.size < 2:
        return 0.0
    width = min(4, scores.size)
    edge = scores[:width] if initial else scores[-width:]
    return float(np.polyfit(np.arange(edge.size, dtype=float), edge, 1)[0])


def candidate_feature_vector(
    run: dict[str, Any],
    *,
    file_scores: np.ndarray,
    file_duration_seconds: float,
    n_runs_file: int,
    generation_threshold: float = 0.1,
    context_windows: int = 5,
) -> np.ndarray:
    """Resume intensidade, continuidade e contexto do candidato sem usar tempo absoluto."""
    run_scores = np.asarray(file_scores[np.asarray(run["local_indices"], dtype=np.int64)], dtype=float)
    duration = float(run["duration_seconds"])
    file_scores = np.asarray(file_scores, dtype=float)
    file_minutes = max(file_duration_seconds / 60.0, 1.0 / 60.0)
    local_indices = np.asarray(run["local_indices"], dtype=np.int64)
    start_idx = int(np.min(local_indices))
    end_idx = int(np.max(local_indices))
    before = file_scores[max(0, start_idx - context_windows) : start_idx]
    after = file_scores[end_idx + 1 : end_idx + 1 + context_windows]
    file_baseline = float(np.median(file_scores))
    before_mean = float(np.mean(before)) if before.size else file_baseline
    after_mean = float(np.mean(after)) if after.size else file_baseline
    local_context = float(np.mean([before_mean, after_mean]))
    positive_mask = run_scores >= generation_threshold
    gaps = _gap_lengths(positive_mask)
    score_diffs = np.diff(run_scores)
    thresholds = (0.1, 0.2, 0.3, 0.5)
    counts = [int(np.sum(run_scores >= item)) for item in thresholds]
    fractions = [count / max(run_scores.size, 1) for count in counts]
    auc_normalized = float(np.trapz(run_scores) / max(run_scores.size - 1, 1))
    return np.asarray(
        [
            np.log1p(duration),
            np.log1p(run_scores.size),
            np.mean(run_scores),
            np.max(run_scores),
            np.median(run_scores),
            np.percentile(run_scores, 75),
            np.std(run_scores),
            np.var(run_scores),
            np.percentile(run_scores, 90),
            np.min(run_scores),
            auc_normalized,
            *(np.log1p(item) for item in counts),
            *fractions,
            float(run.get("evidence_score", 0.0)),
            np.mean(run_scores) - file_baseline,
            np.mean(run_scores) - np.percentile(file_scores, 90),
            before_mean,
            after_mean,
            np.mean(run_scores) - local_context,
            np.max(run_scores) - local_context,
            duration / max(file_duration_seconds, 1.0),
            n_runs_file / file_minutes,
            _longest_true_block(positive_mask) / max(run_scores.size, 1),
            np.log1p(len(gaps)),
            np.log1p(np.mean(gaps) if gaps else 0.0),
            *fractions,
            np.max(score_diffs) if score_diffs.size else 0.0,
            -np.min(score_diffs) if score_diffs.size else 0.0,
            _edge_trend(run_scores, initial=True),
            _edge_trend(run_scores, initial=False),
            np.max(run_scores) - np.mean(run_scores),
        ],
        dtype=np.float32,
    )


def candidate_ranking_score(
    candidate: dict[str, Any],
    *,
    policy: str,
    max_duration_seconds: float,
) -> float:
    """Pontua o candidato sem consultar anotacao ou overlap real."""
    if policy not in RANKING_POLICIES:
        raise ValueError(f"Politica de ranking invalida: {policy}")
    confirmation = float(candidate["confirmation_score"])
    duration_norm = float(candidate["duration_seconds"]) / max(max_duration_seconds, 1.0)
    features = np.asarray(candidate["features"], dtype=float)
    continuity = float(features[FEATURE_NAMES.index("longest_positive_block_ratio")])
    if policy == "confirmation":
        return confirmation
    if policy == "duration":
        return confirmation * duration_norm
    if policy == "continuity":
        return confirmation * continuity
    return confirmation * (0.50 + 0.25 * duration_norm + 0.25 * continuity)


def selecionar_operacao_froc(
    resultados: list[dict[str, Any]],
    *,
    max_false_alarms_per_hour: float,
) -> dict[str, Any]:
    """Seleciona recall, localizacao e precisao sob um teto explicito de FA/h."""
    if not resultados:
        raise ValueError("A selecao FROC requer ao menos um resultado.")
    elegiveis = [
        item
        for item in resultados
        if float(item["false_alarms_per_hour"]) <= max_false_alarms_per_hour + 1e-12
    ]
    candidatos = elegiveis or resultados
    if elegiveis:
        chave = lambda item: (
            float(item["event_sensitivity"]),
            float(item["localized_metrics"]["f1"]),
            float(item["event_precision"]),
            -float(item["false_alarms_per_hour"]),
        )
    else:
        chave = lambda item: (
            -float(item["false_alarms_per_hour"]),
            float(item["event_sensitivity"]),
            float(item["localized_metrics"]["f1"]),
            float(item["event_precision"]),
        )
    selecionado = max(candidatos, key=chave)
    return {
        "max_false_alarms_per_hour": float(max_false_alarms_per_hour),
        "constraint_satisfied": bool(elegiveis),
        "selected": selecionado,
    }


def criar_confirmador(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def calibrar_confirmador_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    thresholds: list[float],
    random_state: int,
) -> dict[str, Any]:
    """Escolhe threshold com predicoes leave-one-patient-out na calibracao."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    if len(np.unique(y)) < 2:
        raise ValueError("Confirmador requer candidatos positivos e negativos.")
    if len(np.unique(groups)) < 2:
        raise ValueError("Confirmador requer ao menos dois pacientes de calibracao.")

    scores = np.zeros(len(y), dtype=np.float32)
    fold_audit: list[dict[str, Any]] = []
    logo = LeaveOneGroupOut()
    for fold_idx, (train_idx, valid_idx) in enumerate(logo.split(x, y, groups), start=1):
        if len(np.unique(y[train_idx])) < 2:
            raise ValueError("Uma dobra do confirmador nao possui as duas classes.")
        modelo = criar_confirmador(random_state + fold_idx)
        modelo.fit(x[train_idx], y[train_idx])
        scores[valid_idx] = modelo.predict_proba(x[valid_idx])[:, 1]
        fold_audit.append(
            {
                "train_groups": sorted({str(item) for item in groups[train_idx]}),
                "validation_groups": sorted({str(item) for item in groups[valid_idx]}),
                "n_train": int(len(train_idx)),
                "n_validation": int(len(valid_idx)),
            }
        )

    resultados = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(np.int64)
        resultados.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "false_positives": int(np.sum((y == 0) & (pred == 1))),
                "false_negatives": int(np.sum((y == 1) & (pred == 0))),
            }
        )
    resultados.sort(
        key=lambda item: (
            item["f1"],
            item["recall"],
            -item["false_positives"],
            item["precision"],
        ),
        reverse=True,
    )
    modelo_final = criar_confirmador(random_state)
    modelo_final.fit(x, y)
    return {
        "model": modelo_final,
        "oof_scores": scores,
        "threshold": resultados[0]["threshold"],
        "best_oof": resultados[0],
        "top_oof": resultados[:5],
        "n_candidates": int(len(y)),
        "n_positive": int(np.sum(y == 1)),
        "n_hard_negative": int(np.sum(y == 0)),
        "feature_names": list(FEATURE_NAMES),
        "oof_folds": fold_audit,
    }
