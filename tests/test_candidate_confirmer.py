import numpy as np
import pytest

from app.ai_engine.candidate_confirmer import (
    FEATURE_NAMES,
    calibrar_confirmador_oof,
    candidate_feature_vector,
    candidate_ranking_score,
    selecionar_operacao_froc,
)


def test_candidate_feature_vector_e_finito():
    run = {
        "local_indices": np.asarray([1, 2, 3]),
        "duration_seconds": 8.0,
        "evidence_score": 0.9,
    }
    vetor = candidate_feature_vector(
        run,
        file_scores=np.asarray([0.1, 0.7, 0.8, 0.75, 0.2]),
        file_duration_seconds=100.0,
        n_runs_file=2,
    )

    assert vetor.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(vetor))


def test_calibrar_confirmador_oof_separa_grupos():
    rng = np.random.default_rng(42)
    groups = np.repeat(["p1", "p2", "p3"], 12)
    y = np.tile(np.asarray([0] * 8 + [1] * 4), 3)
    x = rng.normal(0, 0.1, size=(len(y), len(FEATURE_NAMES))).astype(np.float32)
    x[:, 2] += y * 1.5
    x[:, 3] += y * 1.5

    resultado = calibrar_confirmador_oof(
        x,
        y,
        groups,
        thresholds=[0.3, 0.5, 0.7],
        random_state=42,
    )

    assert resultado["n_candidates"] == 36
    assert resultado["n_positive"] == 12
    assert resultado["best_oof"]["recall"] >= 0.75
    assert len(resultado["oof_folds"]) == 3
    for fold in resultado["oof_folds"]:
        assert set(fold["train_groups"]).isdisjoint(fold["validation_groups"])


def _froc_result(*, fa_h, recall, localized_f1, precision, threshold=0.5, policy="confirmation"):
    return {
        "threshold": threshold,
        "ranking_policy": policy,
        "false_alarms_per_hour": fa_h,
        "event_sensitivity": recall,
        "event_precision": precision,
        "event_f1": 0.0,
        "localized_metrics": {"f1": localized_f1},
    }


def test_selecionar_operacao_froc_prioriza_recall_sob_restricao():
    resultados = [
        _froc_result(fa_h=0.8, recall=0.7, localized_f1=0.8, precision=0.9),
        _froc_result(fa_h=0.9, recall=0.8, localized_f1=0.6, precision=0.7, threshold=0.6),
        _froc_result(fa_h=1.1, recall=1.0, localized_f1=1.0, precision=1.0, threshold=0.7),
    ]

    selecionado = selecionar_operacao_froc(resultados, max_false_alarms_per_hour=1.0)

    assert selecionado["constraint_satisfied"] is True
    assert selecionado["selected"]["threshold"] == 0.6


def test_selecionar_operacao_froc_registra_restricao_inviavel():
    resultados = [
        _froc_result(fa_h=1.4, recall=0.9, localized_f1=0.7, precision=0.5),
        _froc_result(fa_h=1.2, recall=0.7, localized_f1=0.6, precision=0.8, threshold=0.7),
    ]

    selecionado = selecionar_operacao_froc(resultados, max_false_alarms_per_hour=1.0)

    assert selecionado["constraint_satisfied"] is False
    assert selecionado["selected"]["threshold"] == 0.7


def test_ranking_de_candidatos_compara_probabilidade_duracao_e_continuidade():
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    features[FEATURE_NAMES.index("longest_positive_block_ratio")] = 0.8
    candidato = {
        "confirmation_score": 0.75,
        "duration_seconds": 40.0,
        "features": features,
    }

    scores = {
        policy: candidate_ranking_score(
            candidato,
            policy=policy,
            max_duration_seconds=80.0,
        )
        for policy in ("confirmation", "duration", "continuity", "combined")
    }

    assert scores["confirmation"] == 0.75
    assert scores["duration"] == 0.375
    assert scores["continuity"] == pytest.approx(0.6)
    assert 0.0 < scores["combined"] < scores["confirmation"]
