from __future__ import annotations

import numpy as np
import pytest

from app.ai_engine.training import (
    SeizureInterval,
    avaliar_kfold_features,
    extrair_dataset_janelado_de_sinal,
    gerar_janelas_temporais,
    parse_chbmit_summary,
)


def test_parse_chbmit_summary_extrai_intervalos(tmp_path):
    summary = tmp_path / "chb01-summary.txt"
    summary.write_text(
        """
File Name: chb01_01.edf
Number of Seizures in File: 0

File Name: chb01_03.edf
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds
""",
        encoding="utf-8",
    )

    resultado = parse_chbmit_summary(summary)

    assert resultado["chb01_01.edf"] == []
    assert resultado["chb01_03.edf"] == [SeizureInterval(2996.0, 3036.0)]


def test_gerar_janelas_temporais_rotula_por_sobreposicao():
    specs = gerar_janelas_temporais(
        30.0,
        [SeizureInterval(12.0, 18.0)],
        window_seconds=10.0,
        step_seconds=10.0,
    )

    assert [(s.start_seconds, s.end_seconds, s.label) for s in specs] == [
        (0.0, 10.0, 0),
        (10.0, 20.0, 1),
        (20.0, 30.0, 0),
    ]


def test_extrair_dataset_janelado_de_sinal_gera_features_e_labels():
    sfreq = 10.0
    signal = np.sin(np.linspace(0, 20 * np.pi, 300))

    x, y, meta = extrair_dataset_janelado_de_sinal(
        signal,
        sfreq,
        [SeizureInterval(12.0, 18.0)],
        window_seconds=10.0,
        step_seconds=5.0,
    )

    assert x.shape[1] == 19
    assert len(y) == len(meta)
    assert set(y.tolist()) == {0, 1}


def test_avaliar_kfold_features_retorna_metricas():
    rng = np.random.default_rng(42)
    x_normal = rng.normal(0, 0.2, size=(8, 19))
    x_crise = rng.normal(1, 0.2, size=(8, 19))
    x = np.vstack([x_normal, x_crise])
    y = np.array([0] * 8 + [1] * 8)

    metrics = avaliar_kfold_features(x, y, n_splits=4)

    assert metrics["n_samples"] == 16
    assert metrics["n_features"] == 19
    assert metrics["n_splits"] == 4
    assert 0.0 <= metrics["mean"]["accuracy"] <= 1.0


def test_avaliar_kfold_features_exige_duas_classes():
    x = np.ones((4, 19))
    y = np.zeros(4)

    with pytest.raises(ValueError, match="duas classes"):
        avaliar_kfold_features(x, y)
