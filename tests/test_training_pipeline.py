from __future__ import annotations

import numpy as np
import pytest

from app.ai_engine.training import (
    FEATURE_MODE_PER_CHANNEL,
    FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
    SeizureInterval,
    avaliar_kfold_cnn_lstm,
    avaliar_kfold_features,
    carregar_resumos_chbmit,
    extrair_dataset_janelado_de_sinal,
    extrair_dataset_janelado_multicanal,
    extrair_dataset_janelado_edf,
    gerar_janelas_temporais,
    parse_chbmit_summary,
    treinar_cnn_lstm_final,
)
from app.ai_engine.feature_extractor import FEATURE_NAMES, TIME_FREQUENCY_FEATURE_NAMES
from scripts.calibrate_sequence_cnn_lstm import avaliar_combo, intervalos_crise_reais, listar_runs
from scripts.train_sequence_cnn_lstm import construir_sequencias, resolver_class_weight


def test_resolver_class_weight_permite_peso_manual_e_desativado():
    y = np.asarray([0, 0, 0, 1], dtype=np.int64)

    assert resolver_class_weight(y, mode="none", positive_weight=2.0) is None
    assert resolver_class_weight(y, mode="manual", positive_weight=3.0) == {
        0: 1.0,
        1: 3.0,
    }

    balanced = resolver_class_weight(y, mode="balanced", positive_weight=2.0)
    assert balanced is not None
    assert balanced[1] > balanced[0]


def test_construir_sequencias_nao_atravessa_buracos_temporais():
    x = np.ones((4, 2), dtype=np.float32)
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    meta = [
        {"arquivo": "chb01_01.edf", "start_seconds": 0.0, "end_seconds": 10.0},
        {"arquivo": "chb01_01.edf", "start_seconds": 10.0, "end_seconds": 20.0},
        {"arquivo": "chb01_01.edf", "start_seconds": 100.0, "end_seconds": 110.0},
        {"arquivo": "chb01_01.edf", "start_seconds": 110.0, "end_seconds": 120.0},
    ]

    x_seq, y_seq, meta_seq = construir_sequencias(
        x,
        y,
        meta,
        sequence_length=2,
        sequence_stride=1,
    )

    assert x_seq.shape == (2, 2, 2)
    assert y_seq.tolist() == [0, 1]
    assert [(m["start_seconds"], m["end_seconds"]) for m in meta_seq] == [
        (0.0, 20.0),
        (100.0, 120.0),
    ]


def test_listar_runs_nao_junta_buracos_temporais():
    scores = np.asarray([0.9, 0.92, 0.91], dtype=np.float32)
    metas = [
        {"start_seconds": 0.0, "end_seconds": 10.0},
        {"start_seconds": 10.0, "end_seconds": 20.0},
        {"start_seconds": 100.0, "end_seconds": 110.0},
    ]

    runs = listar_runs(scores, metas, np.asarray([0, 1, 2]), threshold=0.5)

    assert len(runs) == 2
    assert {(run["start_seconds"], run["end_seconds"]) for run in runs} == {
        (0.0, 20.0),
        (100.0, 110.0),
    }


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


def test_parse_chbmit_summary_aceita_seizures_numeradas(tmp_path):
    summary = tmp_path / "chb12-summary.txt"
    summary.write_text(
        """
File Name: chb12_06.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 1665 seconds
Seizure 1 End Time: 1726 seconds
Seizure 2 Start Time: 3415 seconds
Seizure 2 End Time: 3447 seconds
""",
        encoding="utf-8",
    )

    resultado = parse_chbmit_summary(summary)

    assert resultado["chb12_06.edf"] == [
        SeizureInterval(1665.0, 1726.0),
        SeizureInterval(3415.0, 3447.0),
    ]


def test_carregar_resumos_chbmit_combina_multiplos_pacientes(tmp_path):
    (tmp_path / "chb01-summary.txt").write_text(
        """
File Name: chb01_03.edf
Seizure Start Time: 10 seconds
Seizure End Time: 20 seconds
""",
        encoding="utf-8",
    )
    (tmp_path / "chb02-summary.txt").write_text(
        """
File Name: chb02_16.edf
Seizure Start Time: 30 seconds
Seizure End Time: 40 seconds
""",
        encoding="utf-8",
    )

    resultado = carregar_resumos_chbmit(tmp_path)

    assert resultado["chb01_03.edf"] == [SeizureInterval(10.0, 20.0)]
    assert resultado["chb02_16.edf"] == [SeizureInterval(30.0, 40.0)]


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


def test_extrair_dataset_janelado_de_sinal_limite_preserva_janelas_continuas():
    sfreq = 10.0
    signal = np.sin(np.linspace(0, 20 * np.pi, 300))

    _x, y, meta = extrair_dataset_janelado_de_sinal(
        signal,
        sfreq,
        [],
        window_seconds=10.0,
        step_seconds=10.0,
        max_windows_per_class=None,
        max_normal_windows=3,
    )

    assert y.tolist() == [0, 0, 0]
    assert [(m["start_seconds"], m["end_seconds"]) for m in meta] == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 30.0),
    ]


def test_extrair_dataset_janelado_de_sinal_normais_perto_da_crise():
    sfreq = 10.0
    signal = np.sin(np.linspace(0, 80 * np.pi, 1200))

    _x, y, meta = extrair_dataset_janelado_de_sinal(
        signal,
        sfreq,
        [SeizureInterval(60.0, 70.0)],
        window_seconds=10.0,
        step_seconds=10.0,
        max_windows_per_class=None,
        max_normal_windows=2,
        max_seizure_windows=1,
    )

    normais = [m for m in meta if m["label"] == 0]
    assert y.tolist().count(1) == 1
    assert [(m["start_seconds"], m["end_seconds"]) for m in normais] == [
        (50.0, 60.0),
        (70.0, 80.0),
    ]


def test_extrair_dataset_janelado_de_sinal_balanceia_contextos_da_crise():
    sfreq = 10.0
    signal = np.sin(np.linspace(0, 160 * np.pi, 2400))

    _x, _y, meta = extrair_dataset_janelado_de_sinal(
        signal,
        sfreq,
        [SeizureInterval(100.0, 120.0)],
        window_seconds=10.0,
        step_seconds=10.0,
        max_windows_per_class=None,
        max_normal_windows=6,
        max_seizure_windows=2,
    )

    contextos_normais = {m["context"] for m in meta if m["label"] == 0}
    assert {"pre_ictal", "post_ictal", "interictal"}.issubset(contextos_normais)
    assert [m["context"] for m in meta if m["label"] == 1] == ["ictal", "ictal"]


def test_extrair_dataset_janelado_de_sinal_normal_espalha_blocos():
    sfreq = 10.0
    signal = np.sin(np.linspace(0, 80 * np.pi, 1200))

    _x, y, meta = extrair_dataset_janelado_de_sinal(
        signal,
        sfreq,
        [],
        window_seconds=10.0,
        step_seconds=10.0,
        max_windows_per_class=None,
        max_normal_windows=4,
    )

    assert y.tolist() == [0, 0, 0, 0]
    assert [m["start_seconds"] for m in meta] == [0.0, 30.0, 70.0, 110.0]


def test_avaliar_combo_penaliza_overlap_baixo_por_proporcao():
    y = np.asarray([0, 1, 1, 1], dtype=np.int64)
    scores = np.asarray([0.9, 0.9, 0.1, 0.1], dtype=np.float32)
    metas = [
        {"arquivo": "chb01_03.edf", "start_seconds": 0.0, "end_seconds": 10.0},
        {"arquivo": "chb01_03.edf", "start_seconds": 10.0, "end_seconds": 20.0},
        {"arquivo": "chb01_03.edf", "start_seconds": 20.0, "end_seconds": 30.0},
        {"arquivo": "chb01_03.edf", "start_seconds": 30.0, "end_seconds": 40.0},
    ]

    resultado = avaliar_combo(
        y,
        scores,
        metas,
        threshold=0.5,
        min_duration_seconds=10.0,
        min_overlap_seconds=1.0,
        min_overlap_ratio=0.5,
    )

    assert resultado["metrics"]["recall"] == 1.0
    assert resultado["localized_metrics"]["recall"] == 0.0
    assert resultado["positive_predictions_without_overlap"] == 1
    assert resultado["files"][0]["trecho_suspeito"]["overlap_crise_real_ratio"] == pytest.approx(1 / 3)


def test_intervalos_crise_reais_separa_multiplas_crises():
    y = np.asarray([1, 1, 0, 1], dtype=np.int64)
    metas = [
        {"start_seconds": 0.0, "end_seconds": 10.0},
        {"start_seconds": 10.0, "end_seconds": 20.0},
        {"start_seconds": 20.0, "end_seconds": 30.0},
        {"start_seconds": 100.0, "end_seconds": 110.0},
    ]

    segmentos = intervalos_crise_reais(y, metas, np.asarray([0, 1, 2, 3]))

    assert segmentos == [
        {"start_seconds": 0.0, "end_seconds": 20.0, "n_sequences": 2},
        {"start_seconds": 100.0, "end_seconds": 110.0, "n_sequences": 1},
    ]


def test_extrair_dataset_janelado_multicanal_preserva_features_por_canal():
    sfreq = 10.0
    base = np.sin(np.linspace(0, 20 * np.pi, 300))
    sinais = np.vstack([base, base * 0.5 + 0.25])

    x, y, meta = extrair_dataset_janelado_multicanal(
        sinais,
        sfreq,
        [SeizureInterval(12.0, 18.0)],
        window_seconds=10.0,
        step_seconds=5.0,
    )

    assert x.shape[1] == 38
    assert len(y) == len(meta)
    assert set(y.tolist()) == {0, 1}


def test_extrair_dataset_janelado_edf_per_channel_zera_canal_ausente(monkeypatch, tmp_path):
    class RawFake:
        def __init__(self) -> None:
            self.ch_names = ["A", "B"]
            self.info = {"sfreq": 10.0}
            self.n_times = 100

        def pick(self, picks):
            if picks == "eeg":
                return self
            self.ch_names = list(picks)
            return self

        def get_data(self, picks=None, start=0, stop=None):
            fim = self.n_times if stop is None else stop
            return np.vstack([np.arange(start, fim, dtype=float), np.arange(start, fim, dtype=float)])

    monkeypatch.setattr("mne.io.read_raw_edf", lambda *args, **kwargs: RawFake())

    caminho = tmp_path / "falso.edf"
    caminho.write_text("stub", encoding="utf-8")

    x, y, meta = extrair_dataset_janelado_edf(
        caminho,
        [],
        canais_selecionados=["CANAL-INEXISTENTE"],
        feature_mode=FEATURE_MODE_PER_CHANNEL,
    )

    assert x.shape == (1, len(FEATURE_NAMES))
    assert np.allclose(x[0], 0.0)
    assert y.tolist() == [0]
    assert meta[0]["canais_omitidos"] == ["CANAL-INEXISTENTE"]


def test_extrair_dataset_janelado_edf_time_frequency_per_channel_zera_canal_ausente(monkeypatch, tmp_path):
    class RawFake:
        def __init__(self) -> None:
            self.ch_names = ["A", "B"]
            self.info = {"sfreq": 64.0}
            self.n_times = 640

        def pick(self, picks):
            if picks == "eeg":
                return self
            self.ch_names = list(picks)
            return self

        def get_data(self, picks=None, start=0, stop=None):
            fim = self.n_times if stop is None else stop
            t = np.arange(start, fim, dtype=float) / self.info["sfreq"]
            return np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 10 * t)])

    monkeypatch.setattr("mne.io.read_raw_edf", lambda *args, **kwargs: RawFake())

    caminho = tmp_path / "falso.edf"
    caminho.write_text("stub", encoding="utf-8")

    x, y, meta = extrair_dataset_janelado_edf(
        caminho,
        [],
        canais_selecionados=["CANAL-INEXISTENTE"],
        feature_mode=FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
    )

    assert x.shape == (1, len(TIME_FREQUENCY_FEATURE_NAMES))
    assert np.allclose(x[0], 0.0)
    assert y.tolist() == [0]
    assert meta[0]["canais_omitidos"] == ["CANAL-INEXISTENTE"]


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


def test_avaliar_kfold_cnn_lstm_exige_duas_classes():
    x = np.ones((4, 19), dtype=np.float32)
    y = np.zeros(4, dtype=np.int64)

    with pytest.raises(ValueError, match="duas classes"):
        avaliar_kfold_cnn_lstm(x, y, epochs=1)


def test_treinar_cnn_lstm_final_salva_modelo_e_scaler(tmp_path):
    rng = np.random.default_rng(123)
    x = np.vstack(
        [
            rng.normal(0, 0.2, size=(4, 19)),
            rng.normal(1, 0.2, size=(4, 19)),
        ]
    ).astype(np.float32)
    y = np.array([0] * 4 + [1] * 4, dtype=np.int64)
    output = tmp_path / "cnn_lstm_test.keras"

    artefatos = treinar_cnn_lstm_final(
        x,
        y,
        output_path=output,
        epochs=1,
        batch_size=4,
    )

    assert output.exists()
    assert (tmp_path / "cnn_lstm_test_scaler.pkl").exists()
    assert artefatos["model_path"].endswith("cnn_lstm_test.keras")
    assert artefatos["scaler_path"].endswith("cnn_lstm_test_scaler.pkl")
