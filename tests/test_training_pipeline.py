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
    limitar_dataset_janelado,
    parse_chbmit_summary,
    treinar_cnn_lstm_final,
)
from app.ai_engine.feature_extractor import FEATURE_NAMES, TIME_FREQUENCY_FEATURE_NAMES
from app.ai_engine.candidate_confirmer import FEATURE_NAMES as CONFIRMER_FEATURE_NAMES
from scripts.calibrate_sequence_cnn_lstm import (
    avaliar_combo,
    intervalos_crise_reais,
    listar_runs,
    preencher_lacunas_curtas,
)
from scripts.train_sequence_cnn_lstm import (
    construir_sequencias,
    resolver_class_weight,
    resolver_sample_weights,
)
from scripts.validate_sequence_by_patient import (
    avaliar_confirmador,
    ranking_calibracao,
    ranking_calibracao_balanceada,
    separar_treino_calibracao,
    validar_isolamento_fold,
)


def test_avaliar_confirmador_calcula_um_falso_alarme_por_hora():
    features = np.zeros(len(CONFIRMER_FEATURE_NAMES), dtype=np.float32)
    candidatos = [
        {
            "arquivo": "chb01_01.edf",
            "label": 0,
            "duration_seconds": 20.0,
            "start_seconds": 100.0,
            "end_seconds": 120.0,
            "score_mean": 0.8,
            "score_max": 0.9,
            "overlap_seconds": 0.0,
            "overlap_ratio": 0.0,
            "features": features,
        }
    ]
    resultado = avaliar_confirmador(
        candidatos,
        np.asarray([0.9]),
        threshold=0.5,
        y_true=np.asarray([0]),
        metas=[{"arquivo": "chb01_01.edf", "start_seconds": 0.0, "end_seconds": 3600.0}],
    )

    assert resultado["false_alarm_events"] == 1
    assert resultado["normal_hours"] == pytest.approx(1.0)
    assert resultado["false_alarms_per_hour"] == pytest.approx(1.0)


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


def test_separar_treino_calibracao_exclui_toda_coorte_reservada():
    por_paciente = {
        "chb01": ["chb01_01.edf"],
        "chb02": ["chb02_01.edf"],
        "chb03": ["chb03_01.edf"],
        "chb09": ["chb09_01.edf"],
        "chb15": ["chb15_01.edf"],
        "chb18": ["chb18_01.edf"],
    }

    treino, calibracao, pacientes_calibracao = separar_treino_calibracao(
        por_paciente,
        "chb09",
        n_calibration_patients=1,
        reserved_test_patients={"chb09", "chb15", "chb18"},
    )

    assert treino == ["chb01_01.edf", "chb02_01.edf"]
    assert calibracao == ["chb03_01.edf"]
    assert pacientes_calibracao == ["chb03"]


def test_separar_treino_calibracao_respeita_coorte_fixa():
    por_paciente = {
        "chb01": ["chb01_01.edf"],
        "chb02": ["chb02_01.edf"],
        "chb03": ["chb03_01.edf"],
        "chb04": ["chb04_01.edf"],
        "chb09": ["chb09_01.edf"],
    }

    treino, calibracao, pacientes_calibracao = separar_treino_calibracao(
        por_paciente,
        "chb09",
        n_calibration_patients=1,
        reserved_test_patients={"chb09"},
        fixed_calibration_patients=["chb02", "chb04"],
    )

    assert treino == ["chb01_01.edf", "chb03_01.edf"]
    assert calibracao == ["chb02_01.edf", "chb04_01.edf"]
    assert pacientes_calibracao == ["chb02", "chb04"]


def test_validar_isolamento_fold_rejeita_reservado_no_treino():
    with pytest.raises(RuntimeError, match="Vazamento de paciente detectado"):
        validar_isolamento_fold(
            train_files=["chb09_01.edf", "chb01_01.edf"],
            calibration_files=["chb16_01.edf"],
            test_files=["chb15_01.edf"],
            reserved_test_patients={"chb09", "chb15", "chb18"},
        )


def test_validar_isolamento_fold_aceita_particoes_disjuntas():
    validar_isolamento_fold(
        train_files=["chb01_01.edf", "chb02_01.edf"],
        calibration_files=["chb16_01.edf"],
        test_files=["chb09_01.edf"],
        reserved_test_patients={"chb09", "chb15", "chb18"},
    )


def test_ranking_balanceado_nao_sacrifica_sensibilidade_para_zerar_fp():
    conservador = {
        "false_positives": 0,
        "false_negatives": 4,
        "localized_false_negatives": 4,
        "positive_predictions": 1,
        "positive_predictions_with_overlap": 1,
        "positive_overlap_seconds_sum": 20.0,
        "localized_metrics": {"f1": 0.25, "recall": 0.15},
        "metrics": {"f1": 0.3, "accuracy": 0.6},
    }
    balanceado = {
        "false_positives": 1,
        "false_negatives": 1,
        "localized_false_negatives": 1,
        "positive_predictions": 4,
        "positive_predictions_with_overlap": 4,
        "positive_overlap_seconds_sum": 80.0,
        "localized_metrics": {"f1": 0.75, "recall": 0.8},
        "metrics": {"f1": 0.8, "accuracy": 0.8},
    }

    assert ranking_calibracao(conservador) > ranking_calibracao(balanceado)
    assert ranking_calibracao_balanceada(balanceado) > ranking_calibracao_balanceada(conservador)


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


def test_construir_sequencias_center_rotula_e_localiza_janela_alvo():
    x = np.ones((4, 2), dtype=np.float32)
    y = np.asarray([1, 0, 0, 0], dtype=np.int64)
    meta = [
        {
            "arquivo": "chb01_03.edf",
            "start_seconds": float(i * 2),
            "end_seconds": float(i * 2 + 4),
            "context": "ictal" if i == 0 else "interictal",
        }
        for i in range(4)
    ]

    _x_seq, y_seq, meta_seq = construir_sequencias(
        x,
        y,
        meta,
        sequence_length=4,
        sequence_stride=1,
        target_mode="center",
    )

    assert y_seq.tolist() == [0]
    assert meta_seq[0]["start_seconds"] == 4.0
    assert meta_seq[0]["end_seconds"] == 8.0
    assert meta_seq[0]["context_start_seconds"] == 0.0
    assert meta_seq[0]["context_end_seconds"] == 10.0


def test_limitar_dataset_janelado_preserva_valores_processados():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.asarray([0, 0, 1, 1, 0, 0], dtype=np.int64)
    meta = [
        {
            "start_seconds": float(indice * 2),
            "end_seconds": float(indice * 2 + 4),
            "context": "ictal" if label else "interictal",
            "ictal_overlap_ratio": float(label),
        }
        for indice, label in enumerate(y)
    ]

    x_limitado, y_limitado, meta_limitado = limitar_dataset_janelado(
        x,
        y,
        meta,
        max_normal_windows=2,
        max_seizure_windows=2,
        min_contiguous_windows=2,
    )

    assert y_limitado.tolist() == [0, 1, 1, 0]
    assert x_limitado.tolist() == x[[0, 2, 3, 4]].tolist()
    assert [item["start_seconds"] for item in meta_limitado] == [0.0, 4.0, 6.0, 8.0]


def test_resolver_sample_weights_reforca_borda_da_crise():
    pesos = resolver_sample_weights(
        [{"context": "interictal"}, {"context": "boundary"}, {"context": "ictal"}],
        boundary_weight=1.75,
        y=np.asarray([0, 0, 1]),
        class_weight={0: 0.5, 1: 2.0},
    )

    assert pesos.tolist() == [0.5, 0.875, 2.0]


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


def test_listar_runs_prioriza_evidencia_acumulada():
    scores = np.asarray([0.7, 0.7, 0.7, 0.1, 0.95], dtype=np.float32)
    metas = [
        {"start_seconds": float(idx * 10), "end_seconds": float((idx + 1) * 10)}
        for idx in range(len(scores))
    ]

    runs = listar_runs(
        scores,
        metas,
        np.arange(len(scores)),
        threshold=0.5,
        min_duration_seconds=10.0,
    )

    assert runs[0]["n_sequences"] == 3
    assert runs[0]["evidence_score"] > runs[1]["evidence_score"]


def test_preencher_lacuna_curta_sem_atravessar_buraco_temporal():
    metas = [
        {"start_seconds": 0.0, "end_seconds": 4.0},
        {"start_seconds": 2.0, "end_seconds": 6.0},
        {"start_seconds": 4.0, "end_seconds": 8.0},
        {"start_seconds": 100.0, "end_seconds": 104.0},
        {"start_seconds": 102.0, "end_seconds": 106.0},
    ]
    resultado = preencher_lacunas_curtas(
        np.asarray([True, False, True, True, False]),
        metas,
        np.arange(5),
        max_gap_seconds=4.0,
    )

    assert resultado.tolist() == [True, True, True, True, False]


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


def test_gerar_janelas_temporais_exige_sobreposicao_minima():
    specs = gerar_janelas_temporais(
        20.0,
        [SeizureInterval(8.0, 16.0)],
        window_seconds=10.0,
        step_seconds=10.0,
        min_ictal_overlap_ratio=0.5,
    )

    assert [(s.label, s.context, s.ictal_overlap_ratio) for s in specs] == [
        (0, "boundary", pytest.approx(0.2)),
        (1, "ictal", pytest.approx(0.6)),
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


def test_amostragem_preserva_bloco_ictal_suficiente_para_sequencia():
    sfreq = 10.0
    signal = np.sin(np.linspace(0, 300 * np.pi, 3000))

    x, y, meta = extrair_dataset_janelado_de_sinal(
        signal,
        sfreq,
        [SeizureInterval(100.0, 140.0)],
        window_seconds=4.0,
        step_seconds=2.0,
        max_windows_per_class=None,
        max_normal_windows=16,
        max_seizure_windows=12,
        min_ictal_overlap_ratio=0.5,
        min_contiguous_windows=8,
    )
    for item in meta:
        item["arquivo"] = "chb01_03.edf"
    _x_seq, y_seq, _meta_seq = construir_sequencias(
        x,
        y,
        meta,
        sequence_length=8,
        sequence_stride=2,
        target_mode="center",
    )

    positivos = [m["start_seconds"] for m, label in zip(meta, y) if label == 1]
    assert len(positivos) == 12
    assert all(atual - anterior == pytest.approx(2.0) for anterior, atual in zip(positivos, positivos[1:]))
    assert 1 in y_seq


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
    assert resultado["false_alarm_events"] == 1
    assert resultado["normal_hours"] == pytest.approx(10.0 / 3600.0)
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


def test_listar_runs_histerese_expande_apenas_bloco_com_semente():
    scores = np.asarray([0.42, 0.55, 0.8, 0.52, 0.1, 0.58, 0.57], dtype=np.float32)
    metas = [
        {"start_seconds": float(i * 2), "end_seconds": float(i * 2 + 4)}
        for i in range(len(scores))
    ]

    runs = listar_runs(
        scores,
        metas,
        np.arange(len(scores)),
        threshold=0.8,
        hysteresis_ratio=0.5,
    )

    assert len(runs) == 1
    assert runs[0]["start_seconds"] == 0.0
    assert runs[0]["end_seconds"] == 10.0


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
