from __future__ import annotations

import numpy as np
import pytest

from app.ai_engine.training import (
    FEATURE_MODE_PER_CHANNEL,
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


def test_extrair_dataset_janelado_edf_per_channel_requer_mesmos_canais(monkeypatch, tmp_path):
    class RawFake:
        def __init__(self) -> None:
            self.ch_names = ["A", "B"]
            self.info = {"sfreq": 10.0}

        def pick(self, picks):
            if picks == "eeg":
                return self
            self.ch_names = list(picks)
            return self

        def get_data(self):
            return np.vstack([np.arange(100, dtype=float), np.arange(100, dtype=float)])

    monkeypatch.setattr("mne.io.read_raw_edf", lambda *args, **kwargs: RawFake())

    caminho = tmp_path / "falso.edf"
    caminho.write_text("stub", encoding="utf-8")

    with pytest.raises(ValueError, match="Canais EEG invalidos"):
        extrair_dataset_janelado_edf(
            caminho,
            [],
            canais_selecionados=["CANAL-INEXISTENTE"],
            feature_mode=FEATURE_MODE_PER_CHANNEL,
        )


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
