"""Prevent referential EEG from becoming zeros in the application only."""

import mne
import numpy as np

from app.ai_engine import feature_extractor as fe
from app.ai_engine.training import extrair_dataset_janelado_multicanal_referencia


def raw_referential():
    sfreq = 64
    time = np.arange(8*sfreq)/sfreq
    names = ["F7-REF", "T7-REF", "P7-REF", "P8-T8", "FP1-F7"]
    values = np.vstack([np.sin(2*np.pi*f*time)*(i+1)*1e-5 for i, f in enumerate([5, 9, 13, 7, 11])])
    return mne.io.RawArray(values, mne.create_info(names, sfreq, "eeg"), verbose=False)


def test_training_and_inference_identical_with_derived_and_missing_channels(monkeypatch):
    raw = raw_referential()
    reference = ["F7-T7", "T7-P7", "T8-P8-0", "FZ-CZ", "FP1-F7"]
    expected, _, train_meta = extrair_dataset_janelado_multicanal_referencia(
        raw.copy(), [], reference, window_seconds=4, step_seconds=2,
        feature_mode=fe.FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
    )
    monkeypatch.setattr(fe, "_preparar_raw_edf", lambda *a, **kw: raw.copy())
    windows = fe.extrair_features_edf_janelado(
        "synthetic.edf", feature_mode=fe.FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
        canais_referencia=reference, window_seconds=4, step_seconds=2,
    )
    np.testing.assert_allclose([w["feature_vector"] for w in windows], expected, rtol=1e-6, atol=1e-10)
    assert windows[0]["canais_omitidos"] == train_meta[0]["canais_omitidos"] == ["FZ-CZ"]
    assert windows[0]["canais_derivados"] == train_meta[0]["canais_derivados"]
    assert windows[0]["canais_derivados"] == ["F7-T7", "T7-P7", "T8-P8-0"]


def test_user_selection_does_not_reintroduce_unselected_source(monkeypatch):
    raw = raw_referential()
    monkeypatch.setattr(fe, "_preparar_raw_edf", lambda *a, **kw: raw.copy())
    windows = fe.extrair_features_edf_janelado(
        "synthetic.edf", feature_mode=fe.FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
        canais_selecionados=["F7-REF"], canais_referencia=["F7-T7"],
    )
    assert windows[0]["canais_omitidos"] == ["F7-T7"]
    assert not np.any(windows[0]["feature_vector"])


def test_resolver_direct_priority_and_no_mismatched_common_reference():
    signal = np.arange(5, dtype=float)
    sources = {"F7-T7": signal, "T7-F7": signal*2, "F7-A": signal, "P7-B": signal}
    resolved, derived = fe.resolver_sinais_referencia(sources, ["F7-T7", "F7-P7"])
    np.testing.assert_array_equal(resolved["F7-T7"], signal)
    assert "F7-P7" not in resolved
    assert derived == set()
