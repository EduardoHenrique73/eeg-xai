"""Verify training/inference parity on four seconds of a referential train EDF."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from app.ai_engine import feature_extractor as fe
from app.ai_engine.training import extrair_dataset_janelado_multicanal_referencia


def main():
    name = "chb12_27.edf"
    report = json.loads((ROOT / "modelos/v57_baseline_seed42_training.json").read_text())
    if name not in report["train_files"]:
        raise ValueError("Parity verification must use a training file")
    path = ROOT / "dataset_amostra" / name
    reference = json.loads((ROOT / "modelos/v43_final_seed42_metadata.json").read_text())["canais_referencia"]
    raw = fe._preparar_raw_edf(str(path), max_duration_seconds=4)
    direct = sum(ch in raw.ch_names for ch in reference)
    expected, _, train_meta = extrair_dataset_janelado_multicanal_referencia(
        raw, [], reference, window_seconds=4, step_seconds=2,
        feature_mode=fe.FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
    )
    windows = fe.extrair_features_edf_janelado(
        str(path), max_duration_seconds=4, canais_referencia=reference,
        feature_mode=fe.FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
    )
    actual = np.asarray([w["feature_vector"] for w in windows], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert windows[0]["canais_omitidos"] == train_meta[0]["canais_omitidos"]
    result = dict(file=name, training_only=True, historical_accessed=False,
                  seconds=4, direct_reference_slots=direct,
                  processed_reference_slots=len(windows[0]["canais_processados"]),
                  derived_channels=windows[0]["canais_derivados"],
                  omitted_channels=windows[0]["canais_omitidos"],
                  feature_shape=list(actual.shape),
                  max_absolute_difference=float(np.max(np.abs(actual-expected))),
                  nonzero_features=int(np.count_nonzero(actual)))
    (ROOT / "modelos/v58_reference_montage_parity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
