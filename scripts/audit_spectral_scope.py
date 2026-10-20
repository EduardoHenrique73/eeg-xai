"""Measure spectral energy outside the declared feature bands on training only."""

import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mne
import numpy as np

from app.ai_engine.training import carregar_resumos_chbmit
from scripts.patient_balancing import subject_for_file


def main():
    dataset = ROOT / "dataset_amostra"
    report = json.loads((ROOT / "modelos/v57_baseline_seed42_training.json").read_text())
    reference = json.loads((ROOT / "modelos/v43_final_seed42_metadata.json").read_text())["canais_referencia"]
    annotations = carregar_resumos_chbmit(dataset)
    rows = []
    skipped = []
    for name in report["train_files"]:
        if subject_for_file(name) in {"chb09", "chb15", "chb18", "chb16", "chb19", "chb20", "chb24"}:
            raise ValueError("Training patients only")
        intervals = annotations[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with mne.io.read_raw_edf(dataset / name, preload=False, verbose=False) as raw:
                sfreq = raw.info["sfreq"]
                duration = raw.n_times / sfreq
                picks = [ch for ch in reference if ch in raw.ch_names]
                if not picks:
                    skipped.append(name)
                    continue
                starts = [("normal", float(t)) for t in np.linspace(30, duration-34, 4)
                          if not any(t < i.end_seconds and t+4 > i.start_seconds for i in intervals)]
                starts.extend(("ictal", max(0., min(duration-4, (i.start_seconds+i.end_seconds-4)/2))) for i in intervals)
                for label, start in starts:
                    values = raw.get_data(picks=picks, start=round(start*sfreq), stop=round((start+4)*sfreq))
                    centered = values - values.mean(axis=-1, keepdims=True)
                    psd = np.abs(np.fft.rfft(centered*np.hanning(values.shape[-1]), axis=-1))**2
                    freqs = np.fft.rfftfreq(values.shape[-1], d=1/sfreq)
                    positive = freqs > 0
                    within = (freqs >= 0.5) & (freqs < 45)
                    total = psd[:, positive].sum(axis=-1)
                    valid = total > 0
                    outside = 1-psd[:, within].sum(axis=-1)[valid]/total[valid]
                    dominant = freqs[positive][np.argmax(psd[:, positive], axis=-1)][valid]
                    rows.append(dict(file=name, context=label, start_seconds=start,
                                     outside_band_fractions=outside.tolist(),
                                     dominant_frequencies_hz=dominant.tolist()))
    result = dict(training_only=True, calibration_accessed=False, historical_accessed=False,
                  note="Direct bipolar channels only. Four normal contexts per EDF and one center per seizure; channel-windows are not independent events. Referential-only files are explicitly excluded, not zero-filled.",
                  skipped_no_direct_bipolar_channels=skipped,
                  summaries={}, windows=rows)
    for label in ("normal", "ictal"):
        selected = [r for r in rows if r["context"] == label]
        fractions = np.concatenate([r["outside_band_fractions"] for r in selected])
        dominant = np.concatenate([r["dominant_frequencies_hz"] for r in selected])
        result["summaries"][label] = dict(windows=len(selected), channel_windows=len(fractions),
            median_outside_band_fraction=float(np.median(fractions)),
            p95_outside_band_fraction=float(np.percentile(fractions, 95)),
            fraction_dominant_above_45_hz=float(np.mean(dominant >= 45)))
    output = ROOT / "modelos/v58_spectral_scope_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
