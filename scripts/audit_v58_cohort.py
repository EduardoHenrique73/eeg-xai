"""Describe the frozen development cohort using annotations and EDF headers."""

import argparse
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mne

from app.ai_engine.training import carregar_resumos_chbmit
from scripts.patient_balancing import subject_for_file


def audit(report, dataset_dir):
    annotations = carregar_resumos_chbmit(dataset_dir)
    result = {"historical_edfs_opened": False, "splits": {}}
    subjects = {}
    for split, key in (("training", "train_files"), ("calibration", "calibration_files")):
        files = report[key]
        subjects[split] = {subject_for_file(name) for name in files}
        if subjects[split] & {"chb09", "chb15", "chb18"}:
            raise ValueError("Historical patients must not be opened")
        records = []
        for name in files:
            if name not in annotations:
                raise ValueError(f"Missing annotation entry: {name}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with mne.io.read_raw_edf(dataset_dir / name, preload=False, verbose=False) as raw:
                    duration = raw.n_times / raw.info["sfreq"]
                    channels = len(raw.ch_names)
            intervals = annotations[name]
            if any(not 0 <= item.start_seconds < item.end_seconds <= duration for item in intervals):
                raise ValueError(f"Invalid annotated interval: {name}")
            records.append(dict(file=name, person=subject_for_file(name),
                                duration_seconds=duration, channel_slots=channels,
                                events=len(intervals),
                                ictal_seconds=sum(i.end_seconds-i.start_seconds for i in intervals)))
        total = sum(row["duration_seconds"] for row in records)
        ictal = sum(row["ictal_seconds"] for row in records)
        result["splits"][split] = dict(
            edfs=len(files), people=len(subjects[split]), events=sum(row["events"] for row in records),
            total_hours=total/3600, ictal_minutes=ictal/60, normal_hours=(total-ictal)/3600,
            edfs_without_seizure=sum(row["events"] == 0 for row in records), files=records,
        )
    if subjects["training"] & subjects["calibration"]:
        raise ValueError("Subject leakage across splits")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "modelos/v57_baseline_seed42_training.json")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_amostra")
    parser.add_argument("--output", type=Path, default=ROOT / "modelos/v58_cohort_audit.json")
    args = parser.parse_args()
    result = audit(json.loads(args.report.read_text(encoding="utf-8")), args.dataset_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for split, record in result["splits"].items():
        print(split, json.dumps({k: v for k, v in record.items() if k != "files"}))
