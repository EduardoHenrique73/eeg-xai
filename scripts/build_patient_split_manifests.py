"""Gera manifests treino/calibracao/teste separados por paciente."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai_engine.training import carregar_resumos_chbmit  # noqa: E402


def patient_from_file(filename: str) -> str:
    match = re.match(r"(chb\d{2})", filename)
    if not match:
        raise ValueError(f"Nome de arquivo CHB-MIT invalido: {filename}")
    return match.group(1)


def parse_patients(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def write_manifest(path: Path, files: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(files) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria manifests por paciente para treino/calibracao/teste.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra" / "manifests")
    parser.add_argument("--calibration-patients", default="chb21,chb22,chb23,chb24")
    parser.add_argument("--test-patients", default="chb03,chb11,chb13,chb14")
    parser.add_argument("--prefix", default="patient_split")
    args = parser.parse_args()

    intervals = carregar_resumos_chbmit(args.dataset_dir)
    local_files = sorted(path.name for path in args.dataset_dir.glob("*.edf"))
    calibration_patients = parse_patients(args.calibration_patients)
    test_patients = parse_patients(args.test_patients)
    overlap = calibration_patients & test_patients
    if overlap:
        raise SystemExit(f"Paciente em calibracao e teste ao mesmo tempo: {sorted(overlap)}")

    train_files: list[str] = []
    calibration_files: list[str] = []
    test_files: list[str] = []
    summary: dict[str, dict[str, int]] = {}

    for filename in local_files:
        patient = patient_from_file(filename)
        bucket = summary.setdefault(patient, {"total": 0, "crisis": 0, "normal": 0})
        bucket["total"] += 1
        if intervals.get(filename):
            bucket["crisis"] += 1
        else:
            bucket["normal"] += 1

        if patient in test_patients:
            test_files.append(filename)
        elif patient in calibration_patients:
            calibration_files.append(filename)
        else:
            train_files.append(filename)

    files = {
        "train": train_files,
        "calibration": calibration_files,
        "test": test_files,
    }
    for name, manifest_files in files.items():
        if not manifest_files:
            raise SystemExit(f"Manifest {name} ficou vazio.")
        write_manifest(args.output_dir / f"{args.prefix}_{name}.txt", manifest_files)

    payload = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "calibration_patients": sorted(calibration_patients),
        "test_patients": sorted(test_patients),
        "counts": {name: len(manifest_files) for name, manifest_files in files.items()},
        "by_patient": dict(sorted(summary.items())),
        "manifests": {
            name: str((args.output_dir / f"{args.prefix}_{name}.txt").resolve())
            for name in files
        },
    }
    summary_path = args.output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
