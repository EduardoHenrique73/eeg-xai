"""Executa experimento intra-paciente em varios pacientes.

Para cada paciente com dados suficientes, o script treina modelos pequenos
CNN-LSTM sequenciais e testa:
1. um EDF com crise deixado fora do treino;
2. um EDF normal deixado fora do treino.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai_engine.training import carregar_resumos_chbmit  # noqa: E402
from scripts.train_chb01_controlled_sequence import treinar_e_avaliar_split  # noqa: E402


def agrupar_edfs(dataset_dir: Path) -> dict[str, dict[str, list[str]]]:
    intervalos = carregar_resumos_chbmit(dataset_dir)
    pacientes: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"crise": [], "normal": []})
    for path in sorted(dataset_dir.glob("*.edf")):
        paciente = path.stem.split("_")[0]
        if intervalos.get(path.name):
            pacientes[paciente]["crise"].append(path.name)
        else:
            pacientes[paciente]["normal"].append(path.name)
    return dict(pacientes)


def montar_splits(
    paciente: str,
    arquivos: dict[str, list[str]],
    *,
    max_train_seizure_files: int,
) -> list[dict[str, Any]]:
    crise = sorted(arquivos["crise"])
    normal = sorted(arquivos["normal"])
    if len(crise) < 2 or len(normal) < 1:
        return []

    crise_holdout = crise[-1]
    crise_treino = crise[:-1][:max_train_seizure_files]
    normal_holdout = normal[0]
    normal_treino = normal[1:2]

    splits = [
        {
            "nome": f"{paciente}_crise_holdout_{Path(crise_holdout).stem}",
            "tipo": "crise_holdout",
            "train_files": normal_treino + [normal_holdout] + crise_treino,
            "holdout_files": [crise_holdout],
        },
        {
            "nome": f"{paciente}_normal_holdout_{Path(normal_holdout).stem}",
            "tipo": "normal_holdout",
            "train_files": normal_treino + crise[:max_train_seizure_files],
            "holdout_files": [normal_holdout],
        },
    ]
    return [split for split in splits if len(set(split["train_files"])) >= 2]


def arquivo_resultado(split: dict[str, Any]) -> dict[str, Any]:
    files = split["holdout"]["edf_eval"]["files"]
    if not files:
        return {
            "pred": None,
            "label": None,
            "overlap_crise_real_seconds": 0.0,
            "trecho_suspeito": None,
        }
    file_eval = files[0]
    trecho = file_eval["trecho_suspeito"]
    return {
        "arquivo": file_eval["arquivo"],
        "label": file_eval["label"],
        "pred": file_eval["pred"],
        "overlap_crise_real_seconds": trecho["overlap_crise_real_seconds"],
        "trecho_suspeito": {
            "start_seconds": trecho["start_seconds"],
            "end_seconds": trecho["end_seconds"],
            "duration_seconds": trecho["duration_seconds"],
            "score_mean": trecho["score_mean"],
            "score_max": trecho["score_max"],
        },
        "crise_real": file_eval["crise_real"],
    }


def resumir_paciente(paciente: str, resultados: list[dict[str, Any]]) -> dict[str, Any]:
    crise_split = next((item for item in resultados if item["split_type"] == "crise_holdout"), None)
    normal_split = next((item for item in resultados if item["split_type"] == "normal_holdout"), None)
    crise_result = arquivo_resultado(crise_split) if crise_split else {}
    normal_result = arquivo_resultado(normal_split) if normal_split else {}
    localizou_crise = bool(
        crise_result
        and crise_result.get("label") == 1
        and crise_result.get("pred") == 1
        and float(crise_result.get("overlap_crise_real_seconds", 0.0)) > 0
    )
    falso_positivo_normal = bool(
        normal_result
        and normal_result.get("label") == 0
        and normal_result.get("pred") == 1
    )
    return {
        "paciente": paciente,
        "teste_crise_arquivo": crise_result.get("arquivo"),
        "teste_crise_pred": crise_result.get("pred"),
        "localizou_crise_real": localizou_crise,
        "overlap_crise_real_seconds": crise_result.get("overlap_crise_real_seconds"),
        "teste_normal_arquivo": normal_result.get("arquivo"),
        "teste_normal_pred": normal_result.get("pred"),
        "falso_positivo_normal": falso_positivo_normal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa CNN-LSTM sequencial intra-paciente em varios pacientes.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--step-seconds", type=float, default=2.0)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=2)
    parser.add_argument("--max-normal-windows-per-file", type=int, default=260)
    parser.add_argument("--max-seizure-windows-per-file", type=int, default=80)
    parser.add_argument("--max-train-seizure-files", type=int, default=999)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-duration-seconds", type=float, default=45.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "modelos" / "controlled_by_patient_sequence_eval.json")
    args = parser.parse_args()

    pacientes = agrupar_edfs(args.dataset_dir)
    selected_patients = args.patients or sorted(pacientes)
    resultados_por_split: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for paciente in selected_patients:
        arquivos = pacientes.get(paciente)
        if not arquivos:
            skipped.append({"paciente": paciente, "motivo": "sem EDF baixado"})
            continue
        splits = montar_splits(
            paciente,
            arquivos,
            max_train_seizure_files=args.max_train_seizure_files,
        )
        if not splits:
            skipped.append(
                {
                    "paciente": paciente,
                    "motivo": "requer pelo menos 2 EDFs com crise e 1 EDF normal",
                    "crise": arquivos["crise"],
                    "normal": arquivos["normal"],
                }
            )
            continue

        for split in splits:
            resultado = treinar_e_avaliar_split(
                dataset_dir=args.dataset_dir,
                nome=split["nome"],
                train_files=split["train_files"],
                holdout_files=split["holdout_files"],
                sequence_length=args.sequence_length,
                sequence_stride=args.sequence_stride,
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
                max_normal_windows_per_file=args.max_normal_windows_per_file,
                max_seizure_windows_per_file=args.max_seizure_windows_per_file,
                epochs=args.epochs,
                batch_size=args.batch_size,
                threshold=args.threshold,
                min_duration_seconds=args.min_duration_seconds,
                random_state=args.random_state,
                verbose=0,
            )
            resultado["paciente"] = paciente
            resultado["split_type"] = split["tipo"]
            resultados_por_split.append(resultado)

    por_paciente: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for resultado in resultados_por_split:
        por_paciente[resultado["paciente"]].append(resultado)
    resumo = [resumir_paciente(paciente, splits) for paciente, splits in sorted(por_paciente.items())]

    payload = {
        "experiment": "controlled_by_patient_sequence_cnn_lstm",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "threshold": args.threshold,
        "min_duration_seconds": args.min_duration_seconds,
        "epochs": args.epochs,
        "summary": resumo,
        "skipped": skipped,
        "splits": resultados_por_split,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": resumo, "skipped": skipped, "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
