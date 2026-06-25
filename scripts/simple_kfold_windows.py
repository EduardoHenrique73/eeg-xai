"""Executa um pipeline simples de janelamento + k-fold sobre EDFs de amostra.

Exemplo:
    .venv\Scripts\python.exe scripts\simple_kfold_windows.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai_engine.training import (  # noqa: E402
    avaliar_kfold_features,
    extrair_dataset_janelado_edf,
    parse_chbmit_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data augmentation simples por janelamento e k-fold em EDFs.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=5.0)
    parser.add_argument("--max-windows-per-class", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "modelos" / "metrics_kfold_simple.json",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    summary_path = args.summary or next(dataset_dir.glob("*-summary.txt"), None)
    if summary_path is None or not summary_path.exists():
        raise SystemExit(f"Arquivo summary nao encontrado em {dataset_dir}")

    intervalos_por_arquivo = parse_chbmit_summary(summary_path)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    metadados: list[dict] = []

    for edf_path in sorted(dataset_dir.glob("*.edf")):
        intervals = intervalos_por_arquivo.get(edf_path.name, [])
        x, y, meta = extrair_dataset_janelado_edf(
            edf_path,
            intervals,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            max_windows_per_class=args.max_windows_per_class,
        )
        if len(y) == 0:
            continue
        x_parts.append(x)
        y_parts.append(y)
        metadados.extend(meta)
        print(
            f"{edf_path.name}: {len(y)} janelas "
            f"(normal={int(np.sum(y == 0))}, crise={int(np.sum(y == 1))})"
        )

    if not x_parts:
        raise SystemExit("Nenhuma janela gerada.")

    x_all = np.vstack(x_parts)
    y_all = np.concatenate(y_parts)
    metrics = avaliar_kfold_features(x_all, y_all, n_splits=args.n_splits)
    metrics["dataset_dir"] = str(dataset_dir)
    metrics["summary"] = str(summary_path.resolve())
    metrics["window_seconds"] = args.window_seconds
    metrics["step_seconds"] = args.step_seconds
    metrics["augmentation"] = "overlapping_temporal_windows"
    metrics["windows_preview"] = metadados[:20]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(metrics["mean"], indent=2, ensure_ascii=False))
    print(f"Metricas salvas em: {args.output}")


if __name__ == "__main__":
    main()
