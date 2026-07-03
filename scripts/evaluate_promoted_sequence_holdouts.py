"""Avalia o modelo sequencial ativo em EDFs locais fora do manifesto de treino."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai_engine.sequence_inference import (  # noqa: E402
    analisar_exame_sequencial,
    carregar_recursos_sequenciais,
    limpar_cache_sequencial,
)
from app.ai_engine.training import carregar_resumos_chbmit  # noqa: E402
from app.config import get_settings  # noqa: E402


DEFAULT_FILES = [
    "chb11_01.edf",
    "chb13_47.edf",
    "chb15_01.edf",
    "chb13_40.edf",
    "chb15_06.edf",
    "chb20_12.edf",
]


def carregar_manifesto(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        linha.strip()
        for linha in path.read_text(encoding="utf-8").splitlines()
        if linha.strip() and not linha.strip().startswith("#")
    }


def overlap_seconds(a: dict[str, Any] | None, intervalos: list[Any]) -> float:
    if not a:
        return 0.0
    total = 0.0
    for intervalo in intervalos:
        inicio = max(float(a["start_seconds"]), float(intervalo.start_seconds))
        fim = min(float(a["end_seconds"]), float(intervalo.end_seconds))
        total += max(0.0, fim - inicio)
    return float(total)


def anexar_overlap_trechos(
    trechos: list[dict[str, Any]],
    intervalos: list[Any],
) -> list[dict[str, Any]]:
    return [
        {
            **trecho,
            "overlap_crise_real_seconds": overlap_seconds(trecho, intervalos),
        }
        for trecho in trechos
    ]


def metricas_binarias(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == 1 and p == 0)
    total = max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float((tp + tn) / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o modelo sequencial ativo em holdouts locais.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "dataset_amostra" / "curated_train_files_expanded_23ch.txt",
    )
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "modelos" / "promoted_sequence_holdout_eval.json",
    )
    args = parser.parse_args()

    settings = get_settings()
    limpar_cache_sequencial()
    recursos = carregar_recursos_sequenciais(settings)
    if recursos is None:
        raise SystemExit("Modelo sequencial ativo nao foi carregado.")

    intervalos_por_arquivo = carregar_resumos_chbmit(args.dataset_dir)
    train_files = carregar_manifesto(args.manifest)

    linhas: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for arquivo in args.files:
        path = args.dataset_dir / arquivo
        if not path.exists():
            raise SystemExit(f"EDF nao encontrado: {path}")

        intervalos = intervalos_por_arquivo.get(arquivo, [])
        label = int(bool(intervalos))
        resultado = analisar_exame_sequencial(str(path), recursos=recursos, settings=settings)
        trecho = resultado.get("trecho_suspeito") or {}
        pred = int(bool(resultado.get("resultado_positivo_conclusivo")))
        overlap = overlap_seconds(trecho, intervalos)
        resultado_conclusivo = bool(resultado.get("resultado_conclusivo"))
        top_trechos = anexar_overlap_trechos(
            list(resultado.get("top_trechos_suspeitos", [])),
            intervalos,
        )
        top_localizou = bool(
            label
            and pred
            and any(float(item.get("overlap_crise_real_seconds", 0.0)) > 0 for item in top_trechos)
        )

        y_true.append(label)
        y_pred.append(pred)
        linhas.append(
            {
                "arquivo": arquivo,
                "usado_no_treino": arquivo in train_files,
                "label": label,
                "pred": pred,
                "montagem_incompleta": bool(resultado.get("montagem_incompleta")),
                "cobertura_excessiva": bool(resultado.get("cobertura_excessiva")),
                "resultado_conclusivo": resultado_conclusivo,
                "canais_processados": resultado.get("canais_processados", []),
                "canais_omitidos": resultado.get("canais_omitidos", []),
                "crises_reais": [
                    {
                        "start_seconds": float(intervalo.start_seconds),
                        "end_seconds": float(intervalo.end_seconds),
                    }
                    for intervalo in intervalos
                ],
                "overlap_crise_real_seconds": overlap,
                "localizou_crise": bool(label and pred and overlap > 0),
                "localizou_crise_top_trechos": top_localizou,
                "classificacao_clinica": resultado.get("classificacao_clinica"),
                "score_geral": resultado.get("score_geral"),
                "threshold": resultado.get("threshold"),
                "min_duration_seconds": resultado.get("min_duration_seconds"),
                "feature_mode": resultado.get("feature_mode"),
                "n_sequences_analisadas": resultado.get("n_sequences_analisadas"),
                "trecho_suspeito": trecho,
                "top_trechos_suspeitos": top_trechos,
            }
        )
        print(
            f"{arquivo}: label={label} pred={pred} "
            f"score={float(resultado.get('score_geral', 0.0)):.4f} overlap={overlap:.1f}s"
        )

    metrics = metricas_binarias(y_true, y_pred)
    conclusivos = [item for item in linhas if item["resultado_conclusivo"]]
    metrics_conclusivos = (
        metricas_binarias(
            [int(item["label"]) for item in conclusivos],
            [int(item["pred"]) for item in conclusivos],
        )
        if conclusivos
        else None
    )
    payload = {
        "model_path": str(Path(recursos.model_path).resolve()),
        "metadata": recursos.metadata,
        "calibration": {
            "path": str(settings.ai_sequence_calibration_path) if settings.ai_sequence_calibration_path else None,
            "best": (recursos.calibration or {}).get("best") if recursos.calibration else None,
            "best_aligned": (recursos.calibration or {}).get("best_aligned") if recursos.calibration else None,
        },
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "files": linhas,
        "metrics": metrics,
        "metrics_conclusivos": metrics_conclusivos,
        "total": {
            "arquivos": len(linhas),
            "arquivos_conclusivos": len(conclusivos),
            "arquivos_limitados": len(linhas) - len(conclusivos),
            "arquivos_crise": int(sum(y_true)),
            "arquivos_normais": int(len(y_true) - sum(y_true)),
            "crises_localizadas": int(sum(1 for item in linhas if item["localizou_crise"])),
            "crises_localizadas_top_trechos": int(
                sum(1 for item in linhas if item["localizou_crise_top_trechos"])
            ),
            "crises_localizadas_conclusivas": int(
                sum(1 for item in conclusivos if item["localizou_crise"])
            ),
            "crises_localizadas_top_trechos_conclusivas": int(
                sum(1 for item in conclusivos if item["localizou_crise_top_trechos"])
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": metrics,
                "metrics_conclusivos": metrics_conclusivos,
                "total": payload["total"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
