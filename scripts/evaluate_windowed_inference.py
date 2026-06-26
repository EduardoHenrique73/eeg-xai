"""Avalia inferencia janelada do modelo em EDFs com resumo CHB-MIT.

O objetivo e comparar o comportamento usado pela interface com arquivos
rotulados, incluindo distribuicao dos scores e top janelas suspeitas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai_engine.feature_extractor import (  # noqa: E402
    FEATURE_MODE_MEAN,
    FEATURE_MODE_PER_CHANNEL,
    extrair_features_edf_janelado,
)
from app.ai_engine.inference import obter_metadados_modelo, obter_modelo_keras, obter_scaler_modelo  # noqa: E402
from app.ai_engine.training import carregar_resumos_chbmit, window_overlaps_seizure  # noqa: E402
from app.services.exame_pipeline import _predizer_janelas  # noqa: E402


def avaliar_arquivo(
    *,
    dataset_dir: Path,
    arquivo: str,
    modelo_path: Path,
    feature_mode: str,
    canais_referencia: list[str] | None,
    window_seconds: float,
    step_seconds: float,
) -> dict[str, Any]:
    modelo = obter_modelo_keras(str(modelo_path))
    scaler = obter_scaler_modelo(str(modelo_path))
    intervalos = carregar_resumos_chbmit(dataset_dir).get(arquivo, [])
    janelas = extrair_features_edf_janelado(
        str(dataset_dir / arquivo),
        feature_mode=feature_mode,
        canais_referencia=canais_referencia,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )

    x = np.asarray([janela["feature_vector"] for janela in janelas], dtype=np.float32)
    if scaler is not None:
        x = scaler.transform(x)
    scores = np.asarray(modelo.predict(x.reshape(x.shape[0], x.shape[1], 1), verbose=0)).reshape(-1)
    score_agregado, janela_pico, top_janelas = _predizer_janelas(
        modelo=modelo,
        scaler=scaler,
        janelas=janelas,
    )

    labels = np.asarray(
        [
            int(
                window_overlaps_seizure(
                    float(janela["window_start_seconds"]),
                    float(janela["window_end_seconds"]),
                    intervalos,
                )
            )
            for janela in janelas
        ],
        dtype=np.int64,
    )

    resultado: dict[str, Any] = {
        "arquivo": arquivo,
        "n_janelas": int(len(janelas)),
        "n_janelas_crise": int(np.sum(labels == 1)),
        "score_agregado_top5_mean": float(score_agregado),
        "score_pico": float(top_janelas[0]["score"]),
        "janela_pico": {
            "start_seconds": float(janela_pico["window_start_seconds"]),
            "end_seconds": float(janela_pico["window_end_seconds"]),
            "label": int(
                window_overlaps_seizure(
                    float(janela_pico["window_start_seconds"]),
                    float(janela_pico["window_end_seconds"]),
                    intervalos,
                )
            ),
        },
        "percentis": {
            str(p): float(np.percentile(scores, p))
            for p in [50, 75, 90, 95, 97, 99, 100]
        },
        "proporcoes_acima": {
            str(c): float(np.mean(scores >= c))
            for c in [0.5, 0.7, 0.9, 0.95, 0.99]
        },
        "top_janelas": top_janelas,
    }
    if np.any(labels == 1):
        resultado["crise"] = {
            "score_medio": float(np.mean(scores[labels == 1])),
            "score_max": float(np.max(scores[labels == 1])),
            "prop_acima_0_9": float(np.mean(scores[labels == 1] >= 0.9)),
        }
        resultado["fora_crise"] = {
            "score_medio": float(np.mean(scores[labels == 0])),
            "score_max": float(np.max(scores[labels == 0])),
            "prop_acima_0_9": float(np.mean(scores[labels == 0] >= 0.9)),
        }
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia inferencia janelada em EDFs.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--step-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metadata = obter_metadados_modelo(str(args.model_path))
    feature_mode = str(metadata.get("feature_mode") or FEATURE_MODE_MEAN)
    canais_referencia = metadata.get("canais_referencia")
    if feature_mode != FEATURE_MODE_PER_CHANNEL:
        canais_referencia = None

    payload = {
        "model_path": str(args.model_path.resolve()),
        "feature_mode": feature_mode,
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "files": [
            avaliar_arquivo(
                dataset_dir=args.dataset_dir,
                arquivo=arquivo,
                modelo_path=args.model_path,
                feature_mode=feature_mode,
                canais_referencia=canais_referencia,
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
            )
            for arquivo in args.files
        ],
    }

    texto = json.dumps(payload, indent=2, ensure_ascii=False)
    print(texto)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
