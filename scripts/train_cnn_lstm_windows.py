"""Treino preliminar CNN-LSTM com janelas temporais de EDF.

Este script e intencionalmente pequeno para TCC/prototipo:
- usa `dataset_amostra/`;
- cria janelas temporais sobrepostas;
- extrai as 19 features por janela;
- avalia CNN-LSTM com k-fold;
- treina um modelo final preliminar em todas as janelas;
- salva modelo `.keras`, scaler lateral e metricas JSON.

Exemplo:
    .venv\Scripts\python.exe scripts\train_cnn_lstm_windows.py --epochs 5
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
    FEATURE_MODE_MEAN,
    FEATURE_MODE_PER_CHANNEL,
    avaliar_kfold_cnn_lstm,
    carregar_resumos_chbmit,
    extrair_dataset_janelado_edf,
    parse_chbmit_summary,
    treinar_cnn_lstm_final,
)
from app.ai_engine.feature_extractor import extrair_metadados_edf  # noqa: E402


def carregar_manifesto(manifest_path: Path) -> list[str]:
    arquivos = [
        linha.strip()
        for linha in manifest_path.read_text(encoding="utf-8").splitlines()
        if linha.strip() and not linha.strip().startswith("#")
    ]
    if not arquivos:
        raise SystemExit(f"Manifesto vazio: {manifest_path}")
    return arquivos


def salvar_metadata_modelo(
    model_output: Path,
    *,
    feature_mode: str,
    canais_referencia: list[str] | None,
) -> Path:
    metadata_path = model_output.with_name(f"{model_output.stem}_metadata.json")
    payload = {
        "feature_mode": feature_mode,
        "canais_referencia": canais_referencia or [],
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata_path


def resolver_canais_referencia(
    dataset_dir: Path,
    *,
    edf_paths: list[Path],
    args: argparse.Namespace,
) -> list[str] | None:
    if args.feature_mode != FEATURE_MODE_PER_CHANNEL:
        return None

    referencia = (
        (dataset_dir / args.channel_reference_edf)
        if args.channel_reference_edf is not None
        else edf_paths[0]
    )
    if not referencia.exists():
        raise SystemExit(f"EDF de referencia nao encontrado: {referencia}")
    return list(extrair_metadados_edf(referencia)["canais_eeg"])


def carregar_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    dataset_dir = args.dataset_dir.resolve()
    if args.summary is not None:
        if not args.summary.exists():
            raise SystemExit(f"Arquivo summary nao encontrado: {args.summary}")
        intervalos_por_arquivo = parse_chbmit_summary(args.summary)
    else:
        intervalos_por_arquivo = carregar_resumos_chbmit(dataset_dir)

    arquivos_manifesto: list[str] | None = None
    if args.manifest is not None:
        if not args.manifest.exists():
            raise SystemExit(f"Manifesto nao encontrado: {args.manifest}")
        arquivos_manifesto = carregar_manifesto(args.manifest.resolve())

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    metadados: list[dict] = []

    if arquivos_manifesto is not None:
        edf_paths = []
        faltantes: list[str] = []
        for nome in arquivos_manifesto:
            caminho = dataset_dir / nome
            if caminho.exists():
                edf_paths.append(caminho)
            else:
                faltantes.append(nome)
        if faltantes:
            raise SystemExit(f"Arquivos do manifesto nao encontrados: {', '.join(faltantes)}")
    else:
        edf_paths = sorted(dataset_dir.glob("*.edf"))

    canais_referencia = resolver_canais_referencia(
        dataset_dir,
        edf_paths=edf_paths,
        args=args,
    )

    for edf_path in edf_paths:
        intervals = intervalos_por_arquivo.get(edf_path.name, [])
        x, y, meta = extrair_dataset_janelado_edf(
            edf_path,
            intervals,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            max_windows_per_class=args.max_windows_per_class,
            max_normal_windows=args.max_normal_windows_per_file,
            max_seizure_windows=args.max_seizure_windows_per_file,
            canais_selecionados=canais_referencia,
            feature_mode=args.feature_mode,
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

    args._canais_referencia = canais_referencia
    return np.vstack(x_parts), np.concatenate(y_parts), metadados


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treino preliminar CNN-LSTM com janelamento temporal.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=5.0)
    parser.add_argument("--max-windows-per-class", type=int, default=30)
    parser.add_argument("--max-normal-windows-per-file", type=int, default=None)
    parser.add_argument("--max-seizure-windows-per-file", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--final-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--feature-mode",
        choices=[FEATURE_MODE_MEAN, FEATURE_MODE_PER_CHANNEL],
        default=FEATURE_MODE_MEAN,
    )
    parser.add_argument("--channel-reference-edf", type=Path, default=None)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "modelos" / "cnn_lstm_hybrid_prelim.keras",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "modelos" / "metrics_cnn_lstm_windows.json",
    )
    args = parser.parse_args()

    x_all, y_all, metadados = carregar_dataset(args)
    metrics = avaliar_kfold_cnn_lstm(
        x_all,
        y_all,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    artefatos = treinar_cnn_lstm_final(
        x_all,
        y_all,
        output_path=args.model_output,
        epochs=args.final_epochs,
        batch_size=args.batch_size,
    )
    metadata_modelo = salvar_metadata_modelo(
        args.model_output,
        feature_mode=args.feature_mode,
        canais_referencia=getattr(args, "_canais_referencia", None),
    )

    metrics["dataset_dir"] = str(args.dataset_dir.resolve())
    metrics["window_seconds"] = args.window_seconds
    metrics["step_seconds"] = args.step_seconds
    metrics["augmentation"] = "overlapping_temporal_windows"
    metrics["max_windows_per_class"] = args.max_windows_per_class
    metrics["max_normal_windows_per_file"] = args.max_normal_windows_per_file
    metrics["max_seizure_windows_per_file"] = args.max_seizure_windows_per_file
    metrics["manifest"] = str(args.manifest.resolve()) if args.manifest else None
    metrics["feature_mode"] = args.feature_mode
    metrics["canais_referencia"] = getattr(args, "_canais_referencia", None)
    metrics["artifacts"] = artefatos
    metrics["model_metadata_path"] = str(metadata_modelo.resolve())
    metrics["windows_preview"] = metadados[:20]

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(metrics["mean"], indent=2, ensure_ascii=False))
    print(f"Modelo salvo em: {artefatos['model_path']}")
    print(f"Scaler salvo em: {artefatos['scaler_path']}")
    print(f"Metricas salvas em: {args.metrics_output}")


if __name__ == "__main__":
    main()
