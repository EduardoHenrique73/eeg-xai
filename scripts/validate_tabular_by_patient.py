"""Valida modelos tabulares com separacao por paciente.

Esta avaliacao responde uma pergunta especifica: as features atuais generalizam
para pacientes fora do treino? O resultado mede desempenho por janela e por EDF.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai_engine.training import (  # noqa: E402
    FEATURE_MODE_MEAN,
    FEATURE_MODE_PER_CHANNEL,
    carregar_resumos_chbmit,
    extrair_dataset_janelado_edf,
)
from app.ai_engine.feature_extractor import extrair_metadados_edf  # noqa: E402


def carregar_manifesto(path: Path) -> list[str]:
    arquivos = [
        linha.strip()
        for linha in path.read_text(encoding="utf-8").splitlines()
        if linha.strip() and not linha.strip().startswith("#")
    ]
    if not arquivos:
        raise SystemExit(f"Manifesto vazio: {path}")
    return arquivos


def paciente_do_arquivo(nome: str) -> str:
    match = re.match(r"(chb\d+)", nome.lower())
    if not match:
        raise ValueError(f"Nao foi possivel identificar paciente em {nome}")
    return match.group(1)


def criar_modelo(nome: str, random_state: int) -> Any:
    if nome == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=random_state,
        )
    if nome == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            l2_regularization=0.05,
            random_state=random_state,
        )
    if nome == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
    raise ValueError(f"Modelo tabular desconhecido: {nome}")


def metricas_binarias(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def extrair_score(modelo: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(modelo, "predict_proba"):
        return np.asarray(modelo.predict_proba(x))[:, 1]
    if hasattr(modelo, "decision_function"):
        decisao = np.asarray(modelo.decision_function(x), dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-decisao))
    pred = np.asarray(modelo.predict(x), dtype=np.float32)
    return pred


def melhor_trecho_continuo(
    metas: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int | bool]:
    melhor: dict[str, float | int | bool] | None = None
    inicio_run: int | None = None

    def fechar_run(fim_exclusivo: int) -> None:
        nonlocal melhor, inicio_run
        if inicio_run is None:
            return
        indices = np.arange(inicio_run, fim_exclusivo)
        inicio = float(metas[int(indices[0])]["start_seconds"])
        fim = float(metas[int(indices[-1])]["end_seconds"])
        candidato = {
            "start_seconds": inicio,
            "end_seconds": fim,
            "duration_seconds": float(max(0.0, fim - inicio)),
            "n_janelas": int(len(indices)),
            "score_medio": float(np.mean(scores[indices])),
            "score_max": float(np.max(scores[indices])),
            "threshold": float(threshold),
        }
        if melhor is None:
            melhor = candidato
        else:
            chave_candidato = (
                float(candidato["duration_seconds"]),
                float(candidato["score_medio"]),
                float(candidato["score_max"]),
            )
            chave_melhor = (
                float(melhor["duration_seconds"]),
                float(melhor["score_medio"]),
                float(melhor["score_max"]),
            )
            if chave_candidato > chave_melhor:
                melhor = candidato
        inicio_run = None

    for idx, score in enumerate(scores):
        if float(score) >= threshold:
            if inicio_run is None:
                inicio_run = idx
        else:
            fechar_run(idx)
    fechar_run(len(scores))

    if melhor is not None:
        return melhor

    idx = int(np.argmax(scores))
    return {
        "start_seconds": float(metas[idx]["start_seconds"]),
        "end_seconds": float(metas[idx]["end_seconds"]),
        "duration_seconds": float(float(metas[idx]["end_seconds"]) - float(metas[idx]["start_seconds"])),
        "n_janelas": 1,
        "score_medio": float(scores[idx]),
        "score_max": float(scores[idx]),
        "threshold": float(threshold),
    }


def carregar_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    dataset_dir = args.dataset_dir.resolve()
    intervalos_por_arquivo = carregar_resumos_chbmit(dataset_dir)
    arquivos = carregar_manifesto(args.manifest)
    canais_referencia = None
    if args.feature_mode == FEATURE_MODE_PER_CHANNEL:
        ref_path = dataset_dir / (args.channel_reference_edf or arquivos[0])
        canais_referencia = list(extrair_metadados_edf(ref_path)["canais_eeg"])

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []

    for arquivo in arquivos:
        path = dataset_dir / arquivo
        if not path.exists():
            raise SystemExit(f"EDF do manifesto nao encontrado: {arquivo}")
        x, y, meta = extrair_dataset_janelado_edf(
            path,
            intervalos_por_arquivo.get(arquivo, []),
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
        paciente = paciente_do_arquivo(arquivo)
        for item in meta:
            item["paciente"] = paciente
            item["arquivo"] = arquivo
        x_parts.append(x)
        y_parts.append(y)
        metas.extend(meta)
        print(
            f"{arquivo}: {len(y)} janelas "
            f"(normal={int(np.sum(y == 0))}, crise={int(np.sum(y == 1))})"
        )

    if not x_parts:
        raise SystemExit("Nenhuma janela gerada.")
    return np.vstack(x_parts), np.concatenate(y_parts), metas


def avaliar_por_edf(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    window_threshold: float,
    min_duration_seconds: float,
) -> dict[str, Any]:
    por_arquivo: dict[str, list[int]] = defaultdict(list)
    for idx, meta in enumerate(metas):
        por_arquivo[str(meta["arquivo"])].append(idx)

    linhas: list[dict[str, Any]] = []
    y_file_true: list[int] = []
    y_file_pred: list[int] = []

    for arquivo, indices in sorted(por_arquivo.items()):
        idxs = np.asarray(indices, dtype=int)
        trecho = melhor_trecho_continuo(
            [metas[int(i)] for i in idxs],
            scores[idxs],
            threshold=window_threshold,
        )
        label = int(np.any(y_true[idxs] == 1))
        pred = int(float(trecho["duration_seconds"]) >= min_duration_seconds)
        y_file_true.append(label)
        y_file_pred.append(pred)
        linhas.append(
            {
                "arquivo": arquivo,
                "paciente": str(metas[int(idxs[0])]["paciente"]),
                "label": label,
                "pred": pred,
                "n_janelas": int(len(idxs)),
                "n_janelas_crise": int(np.sum(y_true[idxs] == 1)),
                "score_max": float(np.max(scores[idxs])),
                "score_medio": float(np.mean(scores[idxs])),
                "trecho_suspeito": trecho,
            }
        )

    return {
        "metrics": metricas_binarias(np.asarray(y_file_true), np.asarray(y_file_pred)),
        "files": linhas,
    }


def avaliar_modelo(args: argparse.Namespace, modelo_nome: str) -> dict[str, Any]:
    x, y, metas = carregar_dataset(args)
    pacientes = np.asarray([str(meta["paciente"]) for meta in metas])
    pacientes_unicos = sorted(set(pacientes))
    folds: list[dict[str, Any]] = []

    for fold_idx, paciente_teste in enumerate(pacientes_unicos, start=1):
        test_mask = pacientes == paciente_teste
        train_mask = ~test_mask
        if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
            continue

        modelo = criar_modelo(modelo_nome, args.random_state + fold_idx)
        modelo.fit(x[train_mask], y[train_mask])
        scores = extrair_score(modelo, x[test_mask])
        pred = (scores >= args.window_threshold).astype(np.int64)
        metas_teste = [meta for meta, selecionado in zip(metas, test_mask, strict=False) if selecionado]
        y_teste = y[test_mask]

        fold = {
            "paciente_teste": paciente_teste,
            "n_train": int(np.sum(train_mask)),
            "n_test": int(np.sum(test_mask)),
            "class_counts_test": {
                "0": int(np.sum(y_teste == 0)),
                "1": int(np.sum(y_teste == 1)),
            },
            "window_metrics": metricas_binarias(y_teste, pred),
            "edf_metrics": avaliar_por_edf(
                y_true=y_teste,
                scores=scores,
                metas=metas_teste,
                window_threshold=args.window_threshold,
                min_duration_seconds=args.min_duration_seconds,
            ),
        }
        folds.append(fold)
        print(
            f"{modelo_nome} / {paciente_teste}: "
            f"janela f1={fold['window_metrics']['f1']:.3f}, "
            f"EDF f1={fold['edf_metrics']['metrics']['f1']:.3f}"
        )

    if not folds:
        raise SystemExit(f"Nenhum fold valido para {modelo_nome}.")

    return {
        "model": modelo_nome,
        "n_folds": len(folds),
        "window_mean": {
            key: float(np.mean([fold["window_metrics"][key] for fold in folds]))
            for key in ["accuracy", "precision", "recall", "f1"]
        },
        "edf_mean": {
            key: float(np.mean([fold["edf_metrics"]["metrics"][key] for fold in folds]))
            for key in ["accuracy", "precision", "recall", "f1"]
        },
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validacao tabular leave-one-patient-out.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "dataset_amostra" / "curated_train_files_expanded_23ch.txt",
    )
    parser.add_argument("--models", nargs="+", default=["random_forest", "hist_gradient_boosting"])
    parser.add_argument("--feature-mode", choices=[FEATURE_MODE_MEAN, FEATURE_MODE_PER_CHANNEL], default=FEATURE_MODE_MEAN)
    parser.add_argument("--channel-reference-edf", default=None)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--step-seconds", type=float, default=2.0)
    parser.add_argument("--max-windows-per-class", type=int, default=None)
    parser.add_argument("--max-normal-windows-per-file", type=int, default=180)
    parser.add_argument("--max-seizure-windows-per-file", type=int, default=30)
    parser.add_argument("--window-threshold", type=float, default=0.5)
    parser.add_argument("--min-duration-seconds", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "modelos" / "tabular_leave_one_patient_eval.json",
    )
    args = parser.parse_args()

    resultados = [avaliar_modelo(args, nome) for nome in args.models]
    payload = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "feature_mode": args.feature_mode,
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "max_windows_per_class": args.max_windows_per_class,
        "max_normal_windows_per_file": args.max_normal_windows_per_file,
        "max_seizure_windows_per_file": args.max_seizure_windows_per_file,
        "window_threshold": args.window_threshold,
        "min_duration_seconds": args.min_duration_seconds,
        "results": resultados,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({r["model"]: {"window_mean": r["window_mean"], "edf_mean": r["edf_mean"]} for r in resultados}, indent=2, ensure_ascii=False))
    print(f"Metricas salvas em: {args.output}")


if __name__ == "__main__":
    main()
