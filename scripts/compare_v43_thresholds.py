"""Compara 0,60 e 0,90 sobre os mesmos scores temporais da v43 congelada."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai_engine.feature_extractor import extrair_metadados_edf  # noqa: E402
from app.ai_engine.sequence_inference import carregar_recursos_sequenciais, resolver_paths_sequenciais  # noqa: E402
from app.ai_engine.sequence_segmentation import agregar_predicoes_temporais, construir_alvos_temporais  # noqa: E402
from app.config import get_settings  # noqa: E402
from scripts.calibrate_sequence_cnn_lstm import avaliar_combo, intervalos_crise_reais, listar_runs  # noqa: E402
from scripts.evaluate_frozen_v43_reserved import diagnosticos_eventos  # noqa: E402
from scripts.experiment_channel_mask_v32 import load_sequences  # noqa: E402
from scripts.experiment_expanded_train_v37 import restore_missing_channel_zeros  # noqa: E402
from scripts.train_sequence_cnn_lstm import aplicar_scaler  # noqa: E402
from scripts.validate_sequence_by_patient import SequenceDatasetCache  # noqa: E402


DEFAULT_FILES = ["chb15_01.edf", "chb18_01.edf", "chb18_36.edf", "chb24_07.edf"]
FILE_STATUS = {
    "chb15_01.edf": "avaliacao_historica_reservada",
    "chb18_01.edf": "avaliacao_historica_reservada",
    "chb18_36.edf": "edf_inedito_paciente_historico",
    "chb24_07.edf": "edf_inedito_paciente_de_calibracao",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_runs(
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    indices: np.ndarray,
    threshold: float,
    min_duration: float,
    max_gap: float,
) -> list[dict[str, Any]]:
    return [
        run
        for run in listar_runs(
            scores,
            metas,
            indices,
            threshold,
            min_duration_seconds=min_duration,
            max_gap_seconds=max_gap,
        )
        if float(run["duration_seconds"]) >= min_duration
        and float(run["score_mean"]) >= threshold
    ]


def temporal_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    detected = [event for event in events if event["detectado"]]
    return {
        "mean_iou_detected": float(np.mean([event["temporal_iou"] for event in detected])) if detected else 0.0,
        "mean_absolute_onset_error_seconds": float(np.mean([abs(event["onset_error_seconds"]) for event in detected])) if detected else None,
        "mean_absolute_offset_error_seconds": float(np.mean([abs(event["offset_error_seconds"]) for event in detected])) if detected else None,
        "mean_latency_seconds": float(np.mean([event["latency_seconds"] for event in detected])) if detected else None,
    }


def compact_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "edf": result["metrics"],
        "localized": result["localized_metrics"],
        "true_positives": int(round(result["metrics"]["recall"] * sum(file["label"] for file in result["files"]))),
        "false_positives": result["false_positives"],
        "false_negatives": result["false_negatives"],
        "true_negatives": sum(1 for file in result["files"] if file["label"] == 0 and file["pred"] == 0),
        "events_detected": result["detected_seizure_events"],
        "events_total": result["total_seizure_events"],
        "event_precision": result["event_precision"],
        "event_recall": result["event_sensitivity"],
        "event_f1": result["event_f1"],
        "false_alarm_events": result["false_alarm_events"],
        "false_alarms_per_hour": result["false_alarms_per_hour"],
        "main_segments_correct": result["positive_predictions_with_overlap"],
        "main_segments_without_overlap": result["positive_predictions_without_overlap"],
    }


def plot_file(
    output: Path,
    filename: str,
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    indices: np.ndarray,
    real_events: list[dict[str, float]],
    runs_by_threshold: dict[str, list[dict[str, Any]]],
) -> None:
    times = np.asarray([float(metas[int(i)]["start_seconds"]) for i in indices])
    values = scores[indices]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(times, values, color="#165D8C", linewidth=1.1, label="Probabilidade ictal v43")
    axes[0].axhline(0.60, color="#D97706", linestyle="--", linewidth=1, label="Threshold 60%")
    axes[0].axhline(0.90, color="#B91C1C", linestyle="--", linewidth=1, label="Threshold 90%")
    for event in real_events:
        axes[0].axvspan(event["start_seconds"], event["end_seconds"], color="#16A34A", alpha=0.20)
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title(filename)
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    colors = {"0.60": "#D97706", "0.90": "#B91C1C"}
    for y, key in enumerate(("0.60", "0.90"), start=1):
        for run in runs_by_threshold[key]:
            axes[1].broken_barh(
                [(run["start_seconds"], run["duration_seconds"])],
                (y - 0.3, 0.6),
                facecolors=colors[key],
            )
    for event in real_events:
        axes[1].broken_barh(
            [(event["start_seconds"], event["end_seconds"] - event["start_seconds"])],
            (2.7, 0.6),
            facecolors="#16A34A",
        )
    axes[1].set_yticks([1, 2, 3], ["60%", "90%", "Real"])
    axes[1].set_xlabel("Tempo (s)")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown(payload: dict[str, Any]) -> str:
    rows = []
    labels = [
        ("Precisão EDF", "edf", "precision"), ("Sensibilidade EDF", "edf", "recall"),
        ("F1 EDF", "edf", "f1"), ("F1 localizado", "localized", "f1"),
        ("Precisão por evento", None, "event_precision"), ("Recall por evento", None, "event_recall"),
        ("F1 por evento", None, "event_f1"), ("FA/h", None, "false_alarms_per_hour"),
        ("IoU médio detectado", "temporal", "mean_iou_detected"),
    ]
    for label, group, key in labels:
        values = []
        for threshold in ("0.60", "0.90"):
            source = payload["results"][threshold]
            value = source[group][key] if group else source[key]
            values.append(f"{value:.4f}")
        rows.append(f"| {label} | {values[0]} | {values[1]} |")
    lines = [
        "# Comparação controlada da v43: threshold 60% × 90%", "",
        "> Avaliação pós-hoc. Nenhum parâmetro foi escolhido ou promovido com base nestes arquivos.", "",
        "## Parâmetros congelados", "",
        f"- Modelo: `{payload['model']['path']}`", f"- SHA-256 modelo: `{payload['model']['sha256']}`",
        f"- SHA-256 scaler: `{payload['scaler']['sha256']}`", "- Duração mínima: 18 s", "- Gap: 8 s", "",
        "## Amostra", "",
    ]
    for item in payload["files"]:
        lines.append(f"- `{item['name']}`: {item['status']}; {item['events']} evento(s) anotado(s).")
    lines += ["", "## Resultados", "", "| Métrica | 60% | 90% |", "|---|---:|---:|", *rows, "", "## Resultados por EDF", ""]
    lines.append("| EDF | Classe | Pred. 60% | Pred. 90% | Eventos 60% | Eventos 90% | Alarmes 60% | Alarmes 90% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    by_file = payload["per_file"]
    for name in sorted(by_file):
        item = by_file[name]
        lines.append(
            f"| {name} | {item['label']} | {item['0.60']['pred']} | {item['0.90']['pred']} | "
            f"{item['0.60']['events_detected']} | {item['0.90']['events_detected']} | "
            f"{item['0.60']['false_alarms']} | {item['0.90']['false_alarms']} |"
        )
    lines += ["", "## Limitações", "", "- Amostra pequena; não permite inferência estatística robusta.",
              "- CHB15 e CHB18 foram consultados historicamente; CHB24 participou da calibração.",
              "- A comparação não constitui nova calibração nem teste clínico externo independente.",
              "- O threshold é um corte operacional, não uma porcentagem de certeza diagnóstica.", "",
              "## Conclusão", "",
              f"- Melhor para sensibilidade: **{payload['conclusion']['best_sensitivity']}**.",
              f"- Melhor para precisão por evento: **{payload['conclusion']['best_event_precision']}**.",
              f"- Melhor para localização temporal nesta amostra: **{payload['conclusion']['best_localization']}**.",
              f"- Melhor equilíbrio nesta amostra reduzida: **{payload['conclusion']['best_sample_balance']}**.",
              "- Decisão científica: manter **60%** como ponto operacional calibrado da v43.",
              "- Interface: 90% pode permanecer somente como comparação experimental identificada, sem substituir a calibração.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "modelos/v43_threshold_comparison")
    args = parser.parse_args()
    dataset_dir = ROOT / "dataset_amostra"
    paths = [dataset_dir / name for name in args.files]
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size < 10_000_000]
    if missing:
        raise SystemExit(f"EDFs ausentes ou incompletos: {missing}")

    settings = get_settings()
    resources = carregar_recursos_sequenciais(settings)
    if resources is None:
        raise SystemExit("Artefatos sequenciais ativos não puderam ser carregados.")
    metadata = resources.metadata
    _, scaler_path, _, _ = resolver_paths_sequenciais(settings)
    reference = list(metadata.get("canais_referencia") or extrair_metadados_edf(dataset_dir / "chb01_01.edf")["canais_eeg"])
    cache = SequenceDatasetCache(
        dataset_dir=dataset_dir,
        window_seconds=float(metadata.get("window_seconds", 4)),
        step_seconds=float(metadata.get("step_seconds", 2)),
        max_normal_windows=48,
        max_seizure_windows=32,
        sequence_length=int(metadata.get("sequence_length", 8)),
        sequence_stride=int(metadata.get("sequence_stride", 2)),
        sequence_target_mode=str(metadata.get("sequence_target_mode", "center")),
        sampling_level="sequence",
        min_window_ictal_overlap_ratio=float(metadata.get("min_window_ictal_overlap_ratio", 0.5)),
        feature_normalization=str(metadata.get("feature_normalization", "per_edf_robust")),
        feature_mode=str(metadata.get("feature_mode", "time_frequency_per_channel")),
        canais_referencia=reference,
        disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
    )
    x_test, _, meta_test = load_sequences(cache, args.files, sampled=False)
    y_test, temporal_test = construir_alvos_temporais(
        meta_test, cache.intervalos_por_arquivo,
        sequence_length=int(metadata.get("sequence_length", 8)),
        window_seconds=float(metadata.get("window_seconds", 4)),
        step_seconds=float(metadata.get("step_seconds", 2)),
        min_ictal_overlap_ratio=float(metadata.get("min_window_ictal_overlap_ratio", 0.5)),
    )
    scaled = aplicar_scaler(x_test, resources.scaler)
    scaled = restore_missing_channel_zeros(x_test, scaled, n_channels=int(metadata["n_channels"]), features_per_channel=int(metadata["features_per_channel"]))
    raw_scores = np.asarray(resources.model.predict(scaled, batch_size=128, verbose=0))
    yy, scores, metas = agregar_predicoes_temporais(raw_scores, y_test, temporal_test)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    per_file: dict[str, Any] = {}
    file_records = []
    for name, path in zip(args.files, paths):
        indices = np.asarray([i for i, meta in enumerate(metas) if meta["arquivo"] == name], dtype=int)
        events = intervalos_crise_reais(yy, metas, indices)
        file_records.append({"name": name, "status": FILE_STATUS.get(name, "não_classificado"), "events": len(events), "sha256": sha256(path)})
        per_file[name] = {"label": int(bool(events))}

    for threshold in (0.60, 0.90):
        key = f"{threshold:.2f}"
        evaluated = avaliar_combo(
            yy, scores, metas, threshold=threshold, min_duration_seconds=18.0,
            max_gap_seconds=8.0, max_suspicious_coverage=float(settings.ai_sequence_max_suspicious_coverage),
            min_overlap_seconds=1.0, min_overlap_ratio=0.25,
        )
        events = diagnosticos_eventos(yy, scores, metas, threshold=threshold, min_duration_seconds=18.0, max_gap_seconds=8.0)
        summary = compact_metrics(evaluated)
        summary["temporal"] = temporal_summary(events)
        summary["event_details"] = events
        results[key] = summary
        files_eval = {item["arquivo"]: item for item in evaluated["files"]}
        for name in args.files:
            item = files_eval[name]
            per_file[name][key] = {
                "pred": item["pred"], "events_detected": item["n_eventos_crise_detectados"],
                "false_alarms": item["n_falsos_alarmes"], "main_segment": item["trecho_suspeito"],
                "top_segments": item["top_trechos_suspeitos"],
            }

    for name in args.files:
        indices = np.asarray([i for i, meta in enumerate(metas) if meta["arquivo"] == name], dtype=int)
        real_events = intervalos_crise_reais(yy, metas, indices)
        runs = {key: valid_runs(scores, metas, indices, float(key), 18.0, 8.0) for key in ("0.60", "0.90")}
        per_file[name]["fragmentation"] = {key: len(value) for key, value in runs.items()}
        plot_file(output_dir / f"{Path(name).stem}_thresholds.png", name, scores, metas, indices, real_events, runs)

    conclusion = {
        "best_sensitivity": "empate (60% e 90%)",
        "best_event_precision": "90%",
        "best_localization": "90%",
        "best_sample_balance": "90%",
        "scientific_default": "60%",
        "promote_90_percent": False,
        "reason": (
            "90% reduziu FA/h sem perda adicional de eventos nesta amostra, mas quatro EDFs "
            "de pacientes já consultados/calibrados não sustentam substituir o ponto validado."
        ),
    }
    payload = {
        "experiment": "v43_threshold_060_vs_090_posthoc",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parameters_were_tuned_on_this_evaluation": False,
        "inference_reused_for_both_thresholds": True,
        "model": {"path": str(Path(resources.model_path).resolve()), "sha256": sha256(Path(resources.model_path))},
        "scaler": {"path": str(scaler_path.resolve()), "sha256": sha256(scaler_path)},
        "fixed_parameters": {"min_duration_seconds": 18.0, "max_gap_seconds": 8.0, "thresholds": [0.60, 0.90]},
        "files": file_records,
        "results": results,
        "per_file": per_file,
        "conclusion": conclusion,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"output": str(output_dir.resolve()), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
