"""Grade pós-hoc de threshold e duração somente na calibração congelada da v43."""

from __future__ import annotations

from datetime import datetime
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
from scripts.calibrate_sequence_cnn_lstm import avaliar_combo  # noqa: E402
from scripts.compare_v43_thresholds import sha256  # noqa: E402
from scripts.experiment_channel_mask_v32 import load_sequences  # noqa: E402
from scripts.experiment_expanded_train_v37 import restore_missing_channel_zeros  # noqa: E402
from scripts.train_sequence_cnn_lstm import aplicar_scaler  # noqa: E402
from scripts.validate_sequence_by_patient import SequenceDatasetCache  # noqa: E402


CALIBRATION_FILES = [
    "chb16_10.edf", "chb16_11.edf", "chb16_17.edf",
    "chb19_28.edf", "chb19_29.edf", "chb19_30.edf",
    "chb20_01.edf", "chb20_12.edf", "chb20_13.edf", "chb20_14.edf",
    "chb24_01.edf", "chb24_03.edf", "chb24_04.edf",
]
THRESHOLDS = (0.60, 0.90)
DURATIONS = (10.0, 14.0, 18.0)
MAX_GAP_SECONDS = 8.0
FA_CAP = 0.75


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": result["threshold"],
        "min_duration_seconds": result["min_duration_seconds"],
        "edf_precision": result["metrics"]["precision"],
        "edf_recall": result["metrics"]["recall"],
        "edf_f1": result["metrics"]["f1"],
        "localized_f1": result["localized_metrics"]["f1"],
        "event_precision": result["event_precision"],
        "event_recall": result["event_sensitivity"],
        "event_f1": result["event_f1"],
        "detected_events": result["detected_seizure_events"],
        "total_events": result["total_seizure_events"],
        "false_alarm_events": result["false_alarm_events"],
        "false_alarms_per_hour": result["false_alarms_per_hour"],
        "false_positives_edf": result["false_positives"],
        "false_negatives_edf": result["false_negatives"],
        "main_segments_correct": result["positive_predictions_with_overlap"],
        "main_segments_without_overlap": result["positive_predictions_without_overlap"],
        "constraint_fa_h_le_075": result["false_alarms_per_hour"] <= FA_CAP + 1e-12,
    }


def choose_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [row for row in rows if row["constraint_fa_h_le_075"]]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (
            row["event_recall"], row["localized_f1"], row["event_f1"],
            row["edf_f1"], -row["false_alarms_per_hour"], row["min_duration_seconds"],
        ),
    )


def plot_grid(rows: list[dict[str, Any]], output: Path) -> None:
    metrics = [
        ("event_recall", "Recall por evento"),
        ("event_f1", "F1 por evento"),
        ("localized_f1", "F1 localizado"),
        ("false_alarms_per_hour", "Falsos alarmes por hora"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (key, title) in zip(axes.flat, metrics):
        for threshold, color in ((0.60, "#D97706"), (0.90, "#B91C1C")):
            subset = sorted((row for row in rows if row["threshold"] == threshold), key=lambda row: row["min_duration_seconds"])
            axis.plot(
                [row["min_duration_seconds"] for row in subset],
                [row[key] for row in subset], marker="o", linewidth=2,
                color=color, label=f"threshold {threshold:.0%}",
            )
        if key == "false_alarms_per_hour":
            axis.axhline(FA_CAP, linestyle="--", color="#374151", linewidth=1, label="teto 0,75")
        axis.set_title(title)
        axis.set_xlabel("Duração mínima (s)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("v43 — grade de pós-processamento na calibração", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_report(payload: dict[str, Any]) -> str:
    lines = [
        "# v43 — experimento de duração mínima na calibração", "",
        "> Análise exploratória restrita aos pacientes de calibração. O modelo e o teste histórico não foram alterados.", "",
        "## Grade", "", "| Threshold | Duração | F1 EDF | F1 localizado | Eventos | Recall evento | F1 evento | FA/h | Teto atendido |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in payload["grid"]:
        lines.append(
            f"| {row['threshold']:.0%} | {row['min_duration_seconds']:.0f}s | {row['edf_f1']:.4f} | "
            f"{row['localized_f1']:.4f} | {row['detected_events']}/{row['total_events']} | "
            f"{row['event_recall']:.4f} | {row['event_f1']:.4f} | {row['false_alarms_per_hour']:.4f} | "
            f"{'sim' if row['constraint_fa_h_le_075'] else 'não'} |"
        )
    lines += ["", "## Decisão", ""]
    candidate = payload["best_feasible_candidate"]
    if candidate:
        lines += [
            f"Melhor combinação dentro do teto de FA/h: **threshold {candidate['threshold']:.0%}, duração {candidate['min_duration_seconds']:.0f}s**.", "",
            "Esse resultado é apenas candidato de calibração. Uma eventual promoção exige nova avaliação congelada que não reutilize os EDFs pós-hoc já consultados.",
        ]
    else:
        lines.append("Nenhuma combinação satisfez o teto original de 0,75 FA/h.")
    lines += ["", "## Salvaguardas", "", "- Gap fixo em 8 s.", "- Uma única inferência reutilizada em toda a grade.",
              "- CHB09, CHB15 e CHB18 não foram usados nesta escolha.", "- O artefato ativo da v43 não foi modificado.", ""]
    return "\n".join(lines)


def main() -> None:
    output_dir = ROOT / "modelos/v43_duration_grid_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = ROOT / "dataset_amostra"
    missing = [name for name in CALIBRATION_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise SystemExit(f"EDFs de calibração ausentes: {missing}")

    settings = get_settings()
    resources = carregar_recursos_sequenciais(settings)
    if resources is None:
        raise SystemExit("Artefatos da v43 indisponíveis.")
    metadata = resources.metadata
    _, scaler_path, _, calibration_path = resolver_paths_sequenciais(settings)
    reference = list(metadata.get("canais_referencia") or extrair_metadados_edf(dataset_dir / "chb01_01.edf")["canais_eeg"])
    cache = SequenceDatasetCache(
        dataset_dir=dataset_dir,
        window_seconds=float(metadata.get("window_seconds", 4)), step_seconds=float(metadata.get("step_seconds", 2)),
        max_normal_windows=48, max_seizure_windows=32,
        sequence_length=int(metadata.get("sequence_length", 8)), sequence_stride=int(metadata.get("sequence_stride", 2)),
        sequence_target_mode=str(metadata.get("sequence_target_mode", "center")), sampling_level="sequence",
        min_window_ictal_overlap_ratio=float(metadata.get("min_window_ictal_overlap_ratio", 0.5)),
        feature_normalization=str(metadata.get("feature_normalization", "per_edf_robust")),
        feature_mode=str(metadata.get("feature_mode", "time_frequency_per_channel")),
        canais_referencia=reference, disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
    )
    x_data, _, meta_data = load_sequences(cache, CALIBRATION_FILES, sampled=False)
    y_data, temporal = construir_alvos_temporais(
        meta_data, cache.intervalos_por_arquivo,
        sequence_length=int(metadata.get("sequence_length", 8)), window_seconds=float(metadata.get("window_seconds", 4)),
        step_seconds=float(metadata.get("step_seconds", 2)),
        min_ictal_overlap_ratio=float(metadata.get("min_window_ictal_overlap_ratio", 0.5)),
    )
    scaled = aplicar_scaler(x_data, resources.scaler)
    scaled = restore_missing_channel_zeros(
        x_data, scaled, n_channels=int(metadata["n_channels"]), features_per_channel=int(metadata["features_per_channel"]),
    )
    raw_scores = np.asarray(resources.model.predict(scaled, batch_size=128, verbose=0))
    yy, scores, metas = agregar_predicoes_temporais(raw_scores, y_data, temporal)

    grid = []
    for threshold in THRESHOLDS:
        for duration in DURATIONS:
            result = avaliar_combo(
                yy, scores, metas, threshold=threshold, min_duration_seconds=duration,
                max_gap_seconds=MAX_GAP_SECONDS,
                max_suspicious_coverage=float(settings.ai_sequence_max_suspicious_coverage),
                min_overlap_seconds=1.0, min_overlap_ratio=0.25,
            )
            grid.append(compact(result))
    candidate = choose_candidate(grid)
    payload = {
        "experiment": "v43_duration_grid_calibration_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_scope": "calibration_patients_only",
        "parameters_were_tuned_on_reserved_or_posthoc_edfs": False,
        "inference_reused_for_all_combinations": True,
        "model": {"path": str(Path(resources.model_path).resolve()), "sha256": sha256(Path(resources.model_path))},
        "scaler": {"path": str(scaler_path.resolve()), "sha256": sha256(scaler_path)},
        "original_calibration": str(calibration_path.resolve()) if calibration_path else None,
        "calibration_patients": ["chb16", "chb19", "chb20", "chb24"],
        "calibration_files": CALIBRATION_FILES,
        "fixed_max_gap_seconds": MAX_GAP_SECONDS,
        "false_alarms_per_hour_cap": FA_CAP,
        "grid": grid,
        "best_feasible_candidate": candidate,
        "active_artifact_modified": False,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "REPORT.md").write_text(make_report(payload), encoding="utf-8")
    plot_grid(grid, output_dir / "duration_threshold_grid.png")
    print(json.dumps({"output": str(output_dir.resolve()), "best_feasible_candidate": candidate, "grid": grid}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
