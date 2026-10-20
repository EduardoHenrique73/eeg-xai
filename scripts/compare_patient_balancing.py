"""Compare paired training reports at the same calibration FA/h constraint."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def normalize_recorded_baseline(payload: dict[str, Any]) -> dict[str, Any]:
    """Make the documented v43 training defaults explicit in its older report."""
    result = deepcopy(payload)
    if result.get("experiment") == "v43_late_channel_fusion":
        defaults = {
            "checkpoint_rank_mode": "event_localization",
            "learning_rate": 0.001,
            "reduce_lr_on_plateau": False,
            "patient_weight_audit": {"exponent": 0.0},
        }
        applied = {}
        for key, value in defaults.items():
            if key not in result:
                result[key] = value
                applied[key] = value
        result["recorded_v43_defaults_applied"] = applied
    # Parameters of a disabled miner cannot change training.
    if not (result.get("hard_negative_mining") or {}).get("enabled", False):
        result["hard_negative_mining"] = {"enabled": False}
    return result


def point_at_cap(run: dict[str, Any], cap: float) -> dict[str, Any]:
    for point in run["calibration_froc"]:
        if abs(float(point["cap"]) - cap) < 1e-9:
            if not point.get("constraint_satisfied", True):
                raise ValueError(f"No feasible operating point at FA/h <= {cap}.")
            return point
    raise ValueError(f"Missing calibration operating point at FA/h <= {cap}.")


def read_runs(paths: list[Path]) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
    result = {}
    for path in paths:
        payload = normalize_recorded_baseline(json.loads(path.read_text(encoding="utf-8")))
        payload["comparison_report_path"] = str(path.resolve())
        if payload.get("test_accessed") is not False:
            raise ValueError(f"Expected calibration-only report: {path}")
        for run in payload["runs"]:
            seed = int(run["seed"])
            if seed in result:
                raise ValueError(f"Duplicate seed {seed}.")
            result[seed] = payload, run
    return result


def compare(baselines: dict, candidates: dict, cap: float) -> dict[str, Any]:
    if not baselines or set(baselines) != set(candidates):
        raise ValueError("Baseline and candidate must contain identical, nonempty seed sets.")
    controls = (
        "train_files", "calibration_files", "reserved_test_files_not_loaded",
        "architecture", "event_balanced_sampling", "checkpoint_monitor",
        "checkpoint_false_alarms_per_hour_cap", "checkpoint_patience",
        "checkpoint_min_epochs", "checkpoint_rank_mode", "learning_rate", "epochs_limit",
        "train_sequences", "calibration_sequences", "short_event_weight",
        "reduce_lr_on_plateau", "hard_negative_mining",
    )
    rows = []
    per_event = []
    for seed in sorted(baselines):
        bp, br = baselines[seed]
        cp, cr = candidates[seed]
        for key in controls:
            if bp.get(key) != cp.get(key):
                raise ValueError(f"Uncontrolled difference at seed {seed}: {key}")
        if bp["patient_weight_audit"]["exponent"] != 0.0:
            raise ValueError("Baseline must disable patient balancing.")
        if cp["patient_weight_audit"]["exponent"] != 0.5:
            raise ValueError("This experiment preregistered exponent 0.5.")
        baseline, candidate = point_at_cap(br, cap), point_at_cap(cr, cap)
        improved = (
            candidate["detected_events"] >= baseline["detected_events"]
            and candidate["localized_f1"] >= baseline["localized_f1"] - 1e-9
            and candidate["edf_f1"] >= baseline["edf_f1"] - 1e-9
            and candidate["false_alarms_per_hour"] <= baseline["false_alarms_per_hour"] + 1e-9
            and any((
                candidate["detected_events"] > baseline["detected_events"],
                candidate["localized_f1"] > baseline["localized_f1"] + 1e-9,
                candidate["edf_f1"] > baseline["edf_f1"] + 1e-9,
                candidate["false_alarms_per_hour"] < baseline["false_alarms_per_hour"] - 1e-9,
            ))
        )
        rows.append({"seed": seed, "baseline": baseline, "candidate": candidate,
                     "candidate_artifacts": cr.get("saved_artifacts", {}),
                     "baseline_epoch": br.get("checkpoint_best_epoch"),
                     "candidate_epoch": cr.get("checkpoint_best_epoch"),
                     "baseline_report": bp.get("comparison_report_path"),
                     "candidate_report": cp.get("comparison_report_path"),
                     "recorded_baseline_defaults_applied": bp.get("recorded_v43_defaults_applied", {}),
                     "pareto_improvement": improved,
                     "eligible_gain_under_cap": (
                         candidate["detected_events"] >= baseline["detected_events"]
                         and candidate["localized_f1"] >= baseline["localized_f1"] - 1e-9
                         and candidate["edf_f1"] >= baseline["edf_f1"] - 1e-9
                         and candidate["false_alarms_per_hour"] <= cap + 1e-9
                         and any((
                             candidate["detected_events"] > baseline["detected_events"],
                             candidate["localized_f1"] > baseline["localized_f1"] + 1e-9,
                             candidate["edf_f1"] > baseline["edf_f1"] + 1e-9,
                             candidate["false_alarms_per_hour"] < baseline["false_alarms_per_hour"] - 1e-9,
                         ))
                     ),
                     "baseline_frontier": br["calibration_froc"],
                     "candidate_frontier": cr["calibration_froc"]})
        b_events = {(e["file"], e["event"]): e for e in br["event_diagnostics_by_fa_cap"][str(cap)]}
        c_events = {(e["file"], e["event"]): e for e in cr["event_diagnostics_by_fa_cap"][str(cap)]}
        if set(b_events) != set(c_events):
            raise ValueError("Calibration event sets differ.")
        for key in sorted(b_events):
            per_event.append({"seed": seed, "file": key[0], "event": key[1],
                              "baseline": b_events[key], "candidate": c_events[key]})
    metrics = ("detected_events", "event_f1", "localized_f1", "edf_f1", "false_alarms_per_hour")
    means = {
        arm: {metric: float(np.mean([row[arm][metric] for row in rows])) for metric in metrics}
        for arm in ("baseline", "candidate")
    }
    wins = sum(row["pareto_improvement"] for row in rows)
    eligible_gains = sum(row["eligible_gain_under_cap"] for row in rows)
    average_preserved = all(
        means["candidate"][key] >= means["baseline"][key] - 1e-9
        for key in ("detected_events", "localized_f1", "edf_f1")
    ) and means["candidate"]["false_alarms_per_hour"] <= cap + 1e-9
    confirmed = len(rows) >= 3 and eligible_gains >= 2 and average_preserved
    promising = confirmed or (len(rows) < 3 and eligible_gains > 0)
    selected = max(rows, key=lambda row: (
        row["candidate"]["detected_events"], row["candidate"]["localized_f1"],
        row["candidate"]["edf_f1"], row["candidate"]["event_precision"],
        -row["candidate"]["false_alarms_per_hour"],
    )) if confirmed else None
    return {
        "experiment": "v57_patient_balance_calibration_only", "fa_cap": cap,
        "test_accessed": False, "paired_seeds": sorted(baselines),
        "pareto_improvements": wins, "eligible_gain_seeds": eligible_gains,
        "calibration_confirmation": confirmed,
        "selected_candidate_seed": selected["seed"] if selected else None,
        "means": means, "runs": rows, "events": per_event,
        "promotion": False,
        "status": "PROMISSOR MAS NAO SUPERIOR" if promising else "REJEITAR",
        "interpretation": "Calibration evidence only; no inter-patient improvement has been established.",
    }


def write_report(result: dict[str, Any], output: Path) -> None:
    lines = ["# v57: comparacao pareada na calibracao", "",
             f"Teto FROC: {result['fa_cap']} FA/h. Pacientes historicos nao carregados.", "",
             "| Seed | Modelo | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["runs"]:
        for arm in ("baseline", "candidate"):
            p = row[arm]
            lines.append(f"| {row['seed']} | {arm} | {p['detected_events']}/{p['total_events']} | "
                         f"{100*p['event_f1']:.1f}% | {100*p['localized_f1']:.1f}% | "
                         f"{100*p['edf_f1']:.1f}% | {p['false_alarms_per_hour']:.3f} |")
    lines.extend(["", "## Pontos escolhidos somente na calibracao", "",
                  "| Seed | Modelo | Epoca | Threshold | Duracao minima (s) | Gap (s) |",
                  "|---|---|---:|---:|---:|---:|"])
    for row in result["runs"]:
        for arm in ("baseline", "candidate"):
            p = row[arm]
            lines.append(f"| {row['seed']} | {arm} | {row[f'{arm}_epoch']} | "
                         f"{p['threshold']:.2f} | {p['min_duration_seconds']:.0f} | {p['max_gap_seconds']:.0f} |")
    lines.extend(["", "## Eventos alterados", "",
                  "| Seed | Arquivo | Evento | Baseline | Candidata | Score max baseline | Score max candidata |",
                  "|---|---|---:|---|---|---:|---:|"])
    for e in result["events"]:
        b, c = e["baseline"], e["candidate"]
        if b["detected"] != c["detected"]:
            lines.append(f"| {e['seed']} | {e['file']} | {e['event']} | {b['detected']} | {c['detected']} | "
                         f"{b['max_ictal_score']:.4f} | {c['max_ictal_score']:.4f} |")
    lines.extend(["", "## Decisao", "", result["status"], "",
                  f"Melhora sem perdas nas quatro metricas principais: {result['pareto_improvements']} "
                  f"de {len(result['runs'])} seeds.", "",
                  f"Confirmacao de ganho na calibracao: {result['calibration_confirmation']}.", "",
                  "O resultado nao autoriza alegar generalizacao melhor. A v43 ativa permanece congelada."])
    output.mkdir(parents=True, exist_ok=True)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if result["calibration_confirmation"]:
        selected = next(row for row in result["runs"] if row["seed"] == result["selected_candidate_seed"])
        artifacts = selected["candidate_artifacts"]
        if artifacts:
            calibration = json.loads(Path(artifacts["calibration_path"]).read_text(encoding="utf-8"))
            operating = calibration.get("best_aligned") or calibration["best"]
            for key in ("threshold", "min_duration_seconds", "max_gap_seconds"):
                if operating[key] != selected["candidate"][key]:
                    raise ValueError("Saved calibration and comparison operating point differ.")
            manifest = {
                "selected_by": "calibration_only_before_historical_evaluation",
                "seed": selected["seed"], "operating_point": operating,
                "files": {key: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                          for key, path in artifacts.items()},
                "historical_evaluation_accessed": False,
            }
            (output / "selected_candidate_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
            )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(result["runs"]), figsize=(6 * len(result["runs"]), 4.5), squeeze=False)
    for axis, row in zip(axes[0], result["runs"]):
        for arm, color, label in (("baseline", "#1668c7", "V43 (controle)"),
                                  ("candidate", "#d46b08", "Balanceamento por paciente")):
            frontier = sorted(
                [p for p in row[f"{arm}_frontier"] if p.get("constraint_satisfied", True)],
                key=lambda p: p["false_alarms_per_hour"],
            )
            axis.plot([p["false_alarms_per_hour"] for p in frontier],
                      [100 * p["event_recall"] for p in frontier], "o-", color=color, label=label)
            selected = row[arm]
            axis.scatter(selected["false_alarms_per_hour"], 100 * selected["event_recall"],
                         marker="*", s=180, color=color, edgecolors="black", zorder=4)
        axis.axvline(result["fa_cap"], linestyle=":", color="gray")
        axis.set(title=f"Seed {row['seed']} - calibração", xlabel="Falsos alarmes por hora não ictal",
                 ylabel="Sensibilidade por evento (%)", ylim=(0, 100))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="lower right")
    figure.suptitle("Fronteira FROC: pontos selecionados por teto de FA/h")
    figure.tight_layout()
    figure.savefig(output / "froc_calibration.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fa-cap", type=float, default=0.75)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    baseline, candidate = read_runs(args.baseline), read_runs(args.candidate)
    if args.seeds:
        if not set(args.seeds).issubset(baseline.keys() & candidate.keys()):
            parser.error("Requested seeds must exist in both arms.")
        baseline = {seed: baseline[seed] for seed in args.seeds}
        candidate = {seed: candidate[seed] for seed in args.seeds}
    result = compare(baseline, candidate, args.fa_cap)
    write_report(result, args.output_dir)
    print(json.dumps({"status": result["status"], "means": result["means"],
                      "paired_seeds": result["paired_seeds"]}, indent=2))


if __name__ == "__main__":
    main()
