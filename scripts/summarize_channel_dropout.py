"""Apply the preregistered v58 gate without reading historical patients."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_patient_balancing import normalize_recorded_baseline, point_at_cap


def summarize(baseline, candidate):
    baseline = normalize_recorded_baseline(baseline)
    controls = (
        "train_files", "calibration_files", "reserved_test_files_not_loaded",
        "train_sequences", "train_positive_sequences", "calibration_sequences",
        "architecture", "event_balanced_sampling", "checkpoint_monitor",
        "checkpoint_false_alarms_per_hour_cap", "checkpoint_patience",
        "checkpoint_min_epochs", "checkpoint_rank_mode", "learning_rate",
        "epochs_limit", "short_event_weight", "reduce_lr_on_plateau",
    )
    for key in controls:
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Uncontrolled difference: {key}")
    for data in (baseline, candidate):
        if data.get("test_accessed") is not False:
            raise ValueError("Calibration-only reports required")
        if (data.get("hard_negative_mining") or {}).get("enabled", False):
            raise ValueError("Hard negatives must be disabled")
        if data["patient_weight_audit"]["exponent"] != 0.0:
            raise ValueError("Patient balance must be disabled")
    if baseline.get("channel_dropout", 0.0) != 0.0 or candidate["channel_dropout"] != 0.1:
        raise ValueError("Preregistered channel dropout is exactly 0 versus 0.1")
    baseline_runs = {run["seed"]: run for run in baseline["runs"]}
    rows = []
    for run in candidate["runs"]:
        control = baseline_runs[run["seed"]]
        b, c = point_at_cap(control, 0.75), point_at_cap(run, 0.75)
        gate = (
            c["detected_events"] > b["detected_events"]
            and c["localized_f1"] >= b["localized_f1"] - 1e-9
            and c["edf_f1"] >= b["edf_f1"] - 1e-9
            and c["false_alarms_per_hour"] <= 0.75
        )
        b_events = {(e["file"], e["event"]): e for e in control["event_diagnostics_by_fa_cap"]["0.75"]}
        c_events = {(e["file"], e["event"]): e for e in run["event_diagnostics_by_fa_cap"]["0.75"]}
        if b_events.keys() != c_events.keys():
            raise ValueError("Event sets differ")
        rows.append({
            "seed": run["seed"], "baseline": b, "candidate": c,
            "best_epoch": run["checkpoint_best_epoch"], "continue_gate": gate,
            "events": [dict(file=k[0], event=k[1], baseline=b_events[k], candidate=c_events[k]) for k in b_events],
            "froc_baseline": control["calibration_froc"],
            "froc_candidate": run["calibration_froc"],
        })
    return dict(experiment="v58_channel_dropout", historical_accessed=False,
                promotion=False, rows=rows)


def write_visual_report(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = ["# v58: resultado controlado de dropout por canal", "",
             "Comparacao somente na calibracao. Teto principal: 0,75 FA/h.", "",
             "| Seed | Modelo | Eventos | F1 localizado | F1 EDF | FA/h |",
             "|---|---|---:|---:|---:|---:|"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in result["rows"]:
        for arm, label, style in (("baseline", "v43", "o-"), ("candidate", "v58", "s--")):
            point = row[arm]
            lines.append(f"| {row['seed']} | {label} | {point['detected_events']}/{point['total_events']} | "
                         f"{100*point['localized_f1']:.1f}% | {100*point['edf_f1']:.1f}% | "
                         f"{point['false_alarms_per_hour']:.3f} |")
            points = sorted([p for p in row[f"froc_{arm}"] if p.get("constraint_satisfied", True)],
                            key=lambda p: p["false_alarms_per_hour"])
            ax.plot([p["false_alarms_per_hour"] for p in points],
                    [100*p["detected_events"]/p["total_events"] for p in points],
                    style, label=f"{label} seed {row['seed']}")
        lines.extend(["", f"Checkpoint v58: epoca {row['best_epoch']}.",
                      f"Criterio para continuar: {'ATENDIDO' if row['continue_gate'] else 'NAO ATENDIDO'}.",
                      "", "| Arquivo | Evento | Detectado v43/v58 | Pico v43 | Pico v58 |",
                      "|---|---:|---|---:|---:|"])
        for item in row["events"]:
            b, c = item["baseline"], item["candidate"]
            bs, cs = b["max_ictal_score"], c["max_ictal_score"]
            lines.append(f"| {item['file']} | {item['event']} | {b['detected']}/{c['detected']} | "
                         f"{bs if bs is None else round(bs, 4)} | {cs if cs is None else round(cs, 4)} |")
    ax.axvline(0.75, color="gray", linestyle=":", label="Teto principal")
    ax.set(xlabel="Falsos alarmes por hora", ylabel="Sensibilidade por evento (%)",
           title="FROC discreta na calibracao", ylim=(0, 105))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=170)
    plt.close(fig)
    lines.extend(["", "FROC mostra pontos selecionados da grade de calibracao; nao e validacao externa.",
                  "Nenhum modelo foi promovido automaticamente."])
    output.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_visual_report(result, args.output)
    for row in result["rows"]:
        print(json.dumps({k: v for k, v in row.items() if k not in {"events", "froc_baseline", "froc_candidate"}}, indent=2))
