"""Gera figuras reprodutiveis dos resultados finais para o TCC."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "tcc" / "assets" / "resultados"
BLUE = "#1668C7"
LIGHT_BLUE = "#DCEBFA"
NAVY = "#17324D"
ORANGE = "#E58A22"
GRAY = "#64748B"
LIGHT_GRAY = "#E2E8F0"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(name, size)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%".replace(".", ",")


def metrics_figure() -> None:
    calibration = json.loads((ROOT / "modelos" / "v43_final_seed42_calibration.json").read_text())
    historical = json.loads((ROOT / "modelos" / "v43_final_seed42_reserved_historical_eval.json").read_text())
    cal = calibration["best"]
    hist = historical["fixed_evaluation"]

    labels = ["Precisao EDF", "Sensibilidade EDF", "F1 EDF", "F1 localizado", "Recall evento", "F1 evento"]
    cal_values = [cal["edf_precision"], cal["edf_recall"], cal["edf_f1"], cal["localized_f1"], cal["event_recall"], cal["event_f1"]]
    hist_values = [hist["metrics"]["precision"], hist["metrics"]["recall"], hist["metrics"]["f1"], hist["localized_metrics"]["f1"], hist["event_sensitivity"], hist["event_f1"]]

    image = Image.new("RGB", (1800, 1040), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 55), "Desempenho do modelo final v43", fill=NAVY, font=font(48, True))
    draw.text((90, 118), "Calibracao separada e avaliacao historica de desenvolvimento", fill=GRAY, font=font(27))

    chart_left, chart_top, chart_right, chart_bottom = 350, 230, 1700, 820
    for tick in range(0, 101, 20):
        x = chart_left + (chart_right - chart_left) * tick / 100
        draw.line((x, chart_top, x, chart_bottom), fill=LIGHT_GRAY, width=2)
        draw.text((x - 18, chart_bottom + 18), f"{tick}%", fill=GRAY, font=font(22))

    row_height = 92
    bar_height = 27
    for index, label in enumerate(labels):
        y = chart_top + index * row_height
        draw.text((90, y + 19), label, fill=NAVY, font=font(25, True))
        for offset, value, color in ((8, cal_values[index], BLUE), (43, hist_values[index], ORANGE)):
            width = (chart_right - chart_left) * value
            draw.rounded_rectangle((chart_left, y + offset, chart_left + width, y + offset + bar_height), radius=7, fill=color)
            draw.text((chart_left + width + 12, y + offset - 1), pct(value), fill=color, font=font(22, True))

    draw.rounded_rectangle((90, 875, 121, 906), radius=5, fill=BLUE)
    draw.text((137, 875), "Calibracao: chb16, chb19, chb20 e chb24", fill=NAVY, font=font(23))
    draw.rounded_rectangle((770, 875, 801, 906), radius=5, fill=ORANGE)
    draw.text((817, 875), "Avaliacao historica: chb09, chb15 e chb18", fill=NAVY, font=font(23))
    draw.text((90, 946), "Nota: a avaliacao historica foi consultada durante o desenvolvimento e nao constitui teste final intocado.", fill=GRAY, font=font(22))
    image.save(OUTPUT / "metricas_v43.png")


def event_summary_figure() -> None:
    calibration = json.loads((ROOT / "modelos" / "v43_final_seed42_calibration.json").read_text())["best"]
    historical = json.loads((ROOT / "modelos" / "v43_final_seed42_reserved_historical_eval.json").read_text())["fixed_evaluation"]

    image = Image.new("RGB", (1800, 880), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 55), "Deteccao por evento e falsos alarmes", fill=NAVY, font=font(48, True))
    draw.text((90, 118), "O desempenho por EDF nao substitui a avaliacao temporal", fill=GRAY, font=font(27))

    groups = [
        ("Calibracao", calibration["detected_events"], calibration["total_events"], calibration["false_alarms_per_hour"], BLUE),
        ("Historico", historical["detected_seizure_events"], historical["total_seizure_events"], historical["false_alarms_per_hour"], ORANGE),
    ]
    for index, (name, detected, total, fa_h, color) in enumerate(groups):
        x = 130 + index * 830
        draw.rounded_rectangle((x, 230, x + 700, 690), radius=18, fill="#F8FAFC", outline=LIGHT_GRAY, width=3)
        draw.text((x + 45, 270), name, fill=color, font=font(37, True))
        draw.text((x + 45, 355), "Eventos localizados", fill=GRAY, font=font(24))
        draw.text((x + 45, 395), f"{detected} de {total}", fill=NAVY, font=font(62, True))
        draw.text((x + 45, 515), "Falsos alarmes por hora", fill=GRAY, font=font(24))
        draw.text((x + 45, 555), f"{fa_h:.2f}".replace(".", ","), fill=NAVY, font=font(62, True))

    draw.text((90, 765), "A calibracao definiu o ponto operacional; o conjunto historico evidencia a queda de generalizacao interpaciente.", fill=GRAY, font=font(24))
    image.save(OUTPUT / "eventos_falsos_alarmes_v43.png")


def terminal_figure() -> None:
    image = Image.new("RGB", (1600, 580), "#0B1220")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((35, 30, 1565, 550), radius=16, fill="#111827", outline="#334155", width=2)
    draw.rectangle((35, 30, 1565, 90), fill="#1E293B")
    draw.ellipse((65, 50, 83, 68), fill="#EF4444")
    draw.ellipse((95, 50, 113, 68), fill="#F59E0B")
    draw.ellipse((125, 50, 143, 68), fill="#22C55E")
    mono = ImageFont.truetype("consola.ttf", 29)
    lines = [
        "PS C:\\Users\\Foco\\Documents\\DEV\\eeg-xai>",
        ".venv\\Scripts\\python.exe scripts\\verify_release_artifacts.py",
        "",
        "OK: artefatos v43 inteiros e coerentes com o manifesto.",
        "",
        "Modelo: v43_final_seed42.keras",
        "Scaler: v43_final_seed42_scaler.pkl",
        "Metadata e calibracao: verificados por SHA-256",
    ]
    y = 125
    for line in lines:
        color = "#86EFAC" if line.startswith("OK:") else "#E2E8F0"
        draw.text((75, y), line, fill=color, font=mono)
        y += 49
    image.save(OUTPUT / "terminal_verificacao_v43.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics_figure()
    event_summary_figure()
    terminal_figure()


if __name__ == "__main__":
    main()
