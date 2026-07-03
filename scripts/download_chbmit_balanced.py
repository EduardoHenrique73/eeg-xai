"""Baixa EDFs do CHB-MIT em lotes pequenos e retomaveis.

O objetivo e alimentar a base local sem tentar baixar o dataset inteiro de uma
vez. O script usa os arquivos `chbXX-summary.txt` ja presentes em
`dataset_amostra` para escolher arquivos faltantes, alternando crises e normais
em pacientes com menor cobertura local.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai_engine.training import carregar_resumos_chbmit  # noqa: E402

PHYSIONET_CHBMIT_BASE = "https://physionet.org/files/chbmit/1.0.0"


@dataclass(frozen=True)
class Candidate:
    patient: str
    filename: str
    has_seizure: bool
    local_count: int


def patient_from_file(filename: str) -> str:
    match = re.match(r"(chb\d{2})", filename)
    if not match:
        raise ValueError(f"Nome de arquivo CHB-MIT invalido: {filename}")
    return match.group(1)


def remote_url(filename: str) -> str:
    patient = patient_from_file(filename)
    return f"{PHYSIONET_CHBMIT_BASE}/{patient}/{quote(filename)}"


def local_counts(dataset_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in dataset_dir.glob("*.edf"):
        patient = patient_from_file(path.name)
        counts[patient] = counts.get(patient, 0) + 1
    return counts


def build_candidates(
    dataset_dir: Path,
    *,
    patients: set[str] | None,
) -> list[Candidate]:
    intervals_by_file = carregar_resumos_chbmit(dataset_dir)
    present = {path.name for path in dataset_dir.glob("*.edf")}
    counts = local_counts(dataset_dir)
    candidates: list[Candidate] = []

    for filename, intervals in intervals_by_file.items():
        if not filename.endswith(".edf") or filename in present:
            continue
        patient = patient_from_file(filename)
        if patients and patient not in patients:
            continue
        candidates.append(
            Candidate(
                patient=patient,
                filename=filename,
                has_seizure=bool(intervals),
                local_count=counts.get(patient, 0),
            )
        )

    return sorted(candidates, key=lambda item: (item.local_count, item.patient, not item.has_seizure, item.filename))


def pick_balanced(
    candidates: list[Candidate],
    *,
    max_files: int,
    crisis_files: int | None,
    normal_files: int | None,
) -> list[Candidate]:
    selected: list[Candidate] = []
    used: set[str] = set()
    patients = sorted({item.patient for item in candidates}, key=lambda p: min(c.local_count for c in candidates if c.patient == p))
    crisis_limit = crisis_files if crisis_files is not None else max(1, max_files // 2)
    normal_limit = normal_files if normal_files is not None else max_files - crisis_limit

    # Primeira passada: crises para melhorar localizacao temporal.
    for patient in patients:
        if sum(1 for item in selected if item.has_seizure) >= crisis_limit:
            break
        match = next((item for item in candidates if item.patient == patient and item.has_seizure), None)
        if match and match.filename not in used:
            selected.append(match)
            used.add(match.filename)

    # Segunda passada: normais para reduzir falso positivo.
    for patient in patients:
        if sum(1 for item in selected if not item.has_seizure) >= normal_limit:
            break
        match = next((item for item in candidates if item.patient == patient and not item.has_seizure), None)
        if match and match.filename not in used:
            selected.append(match)
            used.add(match.filename)

    # Completa com o restante em ordem de menor cobertura local.
    if len(selected) >= max_files:
        return selected[:max_files]
    for item in candidates:
        if item.filename not in used:
            selected.append(item)
            used.add(item.filename)
        if len(selected) >= max_files:
            break
    return selected


def content_length(url: str, timeout: int) -> int | None:
    request = Request(url, method="HEAD", headers={"User-Agent": "eeg-xai-downloader/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.headers.get("Content-Length")
            return int(raw) if raw else None
    except (HTTPError, URLError, TimeoutError):
        return None


def download_file(url: str, destination: Path, *, timeout: int, chunk_size: int) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "eeg-xai-downloader/1.0"}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        mode = "ab" if downloaded > 0 and status == 206 else "wb"
        if mode == "wb":
            downloaded = 0
        with partial.open(mode + "") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)

    partial.replace(destination)


def parse_patients(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def describe(items: Iterable[Candidate]) -> list[dict[str, object]]:
    return [
        {
            "patient": item.patient,
            "filename": item.filename,
            "has_seizure": item.has_seizure,
            "local_count_before": item.local_count,
        }
        for item in items
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa um lote balanceado de EDFs CHB-MIT.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--patients", default=None, help="Lista opcional: chb02,chb06,chb17")
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--crisis-files", type=int, default=None)
    parser.add_argument("--normal-files", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates = build_candidates(args.dataset_dir, patients=parse_patients(args.patients))
    selected = pick_balanced(
        candidates,
        max_files=max(1, args.max_files),
        crisis_files=args.crisis_files,
        normal_files=args.normal_files,
    )
    print({"selected": describe(selected)})
    if args.dry_run:
        return

    for index, item in enumerate(selected, start=1):
        url = remote_url(item.filename)
        destination = args.dataset_dir / item.filename
        size = content_length(url, timeout=args.timeout)
        size_mb = f"{size / 1024 / 1024:.1f} MB" if size else "tamanho desconhecido"
        print(f"[{index}/{len(selected)}] {item.filename} ({size_mb})")
        if destination.exists():
            print(f"  ja existe: {destination}")
            continue
        try:
            download_file(url, destination, timeout=args.timeout, chunk_size=args.chunk_size)
            print(f"  salvo em {destination}")
        except Exception as exc:  # noqa: BLE001
            print(f"  falhou: {exc}")


if __name__ == "__main__":
    main()
