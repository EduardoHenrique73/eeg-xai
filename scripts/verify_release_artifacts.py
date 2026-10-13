"""Gera ou verifica o manifesto SHA-256 dos artefatos da v43."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    ROOT / "modelos/v43_final_seed42.keras",
    ROOT / "modelos/v43_final_seed42_scaler.pkl",
    ROOT / "modelos/v43_final_seed42_metadata.json",
    ROOT / "modelos/v43_final_seed42_calibration.json",
)
DEFAULT_MANIFEST = ROOT / "modelos/v43_final_seed42_artifacts_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for bloco in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def snapshot() -> dict[str, Any]:
    ausentes = [str(path) for path in ARTIFACTS if not path.exists()]
    if ausentes:
        raise FileNotFoundError("Artefatos ausentes: " + ", ".join(ausentes))
    metadata = json.loads(ARTIFACTS[2].read_text(encoding="utf-8"))
    calibration = json.loads(ARTIFACTS[3].read_text(encoding="utf-8"))
    point = calibration.get("best_aligned") or calibration.get("best") or {}
    return {
        "model": "v43_final_seed42",
        "architecture": metadata.get("architecture"),
        "seed": metadata.get("seed"),
        "checkpoint_best_epoch": metadata.get("checkpoint_best_epoch"),
        "operating_point": {
            "threshold": point.get("threshold"),
            "min_duration_seconds": point.get("min_duration_seconds"),
            "max_gap_seconds": point.get("max_gap_seconds"),
        },
        "files": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in ARTIFACTS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    atual = snapshot()
    if args.write:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(atual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Manifesto criado: {args.manifest.resolve()}")
        return
    if not args.manifest.exists():
        raise SystemExit("Manifesto ausente; execute novamente com --write.")
    esperado = json.loads(args.manifest.read_text(encoding="utf-8"))
    if atual != esperado:
        print(json.dumps({"expected": esperado, "actual": atual}, indent=2))
        raise SystemExit("Falha: artefatos diferentes do manifesto.")
    print("OK: artefatos v43 inteiros e coerentes com o manifesto.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
