"""Export or configure a self-contained Windows copy of EEG-XAI.

The bundle contains clinical records and reports. Keep it on trusted storage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import ntpath
from pathlib import Path
import secrets
import shutil
import sqlite3
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = "transfer_data"
MODEL_FILES = ("v43_final_seed42.keras", "v43_final_seed42_scaler.pkl")


def _copy_files(source: Path, destination: Path, *, exclude_partial: bool = False) -> int:
    count = 0
    if not source.exists():
        return count
    for path in source.rglob("*"):
        if not path.is_file() or (exclude_partial and path.name.endswith(".part")):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as original, sqlite3.connect(destination) as copy:
        original.backup(copy)


def export_bundle(destination: Path, *, dry_run: bool = False) -> None:
    from sqlalchemy.engine import make_url

    from app.config import get_settings

    destination = destination.resolve()
    root = ROOT.resolve()
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError("Escolha uma pasta de transferencia fora do projeto.")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("A pasta de destino deve estar vazia.")

    settings = get_settings()
    url = make_url(settings.database_url)
    if not url.drivername.startswith("sqlite") or not url.database:
        raise ValueError("Esta exportacao automatica suporta apenas o banco SQLite local.")
    database = Path(url.database).resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Banco nao encontrado: {database}")
    model_paths = [settings.ai_sequence_model_path]
    model_paths.extend(
        [
            settings.ai_sequence_scaler_path or settings.ai_sequence_model_path.with_name(settings.ai_sequence_model_path.stem + "_scaler.pkl"),
        ]
    )
    for path in model_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Artefato do modelo ausente: {path}")

    sources = [ROOT / "dataset_amostra", settings.edf_storage_path, settings.shap_storage_path]
    local_files = [
        path for source in sources if source.exists()
        for path in source.rglob("*") if path.is_file() and not path.name.endswith(".part")
    ]
    total_bytes = sum(path.stat().st_size for path in local_files + model_paths + [database])
    print(f"Dados locais: {len(local_files)} arquivos, {total_bytes / (1024 ** 3):.2f} GiB (mais o codigo).")
    if dry_run:
        print(f"Previa: o pacote seria criado em {destination}")
        return

    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    for relative in filter(None, tracked):
        source = ROOT / relative
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    # Include the transfer workflow even when exported before its first commit.
    for relative in (
        "scripts/portable_installation.py",
        "scripts/setup_new_pc.ps1",
        "docs/TRANSFERIR_PARA_OUTRO_PC.md",
    ):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    data = destination / DATA_DIR
    _sqlite_backup(database, data / "eeg_xai_dev.db")
    counts = {
        "dataset": _copy_files(ROOT / "dataset_amostra", destination / "dataset_amostra", exclude_partial=True),
        "edf": _copy_files(settings.edf_storage_path, destination / "storage" / "edf", exclude_partial=True),
        "shap": _copy_files(settings.shap_storage_path, destination / "storage" / "shap"),
    }
    for source, filename in zip(model_paths, MODEL_FILES):
        target = destination / "modelos" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "format": 1,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_storage_root": str(settings.storage_root.resolve()),
        "source_database": str(database),
        "counts": counts,
    }
    data.mkdir(parents=True, exist_ok=True)
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Pacote pronto: {destination}")
    print(f"Arquivos locais: {counts}. Cache de features nao incluido.")
    print("Contem dados de pacientes e laudos. Nao publique nem envie ao Git.")


def _remap_path(value: str, old_root: str, new_root: Path, folder: str) -> str:
    if not value:
        return value
    old_folder = ntpath.normcase(ntpath.normpath(ntpath.join(old_root, folder)))
    old_value = ntpath.normcase(ntpath.normpath(value))
    try:
        relative = ntpath.relpath(old_value, old_folder)
    except ValueError as exc:
        raise ValueError(f"Caminho fora de storage/{folder}: {value}") from exc
    if relative == ".." or relative.startswith("..\\") or ntpath.isabs(relative):
        raise ValueError(f"Caminho fora de storage/{folder}: {value}")
    replacement = new_root / folder / Path(relative)
    if not replacement.is_file():
        raise FileNotFoundError(f"Arquivo ausente apos transferencia: {replacement}")
    return str(replacement.resolve())


def _remap_database(database: Path, old_root: str, new_root: Path) -> tuple[int, int]:
    with sqlite3.connect(database) as conn:
        exams = conn.execute("SELECT id, arquivo_path FROM exames").fetchall()
        predictions = conn.execute("SELECT id, mapa_shap_path FROM predicoes_ia").fetchall()
        mapped_exams = [
            (_remap_path(path, old_root, new_root, "edf"), key)
            for key, path in exams
        ]
        mapped_predictions = [
            (_remap_path(path, old_root, new_root, "shap"), key)
            for key, path in predictions
        ]
        conn.executemany("UPDATE exames SET arquivo_path = ? WHERE id = ?", mapped_exams)
        conn.executemany("UPDATE predicoes_ia SET mapa_shap_path = ? WHERE id = ?", mapped_predictions)
    return len(exams), len(predictions)


def configure_bundle() -> None:
    data = ROOT / DATA_DIR
    manifest_path = data / "manifest.json"
    snapshot = data / "eeg_xai_dev.db"
    database = ROOT / "eeg_xai_dev.db"
    environment = ROOT / ".env"
    if not manifest_path.is_file() or not snapshot.is_file():
        raise FileNotFoundError("Pacote incompleto: transfer_data/manifest.json ou banco ausente.")
    if database.exists() or environment.exists():
        raise FileExistsError("Banco ou .env ja existem. Nao sobrescrevo uma instalacao configurada.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != 1:
        raise ValueError("Formato de pacote desconhecido.")
    if not all((ROOT / "modelos" / name).is_file() for name in MODEL_FILES):
        raise FileNotFoundError("Modelo ou scaler nao encontrados no pacote.")

    shutil.copy2(snapshot, database)
    try:
        exams, predictions = _remap_database(
            database,
            manifest["source_storage_root"],
            ROOT / "storage",
        )
    except Exception:
        database.unlink()
        raise
    root = ROOT.as_posix()
    jwt_secret = secrets.token_urlsafe(48)
    lines = [
        "APP_NAME=EEG-XAI",
        "APP_ENV=development",
        "DEBUG=false",
        "DATABASE_URL=sqlite+aiosqlite:///./eeg_xai_dev.db",
        f"STORAGE_ROOT={root}/storage",
        f"EDF_STORAGE_PATH={root}/storage/edf",
        f"SHAP_STORAGE_PATH={root}/storage/shap",
        "AI_MODEL_TYPE=sequence_cnn_lstm",
        f"AI_SEQUENCE_MODEL_PATH={root}/modelos/{MODEL_FILES[0]}",
        f"AI_SEQUENCE_SCALER_PATH={root}/modelos/{MODEL_FILES[1]}",
        f"AI_SEQUENCE_METADATA_PATH={root}/modelos/v43_final_seed42_metadata.json",
        f"AI_SEQUENCE_CALIBRATION_PATH={root}/modelos/v43_final_seed42_calibration.json",
        f"JWT_SECRET_KEY={jwt_secret}",
    ]
    environment.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Instalacao configurada: {ROOT}")
    print(f"Caminhos atualizados no banco: {exams} exames, {predictions} predicoes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Cria uma pasta completa para transferencia.")
    export.add_argument("--destination", type=Path, required=True)
    export.add_argument("--dry-run", action="store_true", help="Mostra o tamanho sem copiar arquivos.")
    commands.add_parser("configure", help="Ajusta banco e .env no PC novo.")
    args = parser.parse_args()
    if args.command == "export":
        export_bundle(args.destination, dry_run=args.dry_run)
    else:
        configure_bundle()


if __name__ == "__main__":
    main()
