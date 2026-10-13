import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import config
from scripts import portable_installation


def test_configure_bundle_preserves_reports_and_remaps_files(tmp_path, monkeypatch):
    monkeypatch.setattr(portable_installation, "ROOT", tmp_path)
    data = tmp_path / "transfer_data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps({"format": 1, "source_storage_root": "C:\\old\\eeg-xai\\storage"}),
        encoding="utf-8",
    )
    (tmp_path / "modelos").mkdir()
    for name in portable_installation.MODEL_FILES:
        (tmp_path / "modelos" / name).write_bytes(b"test")
    for folder, filename in (("edf", "exam.edf"), ("shap", "map.png")):
        target = tmp_path / "storage" / folder
        target.mkdir(parents=True)
        (target / filename).write_bytes(b"test")

    snapshot = data / "eeg_xai_dev.db"
    with sqlite3.connect(snapshot) as conn:
        conn.execute("CREATE TABLE exames (id INTEGER PRIMARY KEY, arquivo_path TEXT, laudo_texto TEXT)")
        conn.execute("CREATE TABLE predicoes_ia (id INTEGER PRIMARY KEY, mapa_shap_path TEXT)")
        conn.execute(
            "INSERT INTO exames VALUES (1, ?, ?)",
            (r"C:\old\eeg-xai\storage\edf\exam.edf", "Laudo original"),
        )
        conn.execute(
            "INSERT INTO predicoes_ia VALUES (1, ?)",
            (r"C:\old\eeg-xai\storage\shap\map.png",),
        )

    portable_installation.configure_bundle()

    with sqlite3.connect(tmp_path / "eeg_xai_dev.db") as conn:
        exam_path, report = conn.execute("SELECT arquivo_path, laudo_texto FROM exames").fetchone()
        shap_path = conn.execute("SELECT mapa_shap_path FROM predicoes_ia").fetchone()[0]
    assert report == "Laudo original"
    assert Path(exam_path) == (tmp_path / "storage" / "edf" / "exam.edf").resolve()
    assert Path(shap_path) == (tmp_path / "storage" / "shap" / "map.png").resolve()
    assert "AI_MODEL_TYPE=sequence_cnn_lstm" in (tmp_path / ".env").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        portable_installation.configure_bundle()


def test_remap_rejects_file_outside_storage(tmp_path):
    with pytest.raises(ValueError, match="fora de storage"):
        portable_installation._remap_path(
            r"C:\old\other\exam.edf",
            r"C:\old\eeg-xai\storage",
            tmp_path,
            "edf",
        )


def test_export_bundle_copies_code_database_and_local_data(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "package"
    (source / "scripts").mkdir(parents=True)
    (source / "docs").mkdir()
    (source / "modelos").mkdir()
    (source / "dataset_amostra").mkdir()
    (source / "storage" / "edf").mkdir(parents=True)
    (source / "storage" / "shap").mkdir()
    (source / "README.md").write_text("EEG-XAI", encoding="utf-8")
    for relative in (
        "scripts/portable_installation.py",
        "scripts/setup_new_pc.ps1",
        "docs/TRANSFERIR_PARA_OUTRO_PC.md",
    ):
        (source / relative).write_text("test", encoding="utf-8")
    for name in portable_installation.MODEL_FILES:
        (source / "modelos" / name).write_bytes(b"model")
    (source / "dataset_amostra" / "sample.edf").write_bytes(b"edf")
    (source / "storage" / "edf" / "exam.edf").write_bytes(b"exam")
    (source / "storage" / "shap" / "map.png").write_bytes(b"png")
    with sqlite3.connect(source / "eeg_xai_dev.db") as conn:
        conn.execute("CREATE TABLE exames (id INTEGER PRIMARY KEY)")

    monkeypatch.setattr(portable_installation, "ROOT", source)
    monkeypatch.setattr(
        config,
        "get_settings",
        lambda: SimpleNamespace(
            database_url=f"sqlite+aiosqlite:///{(source / 'eeg_xai_dev.db').as_posix()}",
            storage_root=source / "storage",
            edf_storage_path=source / "storage" / "edf",
            shap_storage_path=source / "storage" / "shap",
            ai_sequence_model_path=source / "modelos" / portable_installation.MODEL_FILES[0],
            ai_sequence_scaler_path=source / "modelos" / portable_installation.MODEL_FILES[1],
        ),
    )
    monkeypatch.setattr(
        portable_installation.subprocess,
        "check_output",
        lambda *args, **kwargs: b"README.md\0",
    )

    portable_installation.export_bundle(destination, dry_run=True)
    assert not destination.exists()
    portable_installation.export_bundle(destination)
    assert (destination / "README.md").read_text(encoding="utf-8") == "EEG-XAI"
    assert (destination / "dataset_amostra" / "sample.edf").read_bytes() == b"edf"
    assert (destination / "storage" / "edf" / "exam.edf").read_bytes() == b"exam"
    assert (destination / "storage" / "shap" / "map.png").read_bytes() == b"png"
    assert (destination / "transfer_data" / "eeg_xai_dev.db").is_file()
    assert (destination / "scripts" / "setup_new_pc.ps1").is_file()
