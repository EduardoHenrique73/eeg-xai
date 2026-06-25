"""Persistência física de arquivos .edf."""

import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings


def validar_extensao_edf(nome_arquivo: str | None) -> bool:
    if not nome_arquivo:
        return False
    return Path(nome_arquivo).suffix.lower() == ".edf"


async def salvar_arquivo_edf(arquivo: UploadFile, settings: Settings) -> Path:
    """Salva o upload em storage/edf/ com nome UUID único."""
    settings.edf_storage_path.mkdir(parents=True, exist_ok=True)
    nome_unico = f"{uuid.uuid4()}.edf"
    destino = settings.edf_storage_path / nome_unico

    conteudo = await arquivo.read()
    destino.write_bytes(conteudo)

    return destino.resolve()
