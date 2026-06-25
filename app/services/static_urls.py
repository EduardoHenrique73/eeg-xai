"""Conversao de paths internos em disco para URLs publicas da API."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings


def classificar_score_clinico(score: float, threshold: float = 0.5) -> str:
    """Classificacao binaria do laudo conforme limiar clinico configurado."""
    if score > threshold:
        return "Crise Epiléptica"
    return "Atividade Normal"


def mapa_shap_path_para_url(mapa_shap_path: str, settings: Settings) -> str:
    """
    Converte path interno (ex: storage/shap/exame_1_shap.png) em URL publica.

    A pasta `storage` e exposta via StaticFiles em `/static`.
    """
    caminho = Path(mapa_shap_path).resolve()
    raiz = settings.storage_root.resolve()

    try:
        relativo = caminho.relative_to(raiz)
    except ValueError:
        relativo = Path("shap") / caminho.name

    return f"/static/{relativo.as_posix()}"
