"""Conversão de paths internos em disco para URLs públicas da API."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings


def classificar_score_clinico(score: float) -> str:
    """Classificação binária do laudo conforme limiar clínico de 0.5."""
    if score > 0.5:
        return "Crise Epiléptica"
    return "Atividade Normal"


def mapa_shap_path_para_url(mapa_shap_path: str, settings: Settings) -> str:
    """
    Converte path interno (ex: storage/shap/exame_1_shap.png) em URL pública.

    A pasta `storage` é exposta via StaticFiles em `/static`.
    """
    caminho = Path(mapa_shap_path).resolve()
    raiz = settings.storage_root.resolve()

    try:
        relativo = caminho.relative_to(raiz)
    except ValueError:
        relativo = Path("shap") / caminho.name

    return f"/static/{relativo.as_posix()}"
