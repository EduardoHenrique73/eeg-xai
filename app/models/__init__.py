"""Exportação centralizada dos modelos ORM."""

from app.models.exame import Exame
from app.models.paciente import Paciente
from app.models.predicao_ia import PredicaoIA
from app.models.usuario import Usuario

__all__ = ["Usuario", "Paciente", "Exame", "PredicaoIA"]
