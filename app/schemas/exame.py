"""Schemas Pydantic — exames clínicos."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ExameUploadResponse(BaseModel):
    message: str = Field(
        default="Exame recebido com sucesso. Selecione os canais e solicite a análise IA.",
    )
    exame_id: int
    arquivo_path: str
    status_exame: str = "pendente"
    laudo_texto: str | None = None


class LaudoUpdate(BaseModel):
    laudo_texto: str = Field(..., min_length=1)


class LaudoExameResponse(BaseModel):
    exame_id: int
    laudo_texto: str
    status_exame: str
    message: str = "Laudo médico emitido e salvo com sucesso."


class DiagnosticoExameBase(BaseModel):
    """Metadados do exame presentes em qualquer estado do laudo."""

    exame_id: int
    id_paciente: int
    taxa_amostragem: float
    data_upload: datetime
    status_exame: str
    laudo_texto: str | None = None


class DiagnosticoEmProcessamento(DiagnosticoExameBase):
    status: Literal["em_processamento"] = "em_processamento"
    message: str = "Análise IA em andamento. Tente novamente em instantes."


class DiagnosticoConcluido(DiagnosticoExameBase):
    status: Literal["concluido"] = "concluido"
    resultado_score: float = Field(ge=0.0, le=1.0)
    classificacao_clinica: str
    mapa_shap_url: str
    data_analise: datetime


class EegPontoVisualizacao(BaseModel):
    tempo: float
    amplitude: float


class SinaisExameResponse(BaseModel):
    exame_id: int
    pontos: list[EegPontoVisualizacao]
    taxa_amostragem_hz: float
    n_canais_eeg: int
    canais_eeg: list[str] = Field(default_factory=list)
    n_pontos_original: int
    n_pontos_retornados: int


class AnaliseIARequest(BaseModel):
    """Parâmetros para enfileirar inferência CNN-LSTM multicanal."""

    canais_selecionados: list[str] | None = Field(
        default=None,
        description="Nomes dos canais EEG a processar; omitir ou null = todos os canais EEG.",
    )


class AnaliseIAResponse(BaseModel):
    exame_id: int
    status: Literal["em_processamento"] = "em_processamento"
    canais_processados: list[str]
    message: str = "Análise IA enfileirada. Consulte GET /diagnostico para acompanhar."
