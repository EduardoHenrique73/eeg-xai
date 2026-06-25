"""Rotas de ingestão e consulta de exames EEG."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.deps import get_current_user
from app.database import get_db
from app.models import Exame, Paciente, PredicaoIA, Usuario
from app.models.exame import STATUS_EXAME_CONCLUIDO
from app.schemas.exame import (
    AnaliseIARequest,
    AnaliseIAResponse,
    DiagnosticoConcluido,
    DiagnosticoEmProcessamento,
    ExameUploadResponse,
    LaudoExameResponse,
    LaudoUpdate,
    SinaisExameResponse,
)
from app.services.exame_pipeline import processar_exame_ia
from app.services.exame_sinais import extrair_sinais_para_visualizacao
from app.services.exame_storage import salvar_arquivo_edf, validar_extensao_edf
from app.services.static_urls import classificar_score_clinico, mapa_shap_path_para_url
from app.ai_engine.feature_extractor import extrair_metadados_edf, listar_canais_eeg_edf

router = APIRouter(
    prefix="/api/exames",
    tags=["Exames"],
    dependencies=[Depends(get_current_user)],
)


@router.get(
    "/{exame_id}/diagnostico",
    response_model=DiagnosticoConcluido,
    responses={
        status.HTTP_206_PARTIAL_CONTENT: {
            "model": DiagnosticoEmProcessamento,
            "description": "Exame cadastrado; IA ainda processando em background.",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Exame não encontrado."},
    },
    summary="Laudo clínico com score da IA e mapa SHAP",
)
async def obter_diagnostico_exame(
    exame_id: int,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    usuario: Usuario = Depends(get_current_user),
) -> DiagnosticoConcluido | JSONResponse:
    """
    Retorna metadados do exame e laudo da CNN-LSTM + explicabilidade (XAI).

    - **404**: exame inexistente.
    - **206**: exame existe, mas `PredicaoIA` ainda não foi persistida.
    - **200**: laudo concluído com score, classificação e URL pública do mapa SHAP.
    """
    exame = await db.get(Exame, exame_id)
    if exame is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exame com id {exame_id} não encontrado.",
        )

    metadados = {
        "exame_id": exame.id,
        "id_paciente": exame.id_paciente,
        "taxa_amostragem": exame.taxa_amostragem,
        "data_upload": exame.data_upload,
        "status_exame": exame.status_exame,
        "laudo_texto": exame.laudo_texto,
    }

    result = await db.execute(
        select(PredicaoIA)
        .where(PredicaoIA.id_exame == exame_id)
        .order_by(PredicaoIA.data_analise.desc())
        .limit(1)
    )
    predicao = result.scalar_one_or_none()

    if predicao is None:
        payload = DiagnosticoEmProcessamento(**metadados)
        return JSONResponse(
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            content=payload.model_dump(mode="json"),
        )

    mapa_shap_url = (
        mapa_shap_path_para_url(predicao.mapa_shap_path, settings)
        if usuario.exibir_shap
        else None
    )

    return DiagnosticoConcluido(
        **metadados,
        resultado_score=predicao.resultado_score,
        classificacao_clinica=classificar_score_clinico(
            predicao.resultado_score,
            usuario.threshold_confianca,
        ),
        threshold_confianca=usuario.threshold_confianca,
        mapa_shap_url=mapa_shap_url,
        data_analise=predicao.data_analise,
    )


@router.patch(
    "/{exame_id}/laudo",
    response_model=LaudoExameResponse,
    summary="Emitir laudo médico final do exame",
)
async def salvar_laudo_exame(
    exame_id: int,
    payload: LaudoUpdate,
    db: AsyncSession = Depends(get_db),
) -> LaudoExameResponse:
    """
    Persiste o parecer médico e marca o exame como concluído.

    Após emitido, o laudo torna-se imutável (HTTP 409 em novas tentativas).
    """
    exame = await db.get(Exame, exame_id)
    if exame is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exame com id {exame_id} não encontrado.",
        )

    if exame.status_exame == STATUS_EXAME_CONCLUIDO:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Laudo médico já foi emitido e não pode ser alterado.",
        )

    texto = payload.laudo_texto.strip()
    if not texto:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="O texto do laudo não pode ser vazio.",
        )

    exame.laudo_texto = texto
    exame.status_exame = STATUS_EXAME_CONCLUIDO
    await db.flush()
    await db.refresh(exame)

    return LaudoExameResponse(
        exame_id=exame.id,
        laudo_texto=exame.laudo_texto,
        status_exame=exame.status_exame,
    )


@router.get(
    "/{exame_id}/sinais",
    response_model=SinaisExameResponse,
    summary="Série temporal EEG downsampled para visualização",
)
async def obter_sinais_exame(
    exame_id: int,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> SinaisExameResponse:
    """
    Retorna amplitudes reais do .edf (média dos canais EEG) com downsampling.

    Limita a ~1500 pontos para renderização fluida no Recharts.
    """
    exame = await db.get(Exame, exame_id)
    if exame is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exame com id {exame_id} não encontrado.",
        )

    arquivo = Path(exame.arquivo_path)
    if not arquivo.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Arquivo EDF não encontrado no disco: {exame.arquivo_path}",
        )

    try:
        dados = await asyncio.to_thread(
            extrair_sinais_para_visualizacao,
            arquivo,
            max_duration_seconds=settings.max_edf_duration_seconds,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return SinaisExameResponse(
        exame_id=exame_id,
        pontos=dados["pontos"],  # type: ignore[arg-type]
        taxa_amostragem_hz=float(dados["taxa_amostragem_hz"]),  # type: ignore[arg-type]
        n_canais_eeg=int(dados["n_canais_eeg"]),  # type: ignore[arg-type]
        canais_eeg=list(dados.get("canais_eeg", [])),  # type: ignore[arg-type]
        n_pontos_original=int(dados["n_pontos_original"]),  # type: ignore[arg-type]
        n_pontos_retornados=int(dados["n_pontos_retornados"]),  # type: ignore[arg-type]
    )


@router.post(
    "/{exame_id}/analise",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AnaliseIAResponse,
    summary="Enfileirar análise IA multicanal (CNN-LSTM + SHAP)",
)
async def solicitar_analise_ia(
    exame_id: int,
    payload: AnaliseIARequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> AnaliseIAResponse:
    """
    Dispara o pipeline de IA para o exame, processando os canais EEG selecionados.

    Se `canais_selecionados` for omitido ou vazio, todos os canais EEG do .edf são usados.
    """
    exame = await db.get(Exame, exame_id)
    if exame is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exame com id {exame_id} não encontrado.",
        )

    arquivo = Path(exame.arquivo_path)
    if not arquivo.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Arquivo EDF não encontrado no disco: {exame.arquivo_path}",
        )

    canais = payload.canais_selecionados
    if canais is not None and len(canais) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Selecione ao menos um canal EEG para análise.",
        )

    try:
        canais_disponiveis = await asyncio.to_thread(
            listar_canais_eeg_edf,
            arquivo,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    alvos = canais if canais else canais_disponiveis
    invalidos = [c for c in alvos if c not in canais_disponiveis]
    if invalidos:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Canais EEG inválidos no arquivo: {', '.join(invalidos)}",
        )

    background_tasks.add_task(
        processar_exame_ia,
        exame_id,
        canais_selecionados=canais,
    )

    return AnaliseIAResponse(
        exame_id=exame_id,
        canais_processados=alvos,
        message=(
            f"Análise IA enfileirada para {len(alvos)} canal(is). "
            "Consulte GET /diagnostico para acompanhar."
        ),
    )


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ExameUploadResponse,
    summary="Upload de exame EEG (.edf)",
)
async def upload_exame(
    arquivo: UploadFile,
    paciente_id: int = Form(..., description="ID do paciente vinculado ao exame"),
    taxa_amostragem: float | None = Form(
        default=None,
        gt=0,
        description="Taxa de amostragem em Hz; se omitida, sera extraida do EDF.",
    ),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ExameUploadResponse:
    """
    Recebe um arquivo .edf e persiste no disco.

    A análise IA é disparada separadamente via POST /{exame_id}/analise
    após o médico selecionar os canais EEG desejados.
    """
    if not validar_extensao_edf(arquivo.filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Apenas arquivos com extensão .edf são aceitos.",
        )

    paciente = await db.get(Paciente, paciente_id)
    if paciente is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paciente com id {paciente_id} não encontrado.",
        )

    caminho_absoluto = await salvar_arquivo_edf(arquivo, settings)

    try:
        metadados_edf = await asyncio.to_thread(extrair_metadados_edf, caminho_absoluto)
    except (FileNotFoundError, ValueError) as exc:
        caminho_absoluto.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    taxa_real = float(metadados_edf["taxa_amostragem"])
    canais_eeg = list(metadados_edf["canais_eeg"])

    exame = Exame(
        id_paciente=paciente_id,
        taxa_amostragem=taxa_real if taxa_amostragem is None else taxa_real,
        canais_eeg=json.dumps(canais_eeg),
        arquivo_path=str(caminho_absoluto),
    )
    db.add(exame)
    await db.flush()
    await db.refresh(exame)

    return ExameUploadResponse(
        exame_id=exame.id,
        arquivo_path=exame.arquivo_path,
        taxa_amostragem=exame.taxa_amostragem,
        canais_eeg=canais_eeg,
        status_exame=exame.status_exame,
        laudo_texto=exame.laudo_texto,
    )
