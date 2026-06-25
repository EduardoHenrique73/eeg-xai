"""Testes do modelo PredicaoIA (laudo da máquina + XAI)."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.models import PredicaoIA


@pytest.mark.asyncio
async def test_criar_predicao_com_score_e_mapa_shap(db_session, exame):
  caminho_shap = "C:/Users/Foco/Documents/DEV/eeg-xai/storage/shap/exame_001_shap.png"
  predicao = PredicaoIA(
      id_exame=exame.id,
      resultado_score=0.73,
      mapa_shap_path=caminho_shap,
  )
  db_session.add(predicao)
  await db_session.commit()
  await db_session.refresh(predicao)

  assert predicao.id is not None
  assert predicao.id_exame == exame.id
  assert predicao.resultado_score == pytest.approx(0.73)
  assert predicao.mapa_shap_path == caminho_shap
  assert predicao.data_analise is not None


@pytest.mark.asyncio
async def test_predicao_nao_armazena_imagem_no_banco(db_session, exame):
  colunas = {c.name for c in PredicaoIA.__table__.columns}
  assert "mapa_shap_path" in colunas
  assert "mapa_shap_blob" not in colunas
  assert "imagem" not in colunas


@pytest.mark.asyncio
async def test_resultado_score_deve_estar_entre_zero_e_um(db_session, exame):
  invalida = PredicaoIA(
      id_exame=exame.id,
      resultado_score=1.5,
      mapa_shap_path="/tmp/shap.png",
  )
  db_session.add(invalida)

  with pytest.raises(IntegrityError):
      await db_session.commit()


@pytest.mark.asyncio
async def test_predicao_vinculada_ao_exame(db_session, exame):
  predicao = PredicaoIA(
      id_exame=exame.id,
      resultado_score=0.42,
      mapa_shap_path="/storage/shap/map.png",
  )
  db_session.add(predicao)
  await db_session.commit()

  result = await db_session.execute(
      select(PredicaoIA)
      .options(selectinload(PredicaoIA.exame))
      .where(PredicaoIA.id_exame == exame.id)
  )
  laudo = result.scalar_one()

  assert laudo.exame.id == exame.id
  assert laudo.exame.arquivo_path.endswith(".edf")
