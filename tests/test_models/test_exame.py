"""Testes do modelo Exame (prontuário do sinal EEG)."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.models import Exame


@pytest.mark.asyncio
async def test_criar_exame_com_arquivo_path_no_disco(db_session, paciente):
  caminho = "C:/Users/Foco/Documents/DEV/eeg-xai/storage/edf/paciente_01.edf"
  exame = Exame(
      id_paciente=paciente.id,
      taxa_amostragem=512.0,
      arquivo_path=caminho,
  )
  db_session.add(exame)
  await db_session.commit()
  await db_session.refresh(exame)

  assert exame.id is not None
  assert exame.arquivo_path == caminho
  assert exame.taxa_amostragem == 512.0
  assert exame.id_paciente == paciente.id
  assert exame.data_upload is not None
  assert exame.status_exame == "pendente"
  assert exame.laudo_texto is None


@pytest.mark.asyncio
async def test_exame_nao_armazena_dados_binarios_no_banco(db_session, exame):
  """Regra de ouro: apenas metadados e caminho do arquivo."""
  colunas = {c.name for c in Exame.__table__.columns}
  assert "arquivo_path" in colunas
  assert "dados_sinal" not in colunas
  assert "valores_brutos" not in colunas
  assert "blob" not in colunas


@pytest.mark.asyncio
async def test_exame_requer_paciente_valido(db_session):
  invalido = Exame(
      id_paciente=99999,
      taxa_amostragem=256.0,
      arquivo_path="/tmp/inexistente.edf",
  )
  db_session.add(invalido)

  with pytest.raises(IntegrityError):
      await db_session.commit()


@pytest.mark.asyncio
async def test_exame_possui_relacionamento_com_predicao(db_session, exame):
  from app.models import PredicaoIA

  predicao = PredicaoIA(
      id_exame=exame.id,
      resultado_score=0.87,
      mapa_shap_path="C:/storage/shap/exame_001_shap.png",
  )
  db_session.add(predicao)
  await db_session.commit()

  result = await db_session.execute(
      select(Exame)
      .options(selectinload(Exame.predicoes))
      .where(Exame.id == exame.id)
  )
  registro = result.scalar_one()

  assert len(registro.predicoes) == 1
  assert registro.predicoes[0].resultado_score == pytest.approx(0.87)
