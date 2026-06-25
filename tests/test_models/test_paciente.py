"""Testes do modelo Paciente."""

from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.models import Paciente, Usuario


@pytest.mark.asyncio
async def test_criar_paciente_vinculado_a_medico(db_session, usuario_medico):
  paciente = Paciente(
      nome="Maria Oliveira",
      data_nascimento=date(1990, 7, 22),
      sexo="F",
      cpf="98765432100",
      telefone="21988887777",
      observacoes=None,
      id_usuario=usuario_medico.id,
  )
  db_session.add(paciente)
  await db_session.commit()
  await db_session.refresh(paciente)

  assert paciente.id is not None
  assert paciente.nome == "Maria Oliveira"
  assert paciente.sexo == "F"
  assert paciente.id_usuario == usuario_medico.id
  assert paciente.created_at is not None


@pytest.mark.asyncio
async def test_paciente_requer_medico_valido(db_session):
  orfao = Paciente(
      nome="Sem Médico",
      data_nascimento=date(2000, 1, 1),
      sexo="M",
      id_usuario=99999,
  )
  db_session.add(orfao)

  with pytest.raises(IntegrityError):
      await db_session.commit()


@pytest.mark.asyncio
async def test_paciente_possui_relacionamento_com_exames(db_session, paciente, exame):
  result = await db_session.execute(
      select(Paciente)
      .options(selectinload(Paciente.exames))
      .where(Paciente.id == paciente.id)
  )
  registro = result.scalar_one()

  assert len(registro.exames) == 1
  assert registro.exames[0].arquivo_path.endswith(".edf")
