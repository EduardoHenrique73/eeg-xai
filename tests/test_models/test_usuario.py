"""Testes do modelo Usuario (médico)."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.models import Usuario


@pytest.mark.asyncio
async def test_criar_usuario_com_campos_obrigatorios(db_session):
  medico = Usuario(
      nome="Dr. Carlos Mendes",
      crm="654321-RJ",
      email="carlos@clinica.com",
      senha_hash="hash_seguro",
  )
  db_session.add(medico)
  await db_session.commit()
  await db_session.refresh(medico)

  assert medico.id is not None
  assert medico.nome == "Dr. Carlos Mendes"
  assert medico.crm == "654321-RJ"
  assert medico.email == "carlos@clinica.com"
  assert medico.senha_hash == "hash_seguro"
  assert medico.created_at is not None


@pytest.mark.asyncio
async def test_email_usuario_deve_ser_unico(db_session, usuario_medico):
  duplicado = Usuario(
      nome="Outro Médico",
      crm="999999-SP",
      email=usuario_medico.email,
      senha_hash="outro_hash",
  )
  db_session.add(duplicado)

  with pytest.raises(IntegrityError):
      await db_session.commit()


@pytest.mark.asyncio
async def test_usuario_possui_relacionamento_com_pacientes(db_session, usuario_medico, paciente):
  result = await db_session.execute(
      select(Usuario)
      .options(selectinload(Usuario.pacientes))
      .where(Usuario.id == usuario_medico.id)
  )
  medico = result.scalar_one()

  assert len(medico.pacientes) == 1
  assert medico.pacientes[0].id == paciente.id
