"""Inicializa tabelas e dados mínimos para desenvolvimento local."""

from __future__ import annotations

import asyncio
from datetime import date

from sqlalchemy import select

from app.database import Base, engine, AsyncSessionLocal
from app.models import Paciente, Usuario


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        medico = await session.scalar(select(Usuario).where(Usuario.email == "ana.silva@hospital.com"))
        if medico is None:
            medico = Usuario(
                nome="Dr. Ana Silva",
                crm="123456-SP",
                email="ana.silva@hospital.com",
                senha_hash="$2b$12$placeholderhash",
            )
            session.add(medico)
            await session.flush()

        paciente = await session.scalar(select(Paciente).where(Paciente.id == 1))
        if paciente is None:
            existente = await session.scalar(
                select(Paciente).where(Paciente.cpf == "12345678901")
            )
            if existente is None:
                session.add(
                    Paciente(
                        nome="João da Costa",
                        data_nascimento=date(1985, 3, 15),
                        sexo="M",
                        cpf="12345678901",
                        telefone="11999998888",
                        observacoes="Histórico familiar de epilepsia",
                        id_usuario=medico.id,
                    )
                )

        await session.commit()

    print("Banco inicializado com sucesso (tabelas + paciente de desenvolvimento).")


if __name__ == "__main__":
    asyncio.run(main())
