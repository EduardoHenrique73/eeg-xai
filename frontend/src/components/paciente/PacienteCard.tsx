import type { Paciente } from '../../types/api'

interface PacienteCardProps {
  paciente: Paciente
}

function formatarData(iso: string): string {
  const [ano, mes, dia] = iso.split('-')
  return `${dia}/${mes}/${ano}`
}

export function PacienteCard({ paciente }: PacienteCardProps) {
  return (
    <section className="min-w-0 rounded-md border border-clinical-200 bg-white p-4 shadow-clinical xl:flex-[1.2] 2xl:flex-none">
      <header className="mb-4 border-b border-clinical-100 pb-3 xl:mb-2 xl:pb-2 2xl:mb-4 2xl:pb-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-clinical-500">
          Paciente
        </p>
        <h2 className="mt-1 text-lg font-semibold text-clinical-900">
          {paciente.nome}
        </h2>
      </header>

      <dl className="space-y-3 text-sm xl:grid xl:grid-cols-2 xl:gap-x-5 xl:gap-y-1 xl:space-y-0 2xl:block 2xl:space-y-3">
        <div className="flex justify-between gap-4">
          <dt className="text-clinical-500">ID</dt>
          <dd className="font-medium text-clinical-800">#{paciente.id}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-clinical-500">Nascimento</dt>
          <dd className="font-medium text-clinical-800">
            {formatarData(paciente.data_nascimento)}
          </dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-clinical-500">Sexo</dt>
          <dd className="font-medium text-clinical-800">{paciente.sexo}</dd>
        </div>
        {paciente.cpf && (
          <div className="flex justify-between gap-4">
            <dt className="text-clinical-500">CPF</dt>
            <dd className="font-mono text-clinical-800">{paciente.cpf}</dd>
          </div>
        )}
        {paciente.telefone && (
          <div className="flex justify-between gap-4">
            <dt className="text-clinical-500">Telefone</dt>
            <dd className="font-medium text-clinical-800">{paciente.telefone}</dd>
          </div>
        )}
      </dl>

      {paciente.observacoes && (
        <p className="mt-3 rounded-md bg-clinical-50 p-3 text-sm leading-relaxed text-clinical-700 2xl:mt-4">
          {paciente.observacoes}
        </p>
      )}
    </section>
  )
}
