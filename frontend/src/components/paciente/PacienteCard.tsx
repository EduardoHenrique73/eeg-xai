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
    <section className="rounded-xl border border-clinical-200 bg-white p-5 shadow-clinical">
      <header className="mb-4 border-b border-clinical-100 pb-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-clinical-500">
          Paciente
        </p>
        <h2 className="mt-1 text-lg font-semibold text-clinical-900">
          {paciente.nome}
        </h2>
      </header>

      <dl className="space-y-3 text-sm">
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
        <p className="mt-4 rounded-lg bg-clinical-50 p-3 text-sm leading-relaxed text-clinical-700">
          {paciente.observacoes}
        </p>
      )}
    </section>
  )
}
