interface ParecerMedicoProps {
  value: string
  onChange: (value: string) => void
  onEmitirLaudo: () => void
  disabled?: boolean
  bloqueado?: boolean
  salvando?: boolean
}

export function ParecerMedico({
  value,
  onChange,
  onEmitirLaudo,
  disabled = false,
  bloqueado = false,
  salvando = false,
}: ParecerMedicoProps) {
  const somenteLeitura = disabled || bloqueado

  return (
    <section className="rounded-xl border border-clinical-200 bg-white p-5 shadow-clinical">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-clinical-500">
        Parecer Médico
      </h3>

      {bloqueado && (
        <p className="mt-2 rounded-lg bg-green-50 px-3 py-2 text-xs font-medium text-alert-normal">
          Laudo emitido — registro clínico imutável.
        </p>
      )}

      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        readOnly={somenteLeitura}
        placeholder="Descreva a interpretação clínica do exame, correlacionando o laudo da IA com o quadro do paciente..."
        rows={8}
        className={[
          'mt-3 w-full resize-none rounded-lg border px-3 py-2 text-sm leading-relaxed outline-none transition',
          somenteLeitura
            ? 'cursor-not-allowed border-clinical-200 bg-clinical-100 text-clinical-600'
            : 'border-clinical-200 bg-clinical-50 text-clinical-800 focus:border-accent focus:bg-white focus:ring-2 focus:ring-accent/20',
        ].join(' ')}
      />

      <button
        type="button"
        onClick={onEmitirLaudo}
        disabled={somenteLeitura || salvando || value.trim().length === 0}
        className="mt-4 w-full rounded-lg border border-clinical-300 bg-white px-4 py-3 text-sm font-semibold text-clinical-800 transition hover:border-clinical-500 hover:bg-clinical-50 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {salvando
          ? 'Salvando laudo...'
          : bloqueado
            ? 'Laudo Finalizado'
            : 'Emitir Laudo Finalizado'}
      </button>
    </section>
  )
}
