interface CanalSelectorProps {
  canais: string[]
  selecionados: string[]
  onChange: (canais: string[]) => void
  disabled?: boolean
  carregando?: boolean
}

export function CanalSelector({
  canais,
  selecionados,
  onChange,
  disabled = false,
  carregando = false,
}: CanalSelectorProps) {
  const todosSelecionados =
    canais.length > 0 && canais.every((c) => selecionados.includes(c))

  const toggleCanal = (canal: string) => {
    if (disabled) return
    if (selecionados.includes(canal)) {
      onChange(selecionados.filter((c) => c !== canal))
    } else {
      onChange([...selecionados, canal])
    }
  }

  const selecionarTodos = () => {
    if (disabled) return
    onChange([...canais])
  }

  const limparSelecao = () => {
    if (disabled) return
    onChange([])
  }

  if (carregando) {
    return (
      <section className="rounded-xl border border-clinical-200 bg-white p-4 shadow-clinical">
        <p className="text-sm text-clinical-500 animate-pulse">
          Carregando canais EEG do exame...
        </p>
      </section>
    )
  }

  if (canais.length === 0) {
    return (
      <section className="rounded-xl border border-dashed border-clinical-200 bg-clinical-50 p-4">
        <p className="text-sm text-clinical-500">
          Envie um arquivo .edf para listar os canais disponíveis.
        </p>
      </section>
    )
  }

  return (
    <section className="rounded-xl border border-clinical-200 bg-white p-4 shadow-clinical">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-clinical-500">
            Canais EEG
          </h3>
          <p className="mt-0.5 text-xs text-clinical-500">
            {selecionados.length} de {canais.length} selecionado
            {selecionados.length !== 1 ? 's' : ''}
          </p>
        </div>
        <button
          type="button"
          onClick={todosSelecionados ? limparSelecao : selecionarTodos}
          disabled={disabled}
          className="shrink-0 rounded-lg border border-accent/30 bg-accent/5 px-2.5 py-1 text-xs font-semibold text-accent transition hover:bg-accent/10 disabled:opacity-50"
        >
          {todosSelecionados ? 'Limpar seleção' : 'Selecionar todos'}
        </button>
      </div>

      <div className="max-h-48 space-y-1 overflow-y-auto pr-1">
        {canais.map((canal) => {
          const marcado = selecionados.includes(canal)
          return (
            <label
              key={canal}
              className={[
                'flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition',
                marcado ? 'bg-accent/10 text-clinical-900' : 'hover:bg-clinical-50',
                disabled ? 'cursor-not-allowed opacity-60' : '',
              ].join(' ')}
            >
              <input
                type="checkbox"
                checked={marcado}
                disabled={disabled}
                onChange={() => toggleCanal(canal)}
                className="h-4 w-4 rounded border-clinical-300 text-accent focus:ring-accent/30"
              />
              <span className="font-mono text-xs">{canal}</span>
            </label>
          )
        })}
      </div>

      {selecionados.length === 0 && (
        <p className="mt-2 text-xs font-medium text-orange-600" role="alert">
          Selecione ao menos um canal para solicitar a análise IA.
        </p>
      )}
    </section>
  )
}
