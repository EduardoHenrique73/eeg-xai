interface ScoreConfiancaProps {
  score?: number | null
  classificacao?: string | null
  threshold?: number | null
  status?: 'idle' | 'processando' | 'concluido'
}

export function ScoreConfianca({
  score,
  classificacao,
  threshold = 0.5,
  status = 'idle',
}: ScoreConfiancaProps) {
  const percentual = score != null ? Math.round(score * 100) : null
  const limiar = threshold ?? 0.5
  const barraCor = score != null && score > limiar ? 'bg-alert-crisis' : 'bg-alert-normal'

  return (
    <section className="rounded-xl border border-clinical-200 bg-white p-5 shadow-clinical">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-clinical-500">
        Score de Suspeita
      </h3>

      {status === 'idle' && (
        <p className="mt-4 text-sm text-clinical-500">
          A analise ainda nao foi iniciada.
        </p>
      )}

      {status === 'processando' && (
        <div className="mt-4">
          <p className="text-sm font-medium text-accent-dark">
            Processando sinais, score e explicabilidade...
          </p>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-clinical-100">
            <div className="h-full w-2/3 animate-pulse rounded-full bg-accent" />
          </div>
        </div>
      )}

      {status === 'concluido' && percentual != null && (
        <div className="mt-4">
          <div className="flex items-end justify-between gap-2">
            <p className="text-4xl font-bold tabular-nums text-clinical-900">
              {percentual}
              <span className="text-xl font-semibold text-clinical-500">%</span>
            </p>
            {classificacao && (
              <span
                className={[
                  'rounded-full px-3 py-1 text-xs font-semibold',
                  score != null && score > limiar
                    ? 'bg-red-50 text-alert-crisis'
                    : 'bg-green-50 text-alert-normal',
                ].join(' ')}
              >
                {classificacao}
              </span>
            )}
          </div>

          <div className="mt-4 h-3 overflow-hidden rounded-full bg-clinical-100">
            <div
              className={`h-full rounded-full transition-all ${barraCor}`}
              style={{ width: `${percentual}%` }}
            />
          </div>

          <p className="mt-2 text-xs text-clinical-500">
            Estimativa do modelo para atividade compativel com crise epileptica.
          </p>
          <p className="mt-1 text-xs text-clinical-500">
            Limiar clinico configurado: {Math.round(limiar * 100)}%.
          </p>
        </div>
      )}
    </section>
  )
}
