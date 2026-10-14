interface ScoreConfiancaProps {
  score?: number | null
  classificacao?: string | null
  threshold?: number | null
  status?: 'idle' | 'processando' | 'concluido'
  scoreMedioTrecho?: number | null
  resultadoPositivoConclusivo?: boolean
  resultadoConclusivo?: boolean
  indiceTrecho?: number
}

function formatarClassificacao(texto: string): string {
  return texto
    .replace(/Padrao/g, 'Padrão')
    .replace(/padrao/g, 'padrão')
    .replace(/revisao/g, 'revisão')
    .replace(/medica/g, 'médica')
    .replace(/persistencia/g, 'persistência')
    .replace(/necessaria/g, 'necessária')
}

export function ScoreConfianca({
  score,
  classificacao,
  threshold = 0.5,
  status = 'idle',
  scoreMedioTrecho,
  resultadoPositivoConclusivo = false,
  resultadoConclusivo = true,
  indiceTrecho = 0,
}: ScoreConfiancaProps) {
  const scoreExibido = scoreMedioTrecho ?? score
  const percentual = scoreExibido != null ? scoreExibido * 100 : null
  const limiar = threshold ?? 0.5
  const barraCor = resultadoPositivoConclusivo ? 'bg-alert-crisis' : 'bg-accent'

  return (
    <section className="rounded-md border border-clinical-200 bg-white p-4 shadow-clinical">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-clinical-500">
        {scoreMedioTrecho != null ? 'Score do candidato em destaque' : 'Score de suspeita'}
      </h3>

      {status === 'idle' && (
        <p className="mt-4 text-sm text-clinical-500">
          A análise ainda não foi iniciada.
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
              {percentual.toFixed(2)}
              <span className="text-xl font-semibold text-clinical-500">%</span>
            </p>
            {classificacao && (
              <div className="max-w-[65%] text-right">
                <span className="block text-[10px] font-semibold uppercase text-clinical-500">Conclusão global da IA</span>
                <span
                  className={[
                    'mt-1 inline-block rounded-md px-2.5 py-1 text-xs font-semibold',
                    resultadoPositivoConclusivo
                      ? 'bg-red-50 text-alert-crisis'
                      : 'bg-accent-light text-accent-dark',
                  ].join(' ')}
                >
                  {formatarClassificacao(classificacao).replace(/\s*\([^)]*\)\s*$/, '')}
                </span>
                {resultadoPositivoConclusivo && (
                  <span className="mt-1 block text-[10px] font-medium text-clinical-500">
                    Revisão médica necessária
                  </span>
                )}
              </div>
            )}
          </div>

          <div className="mt-4 h-3 overflow-hidden rounded-full bg-clinical-100">
            <div
              className={`h-full rounded-full transition-all ${barraCor}`}
              style={{ width: `${Math.min(100, Math.max(0, percentual))}%` }}
            />
          </div>

          <p className="mt-2 text-xs text-clinical-500">
            {scoreMedioTrecho != null
              ? `Score médio do candidato ${indiceTrecho + 1} em destaque; não representa certeza diagnóstica.`
              : 'Maior score produzido pelo modelo; não representa certeza diagnóstica.'}
          </p>
          <p className="mt-1 text-xs text-clinical-500">
            Limiar de decisão usado: {Math.round(limiar * 100)}%.
          </p>
          {!resultadoConclusivo && (
            <p className="mt-2 rounded-md bg-amber-50 px-2.5 py-2 text-xs font-medium text-amber-800">
              Resultado inconclusivo pelas regras temporais. Revise os trechos e o traçado antes de interpretar.
            </p>
          )}
        </div>
      )}
    </section>
  )
}
