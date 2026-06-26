import { useCallback, useEffect, useState } from 'react'
import { salvarLaudo } from '../../api/exames'
import { useToast } from '../../contexts/ToastContext'
import { AnaliseLoadingOverlay } from '../visualizador/AnaliseLoadingOverlay'
import { CanalSelector } from '../visualizador/CanalSelector'
import { ScoreConfianca } from './ScoreConfianca'
import { ParecerMedico } from './ParecerMedico'

interface IaLaudoPanelProps {
  exameId: number | null
  canaisEeg: string[]
  canaisSelecionados: string[]
  onCanaisChange: (canais: string[]) => void
  carregandoCanais?: boolean
  onSolicitarAnalise: () => void
  analiseEmAndamento: boolean
  analiseConcluida: boolean
  solicitarDesabilitado?: boolean
  score?: number | null
  classificacao?: string | null
  threshold?: number | null
  featureMode?: string | null
  canaisProcessados?: string[]
  canaisOmitidos?: string[]
  canaisDestaque?: Array<{
    canal: string
    score: number
    impacto?: number
    score_sem_canal?: number
  }>
  nJanelasAnalisadas?: number | null
  janelaPico?: {
    start_seconds: number
    end_seconds: number
    score?: number
  } | null
  janelasTop?: Array<{
    start_seconds: number
    end_seconds: number
    score: number
  }>
  laudoTextoInicial?: string | null
  statusExameInicial?: string | null
  erro?: string | null
}

export function IaLaudoPanel({
  exameId,
  canaisEeg,
  canaisSelecionados,
  onCanaisChange,
  carregandoCanais = false,
  onSolicitarAnalise,
  analiseEmAndamento,
  analiseConcluida,
  solicitarDesabilitado = false,
  score,
  classificacao,
  threshold,
  featureMode,
  canaisProcessados = [],
  canaisOmitidos = [],
  canaisDestaque = [],
  nJanelasAnalisadas,
  janelaPico,
  janelasTop = [],
  laudoTextoInicial,
  statusExameInicial,
  erro,
}: IaLaudoPanelProps) {
  const { sucesso: toastSucesso, erro: toastErro } = useToast()
  const [parecer, setParecer] = useState('')
  const [laudoSalvo, setLaudoSalvo] = useState(false)
  const [salvandoLaudo, setSalvandoLaudo] = useState(false)
  const [erroLaudo, setErroLaudo] = useState<string | null>(null)

  useEffect(() => {
    setParecer(laudoTextoInicial ?? '')
    setLaudoSalvo(statusExameInicial === 'concluido')
    setErroLaudo(null)
  }, [exameId, laudoTextoInicial, statusExameInicial])

  const handleEmitirLaudo = useCallback(async () => {
    if (exameId == null || laudoSalvo) return

    const texto = parecer.trim()
    if (!texto) return

    setSalvandoLaudo(true)
    setErroLaudo(null)

    try {
      await salvarLaudo(exameId, texto)
      setLaudoSalvo(true)
      toastSucesso('Laudo emitido com sucesso. Registro clinico finalizado.')
    } catch {
      const msg = 'Nao foi possivel salvar o laudo. Tente novamente.'
      setErroLaudo(msg)
      toastErro(msg)
    } finally {
      setSalvandoLaudo(false)
    }
  }, [exameId, laudoSalvo, parecer, toastSucesso, toastErro])

  const scoreStatus = analiseConcluida
    ? 'concluido'
    : analiseEmAndamento
      ? 'processando'
      : 'idle'

  const semCanaisSelecionados = canaisSelecionados.length === 0
  const botaoDesabilitado =
    analiseEmAndamento ||
    solicitarDesabilitado ||
    laudoSalvo ||
    semCanaisSelecionados ||
    carregandoCanais

  const formatarPercentual = (valor?: number) => {
    if (valor == null || Number.isNaN(valor)) return null
    return `${(valor * 100).toFixed(2)}%`
  }

  const formatarTempo = (valor?: number) => {
    if (valor == null || Number.isNaN(valor)) return '-'
    return `${valor.toFixed(1)}s`
  }

  return (
    <div className="relative flex h-full flex-col gap-4">
      <AnaliseLoadingOverlay
        visivel={analiseEmAndamento}
        nCanais={canaisSelecionados.length || canaisEeg.length}
      />

      <CanalSelector
        canais={canaisEeg}
        selecionados={canaisSelecionados}
        onChange={onCanaisChange}
        disabled={analiseEmAndamento || analiseConcluida || laudoSalvo}
        carregando={carregandoCanais}
      />

      <section className="rounded-xl border border-clinical-200 bg-white p-5 shadow-clinical">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-clinical-500">
          Motor de IA
        </h3>
        <p className="mt-2 text-sm text-clinical-600">
          Executa a analise multicanal, calcula o score de suspeita e gera o mapa de explicabilidade.
        </p>

        <button
          type="button"
          onClick={onSolicitarAnalise}
          disabled={botaoDesabilitado}
          className={[
            'mt-4 flex w-full items-center justify-center gap-2 rounded-lg px-4 py-3 text-sm font-semibold shadow-sm transition disabled:cursor-not-allowed disabled:opacity-60',
            analiseConcluida
              ? 'bg-green-50 text-alert-normal'
              : botaoDesabilitado && !analiseConcluida
                ? 'bg-clinical-100 text-clinical-400'
                : 'bg-accent text-white hover:bg-accent-dark',
          ].join(' ')}
        >
          {analiseEmAndamento && (
            <svg className="h-4 w-4 animate-spin-slow" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 3v3m6.364 1.636l-2.121 2.121M21 12h-3m-1.636 6.364l-2.121-2.121M12 21v-3m-6.364-1.636l2.121 2.121M3 12h3m1.636-6.364l2.121 2.121" />
            </svg>
          )}
          {analiseConcluida && (
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          )}
          {analiseConcluida
            ? 'Analise concluida'
            : analiseEmAndamento
              ? 'Processando analise...'
              : solicitarDesabilitado
                ? 'Aguardando upload do .edf'
                : semCanaisSelecionados
                  ? 'Selecione canais EEG'
                  : `Iniciar analise IA (${canaisSelecionados.length} canal${canaisSelecionados.length !== 1 ? 'is' : ''})`}
        </button>

        {(erro || erroLaudo) && (
          <p className="mt-3 text-sm font-medium text-alert-crisis" role="alert">
            {erroLaudo ?? erro}
          </p>
        )}
      </section>

      <ScoreConfianca
        score={score}
        classificacao={classificacao}
        threshold={threshold}
        status={scoreStatus}
      />

      {analiseConcluida && (
        <section className="rounded-xl border border-clinical-200 bg-white p-5 shadow-clinical">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-clinical-500">
            Interpretacao por Canais
          </h3>
          <div className="mt-3 space-y-3 text-sm text-clinical-700">
            <p>
              <span className="font-medium text-clinical-800">Modo:</span>{' '}
              {featureMode === 'per_channel' ? 'Analise por canal' : 'Analise agregada'}
            </p>

            {nJanelasAnalisadas != null && nJanelasAnalisadas > 0 && (
              <div className="rounded-md bg-clinical-50 px-3 py-2">
                <p>
                  <span className="font-medium text-clinical-800">Janelas analisadas:</span>{' '}
                  {nJanelasAnalisadas}
                </p>
                {janelaPico && (
                  <p className="mt-1 text-xs text-clinical-600">
                    Score calculado a partir da janela mais suspeita:{' '}
                    {formatarTempo(janelaPico.start_seconds)} a {formatarTempo(janelaPico.end_seconds)}
                    {janelaPico.score != null ? ` (${formatarPercentual(janelaPico.score)})` : ''}.
                  </p>
                )}
              </div>
            )}

            {janelasTop.length > 0 && (
              <div>
                <p className="font-medium text-clinical-800">Janelas com maior score</p>
                <div className="mt-2 space-y-2">
                  {janelasTop.slice(0, 3).map((janela) => (
                    <div
                      key={`${janela.start_seconds}-${janela.end_seconds}`}
                      className="flex items-center justify-between gap-3 rounded-md bg-clinical-50 px-3 py-2 text-xs"
                    >
                      <span className="font-medium text-clinical-700">
                        {formatarTempo(janela.start_seconds)} - {formatarTempo(janela.end_seconds)}
                      </span>
                      <span className="text-clinical-600">
                        score {formatarPercentual(janela.score) ?? '-'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {canaisProcessados.length > 0 && (
              <div>
                <p className="font-medium text-clinical-800">Canais incluidos na analise</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {canaisProcessados.map((canal) => (
                    <span
                      key={canal}
                      className="rounded-md bg-clinical-100 px-2 py-1 text-xs font-medium text-clinical-700"
                    >
                      {canal}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {canaisDestaque.length > 0 && (
              <div>
                <p className="font-medium text-clinical-800">Canais com maior influencia no resultado</p>
                <p className="mt-1 text-xs text-clinical-500">
                  O impacto mostra quanto o score muda quando o canal e removido da analise.
                </p>
                <div className="mt-2 space-y-2">
                  {canaisDestaque.map((item) => (
                    <div
                      key={item.canal}
                      className="rounded-md bg-clinical-50 px-3 py-2"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-medium text-clinical-800">{item.canal}</span>
                        <span className="text-xs text-clinical-600">
                          variacao absoluta do score {formatarPercentual(item.score) ?? '-'}
                        </span>
                      </div>
                      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-clinical-600">
                        <span>
                          diferenca no score:{' '}
                          <span className="font-medium text-clinical-700">
                            {formatarPercentual(item.impacto) ?? '-'}
                          </span>
                        </span>
                        <span>
                          score estimado sem este canal:{' '}
                          <span className="font-medium text-clinical-700">
                            {formatarPercentual(item.score_sem_canal) ?? '-'}
                          </span>
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {canaisOmitidos.length > 0 && (
              <div>
                <p className="font-medium text-clinical-800">Canais fora desta execucao</p>
                <p className="mt-1 text-xs text-clinical-500">
                  Estes canais nao entraram no calculo atual e nao influenciaram o score.
                </p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {canaisOmitidos.map((canal) => (
                    <span
                      key={canal}
                      className="rounded-md bg-amber-50 px-2 py-1 text-xs font-medium text-amber-700"
                    >
                      {canal}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      <div className="flex-1">
        <ParecerMedico
          value={parecer}
          onChange={setParecer}
          onEmitirLaudo={() => void handleEmitirLaudo()}
          disabled={!analiseConcluida}
          bloqueado={laudoSalvo}
          salvando={salvandoLaudo}
        />
      </div>
    </div>
  )
}
