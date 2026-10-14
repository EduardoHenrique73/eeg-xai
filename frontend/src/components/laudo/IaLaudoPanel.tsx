import { useCallback, useEffect, useState } from 'react'
import { CheckCircle2, Eye, LoaderCircle } from 'lucide-react'
import { salvarLaudo } from '../../api/exames'
import type { TrechoSuspeito } from '../../types/api'
import { useToast } from '../../contexts/ToastContext'
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
  montagemIncompleta?: boolean
  coberturaExcessiva?: boolean
  resultadoConclusivo?: boolean
  resultadoPositivoConclusivo?: boolean
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
  trechoSuspeito?: TrechoSuspeito | null
  topTrechosSuspeitos?: TrechoSuspeito[]
  trechoSelecionadoIndice?: number
  onSelecionarTrecho?: (indice: number) => void
  modelType?: string | null
  nSequencesAnalisadas?: number | null
  minDurationSeconds?: number | null
  maxGapSeconds?: number | null
  decisionThreshold?: number | null
  xaiMethod?: string | null
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
  montagemIncompleta = false,
  coberturaExcessiva = false,
  resultadoConclusivo = true,
  resultadoPositivoConclusivo = false,
  canaisDestaque = [],
  nJanelasAnalisadas,
  janelaPico,
  janelasTop = [],
  trechoSuspeito,
  topTrechosSuspeitos = [],
  trechoSelecionadoIndice = 0,
  onSelecionarTrecho,
  modelType,
  nSequencesAnalisadas,
  minDurationSeconds,
  maxGapSeconds,
  decisionThreshold,
  xaiMethod,
  laudoTextoInicial,
  statusExameInicial,
  erro,
}: IaLaudoPanelProps) {
  const { sucesso: toastSucesso, erro: toastErro } = useToast()
  const [parecer, setParecer] = useState('')
  const [laudoSalvo, setLaudoSalvo] = useState(false)
  const [salvandoLaudo, setSalvandoLaudo] = useState(false)
  const [erroLaudo, setErroLaudo] = useState<string | null>(null)
  const [aba, setAba] = useState<'analise' | 'laudo'>('analise')
  const [mostrarTodosTrechos, setMostrarTodosTrechos] = useState(false)

  useEffect(() => {
    setParecer(laudoTextoInicial ?? '')
    setLaudoSalvo(statusExameInicial === 'concluido')
    setErroLaudo(null)
    setAba('analise')
    setMostrarTodosTrechos(false)
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
      toastSucesso('Laudo emitido com sucesso. Registro clínico finalizado.')
    } catch {
      const msg = 'Não foi possível salvar o laudo. Tente novamente.'
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
  const trechoEmDestaque = topTrechosSuspeitos[trechoSelecionadoIndice] ?? trechoSuspeito

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
      <div className="flex rounded-md border border-clinical-200 bg-white p-1" role="group" aria-label="Painel clínico">
        <button type="button" onClick={() => setAba('analise')} aria-pressed={aba === 'analise'} className={`flex-1 rounded px-3 py-2 text-sm font-medium ${aba === 'analise' ? 'bg-accent-light text-accent-dark' : 'text-clinical-600 hover:bg-clinical-50'}`}>Análise</button>
        <button type="button" onClick={() => setAba('laudo')} aria-pressed={aba === 'laudo'} className={`flex-1 rounded px-3 py-2 text-sm font-medium ${aba === 'laudo' ? 'bg-accent-light text-accent-dark' : 'text-clinical-600 hover:bg-clinical-50'}`}>Laudo</button>
      </div>

      {aba === 'analise' ? <>

      {!analiseConcluida && <CanalSelector
        canais={canaisEeg}
        selecionados={canaisSelecionados}
        onChange={onCanaisChange}
        disabled={analiseEmAndamento || laudoSalvo}
        carregando={carregandoCanais}
      />}

      {!analiseConcluida && <section className="rounded-md border border-clinical-200 bg-white p-4 shadow-clinical">
        <h3 className="text-sm font-semibold uppercase tracking-wide text-clinical-500">
          Motor de IA
        </h3>
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
            <LoaderCircle size={16} className="animate-spin" aria-hidden="true" />
          )}
          {analiseConcluida && (
            <CheckCircle2 size={16} aria-hidden="true" />
          )}
          {analiseConcluida
            ? 'Análise concluída'
            : analiseEmAndamento
              ? 'Processando análise...'
              : solicitarDesabilitado
                ? 'Aguardando upload do .edf'
                : semCanaisSelecionados
                  ? 'Selecione canais EEG'
                  : `Iniciar análise IA (${canaisSelecionados.length} ${canaisSelecionados.length === 1 ? 'canal' : 'canais'})`}
        </button>

        {(erro || erroLaudo) && (
          <p className="mt-3 text-sm font-medium text-alert-crisis" role="alert">
            {erroLaudo ?? erro}
          </p>
        )}
      </section>}

      <ScoreConfianca
        score={score}
        classificacao={classificacao}
        threshold={threshold}
        status={scoreStatus}
        scoreMedioTrecho={trechoEmDestaque?.score_medio}
        resultadoPositivoConclusivo={resultadoPositivoConclusivo}
        resultadoConclusivo={resultadoConclusivo}
        indiceTrecho={trechoSelecionadoIndice}
      />

      {analiseConcluida && (
        <section className="rounded-md border border-clinical-200 bg-white p-4 shadow-clinical">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-clinical-500">
            Resultado temporal
          </h3>
          <div className="mt-3 space-y-3 text-sm text-clinical-700">
            {modelType === 'sequence_cnn_lstm' && (
              <div className="rounded-md border border-accent/30 bg-accent/5 px-3 py-2">
                <p className="text-xs font-semibold uppercase tracking-wide text-accent-dark">
                  Modelo sequencial global (inter-paciente)
                </p>
                <p className="mt-1 text-xs text-clinical-600">
                  Resultado de apoio à decisão por detecção de trechos contínuos
                  suspeitos. Não constitui diagnóstico final.
                </p>
                {!resultadoConclusivo && (
                  <p className="mt-2 rounded-md bg-amber-50 px-2 py-1 text-xs font-medium text-amber-800">
                    Resultado indeterminado: a IA não gerou um positivo conclusivo para este exame.
                  </p>
                )}
                {resultadoPositivoConclusivo && (
                  <p className="mt-2 rounded-md bg-red-50 px-2 py-1 text-xs font-medium text-alert-crisis">
                    Positivo conclusivo pelo critério automático; exige revisão médica do trecho destacado.
                  </p>
                )}
              </div>
            )}

            {trechoSuspeito && topTrechosSuspeitos.length <= 1 && (
              <div className="rounded-md bg-clinical-50 px-3 py-2">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="font-medium text-clinical-800">Trecho contínuo suspeito</p>
                    <p className="mt-1 text-xs text-clinical-600">
                      {formatarTempo(trechoSuspeito.start_seconds)} - {formatarTempo(trechoSuspeito.end_seconds)}
                      {' '}({trechoSuspeito.duration_seconds.toFixed(1)}s, {trechoSuspeito.n_janelas} janelas)
                    </p>
                  </div>
                  <span
                    className={[
                      'rounded-md px-2 py-1 text-xs font-semibold',
                      trechoSuspeito.atingiu_duracao_minima
                        ? 'bg-red-50 text-alert-crisis'
                        : 'bg-amber-50 text-amber-700',
                    ].join(' ')}
                  >
                    {trechoSuspeito.atingiu_duracao_minima ? 'duração relevante' : 'trecho curto'}
                  </span>
                </div>
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-clinical-600">
                  {trechoSuspeito.coverage_ratio != null && (
                    <span>
                      cobertura do exame:{' '}
                      <span className="font-medium text-clinical-700">
                        {formatarPercentual(trechoSuspeito.coverage_ratio) ?? '-'}
                      </span>
                    </span>
                  )}
                  <span>
                    score médio:{' '}
                    <span className="font-medium text-clinical-700">
                      {formatarPercentual(trechoSuspeito.score_medio) ?? '-'}
                    </span>
                  </span>
                  <span>
                    maior score:{' '}
                    <span className="font-medium text-clinical-700">
                      {formatarPercentual(trechoSuspeito.score_max) ?? '-'}
                    </span>
                  </span>
                  <span>
                    corte por janela:{' '}
                    <span className="font-medium text-clinical-700">
                      {formatarPercentual(trechoSuspeito.threshold) ?? '-'}
                    </span>
                  </span>
                </div>
                {trechoSuspeito.cobertura_excessiva && (
                  <p className="mt-2 text-xs font-medium text-amber-700">
                    Cobertura suspeita extensa demais para conclusão automática; revisar o exame antes de interpretar como crise.
                  </p>
                )}
                {coberturaExcessiva && !trechoSuspeito.cobertura_excessiva && (
                  <p className="mt-2 text-xs font-medium text-amber-700">
                    A regra global marcou cobertura excessiva e bloqueou conclusão automática.
                  </p>
                )}
              </div>
            )}

            {topTrechosSuspeitos.length > 1 && (
              <div>
                <p className="font-semibold text-clinical-900">Candidatos temporais para revisão</p>
                <p className="mt-1 text-xs text-clinical-500">
                  Selecione um candidato para destacá-lo no gráfico. A lista é ordenada pela regra temporal da IA e não confirma diagnóstico.
                </p>
                <div className="mt-2 space-y-2">
                  {(mostrarTodosTrechos ? topTrechosSuspeitos : topTrechosSuspeitos.slice(0, 3)).map((trecho, indice) => {
                    const selecionado = indice === trechoSelecionadoIndice
                    return (
                    <button
                      type="button"
                      key={`${trecho.start_seconds}-${trecho.end_seconds}-${indice}`}
                      onClick={() => onSelecionarTrecho?.(indice)}
                      aria-pressed={selecionado}
                      className={[
                        'w-full rounded-md border px-3 py-2 text-left text-xs transition',
                        selecionado
                          ? 'border-accent bg-accent-light shadow-sm ring-1 ring-accent/20'
                          : 'border-clinical-200 bg-white hover:border-accent/60 hover:bg-clinical-50',
                      ].join(' ')}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="flex items-center gap-2 font-semibold text-clinical-800">
                          <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-accent text-[10px] text-white">{indice + 1}</span>
                          {formatarTempo(trecho.start_seconds)} - {formatarTempo(trecho.end_seconds)}
                        </span>
                        <span
                          className={[
                            'rounded px-2 py-0.5 font-semibold',
                            trecho.atingiu_duracao_minima
                              ? 'bg-red-50 text-alert-crisis'
                              : 'bg-amber-50 text-amber-700',
                          ].join(' ')}
                        >
                          {trecho.duration_seconds.toFixed(1)}s
                        </span>
                      </div>
                      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-clinical-600">
                        <span className="font-medium text-accent-dark">
                          {indice === 0 ? 'principal da IA' : 'candidato secundário'}
                        </span>
                        <span>
                          score médio:{' '}
                          <span className="font-medium text-clinical-700">
                            {formatarPercentual(trecho.score_medio) ?? '-'}
                          </span>
                        </span>
                        <span>
                          maior score:{' '}
                          <span className="font-medium text-clinical-700">
                            {formatarPercentual(trecho.score_max) ?? '-'}
                          </span>
                        </span>
                      </div>
                      {selecionado && (
                        <span className="mt-2 inline-flex items-center gap-1 font-semibold text-accent-dark">
                          <Eye size={13} aria-hidden="true" /> Em destaque no gráfico
                        </span>
                      )}
                    </button>
                  )})}
                </div>
                {topTrechosSuspeitos.length > 3 && (
                  <button
                    type="button"
                    onClick={() => setMostrarTodosTrechos((atual) => !atual)}
                    className="mt-2 w-full rounded-md border border-clinical-200 px-3 py-2 text-xs font-semibold text-accent transition hover:bg-accent-light"
                  >
                    {mostrarTodosTrechos ? 'Mostrar somente os 3 principais' : `Ver mais ${topTrechosSuspeitos.length - 3} candidatos`}
                  </button>
                )}
              </div>
            )}

            {canaisProcessados.length > 0 && (
              <details className="rounded-md border border-clinical-200 bg-white">
                <summary className="cursor-pointer px-3 py-2 text-xs font-semibold text-clinical-700">
                  {canaisProcessados.length} canais incluídos na análise
                </summary>
                <div className="flex flex-wrap gap-2 border-t border-clinical-100 p-3">
                  {canaisProcessados.map((canal) => (
                    <span
                      key={canal}
                      className="rounded-md bg-clinical-100 px-2 py-1 text-xs font-medium text-clinical-700"
                    >
                      {canal}
                    </span>
                  ))}
                </div>
              </details>
            )}

            <details className="rounded-md border border-clinical-200 bg-white">
              <summary className="cursor-pointer px-3 py-2 text-xs font-semibold text-clinical-700">
                Detalhes do processamento
              </summary>
              <div className="space-y-3 border-t border-clinical-100 p-3 text-xs text-clinical-600">
                <p><span className="font-medium text-clinical-800">Modo:</span> {featureMode === 'per_channel' ? 'Análise por canal' : 'Análise agregada'}</p>
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  {nSequencesAnalisadas != null && <span>Sequências: <strong>{nSequencesAnalisadas}</strong></span>}
                  {nJanelasAnalisadas != null && <span>Janelas: <strong>{nJanelasAnalisadas}</strong></span>}
                  {decisionThreshold != null && <span>Limiar: <strong>{formatarPercentual(decisionThreshold)}</strong></span>}
                  {minDurationSeconds != null && <span>Duração mínima: <strong>{minDurationSeconds.toFixed(0)}s</strong></span>}
                  {maxGapSeconds != null && maxGapSeconds > 0 && <span>União: <strong>até {maxGapSeconds.toFixed(0)}s</strong></span>}
                  {xaiMethod && <span>Explicabilidade: <strong>{xaiMethod === 'gradient_shap' ? 'Gradient SHAP' : xaiMethod}</strong></span>}
                </div>
                {janelaPico && <p>Maior janela isolada: <strong>{formatarTempo(janelaPico.start_seconds)} a {formatarTempo(janelaPico.end_seconds)} ({formatarPercentual(janelaPico.score)})</strong>.</p>}
                {janelasTop.length > 0 && <p>As {janelasTop.length} janelas de maior score permanecem registradas no resultado da análise.</p>}
              </div>
            </details>

            {canaisDestaque.length > 0 && (
              <div>
                <p className="font-medium text-clinical-800">Canais com maior influência no resultado</p>
                <p className="mt-1 text-xs text-clinical-500">
                  O impacto mostra quanto o score muda quando o canal é removido da análise.
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
                          variação absoluta do score {formatarPercentual(item.score) ?? '-'}
                        </span>
                      </div>
                      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-clinical-600">
                        <span>
                          diferença no score:{' '}
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
                <p className="font-medium text-clinical-800">Canais fora desta execução</p>
                <p className="mt-1 text-xs text-clinical-500">
                  Estes canais não entraram no cálculo atual e não influenciaram o score.
                </p>
                {montagemIncompleta && (
                  <p className="mt-1 text-xs font-medium text-amber-700">
                    O modelo por canal foi treinado com a montagem completa; uma análise parcial reduz a confiabilidade do resultado.
                  </p>
                )}
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

      </> : <div className="flex-1">
        <ParecerMedico
          value={parecer}
          onChange={setParecer}
          onEmitirLaudo={() => void handleEmitirLaudo()}
          disabled={!analiseConcluida}
          bloqueado={laudoSalvo}
          salvando={salvandoLaudo}
        />
      </div>
      }
    </div>
  )
}
