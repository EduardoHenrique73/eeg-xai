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
      toastSucesso('Laudo emitido com sucesso! Registro clínico finalizado.')
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

  const semCanaisSelecionados = canaisSelecionados.length === 0
  const botaoDesabilitado =
    analiseEmAndamento ||
    solicitarDesabilitado ||
    laudoSalvo ||
    semCanaisSelecionados ||
    carregandoCanais

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
          Enfileira inferência CNN-LSTM multicanal e geração do mapa SHAP no backend.
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
            ? 'Análise concluída'
            : analiseEmAndamento
              ? 'Processando CNN-LSTM...'
              : solicitarDesabilitado
                ? 'Aguardando upload do .edf'
                : semCanaisSelecionados
                  ? 'Selecione canais EEG'
                  : `Solicitar Análise IA (${canaisSelecionados.length} canal${canaisSelecionados.length !== 1 ? 'is' : ''})`}
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
        status={scoreStatus}
      />

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
