import { useEffect, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { obterSinaisExame } from '../../api/exames'
import type { EegPonto } from '../../types/api'

interface EegSignalChartProps {
  exameId: number | null
  mapaShapUrl?: string | null
  placeholder?: string
}

export function EegSignalChart({
  exameId,
  mapaShapUrl,
  placeholder = 'Aguardando importação do exame...',
}: EegSignalChartProps) {
  const [serie, setSerie] = useState<EegPonto[] | undefined>(undefined)
  const [carregando, setCarregando] = useState(false)
  const [erro, setErro] = useState<string | null>(null)
  const [shapExpandido, setShapExpandido] = useState(false)

  useEffect(() => {
    if (exameId == null) {
      setSerie(undefined)
      setCarregando(false)
      setErro(null)
      return
    }

    let cancelado = false
    setCarregando(true)
    setErro(null)
    setSerie(undefined)

    obterSinaisExame(exameId)
      .then((resposta) => {
        if (!cancelado) setSerie(resposta.pontos)
      })
      .catch(() => {
        if (!cancelado) {
          setErro('Não foi possível carregar as ondas cerebrais do exame.')
        }
      })
      .finally(() => {
        if (!cancelado) setCarregando(false)
      })

    return () => {
      cancelado = true
    }
  }, [exameId])

  const possuiSerie = Boolean(serie && serie.length > 0)
  const mensagem = carregando
    ? 'Carregando ondas cerebrais...'
    : erro ?? placeholder

  return (
    <section className="flex h-full min-h-0 flex-col rounded-xl border border-clinical-200 bg-white shadow-clinical">
      <header className="shrink-0 border-b border-clinical-100 px-5 py-4">
        <h2 className="text-base font-semibold text-clinical-900">
          Visualizador de Sinais EEG
        </h2>
        <p className="text-xs text-clinical-500">
          Série temporal real (média dos canais EEG)
        </p>
      </header>

      {/* Gráfico EEG — área superior */}
      <div
        className={[
          'shrink-0 p-4',
          mapaShapUrl ? 'h-[min(42vh,340px)]' : 'min-h-[320px] flex-1',
        ].join(' ')}
      >
        {!possuiSerie ? (
          <div className="flex h-full min-h-[260px] items-center justify-center rounded-lg border border-dashed border-clinical-200 bg-clinical-50">
            <p
              className={[
                'max-w-md text-center text-sm font-medium',
                carregando ? 'text-accent-dark' : 'text-clinical-500',
                erro ? 'text-alert-crisis' : '',
              ].join(' ')}
            >
              {mensagem}
            </p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={serie} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="tempo"
                tick={{ fontSize: 11, fill: '#64748b' }}
                label={{
                  value: 'Tempo (s)',
                  position: 'insideBottom',
                  offset: -4,
                  fill: '#64748b',
                  fontSize: 11,
                }}
              />
              <YAxis
                tick={{ fontSize: 11, fill: '#64748b' }}
                label={{
                  value: 'µV',
                  angle: -90,
                  position: 'insideLeft',
                  fill: '#64748b',
                  fontSize: 11,
                }}
              />
              <Tooltip
                contentStyle={{
                  borderRadius: 8,
                  borderColor: '#e2e8f0',
                  fontSize: 12,
                }}
              />
              <Line
                type="monotone"
                dataKey="amplitude"
                stroke="#0d9488"
                strokeWidth={1.2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Mapa SHAP — painel inferior separado */}
      {mapaShapUrl && (
        <div className="flex min-h-0 flex-1 flex-col border-t border-clinical-200">
          <div className="flex shrink-0 items-center justify-between border-b border-clinical-100 bg-clinical-50 px-5 py-3">
            <div>
              <h3 className="text-sm font-semibold text-clinical-900">
                Mapa de Explicabilidade (SHAP)
              </h3>
              <p className="text-xs text-clinical-500">
                Regiões e features que mais influenciaram a decisão da IA
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShapExpandido(true)}
              className="rounded-lg border border-clinical-300 bg-white px-3 py-1.5 text-xs font-medium text-clinical-700 transition hover:border-accent hover:text-accent"
            >
              Expandir
            </button>
          </div>

          <div className="min-h-0 flex-1 overflow-auto bg-clinical-50 p-4">
            <img
              src={mapaShapUrl}
              alt="Mapa de explicabilidade SHAP"
              className="mx-auto w-full max-w-3xl rounded-lg border border-clinical-200 bg-white object-contain shadow-sm"
            />
          </div>
        </div>
      )}

      {/* Modal fullscreen para inspeção detalhada */}
      {shapExpandido && mapaShapUrl && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-backdrop-in"
          role="dialog"
          aria-modal="true"
          aria-label="Mapa SHAP expandido"
        >
          <div
            className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm"
            onClick={() => setShapExpandido(false)}
            aria-hidden
          />
          <div className="relative flex max-h-[95vh] max-w-5xl flex-col rounded-2xl bg-white shadow-2xl animate-modal-in">
            <div className="flex items-center justify-between border-b border-clinical-100 px-5 py-3">
              <h3 className="text-sm font-semibold text-clinical-900">
                Mapa SHAP — Visualização ampliada
              </h3>
              <button
                type="button"
                onClick={() => setShapExpandido(false)}
                className="rounded-lg p-1.5 text-clinical-500 transition hover:bg-clinical-100"
                aria-label="Fechar"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="overflow-auto p-4">
              <img
                src={mapaShapUrl}
                alt="Mapa de explicabilidade SHAP ampliado"
                className="mx-auto w-full object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
