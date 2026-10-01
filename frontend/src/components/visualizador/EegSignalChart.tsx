import { useEffect, useMemo, useState } from 'react'
import {
  Brush,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { obterSinaisExame } from '../../api/exames'
import type { SinaisExameResponse, TrechoSuspeito } from '../../types/api'

interface EegSignalChartProps {
  exameId: number | null
  mapaShapUrl?: string | null
  canaisSelecionados?: string[]
  trechoSuspeito?: TrechoSuspeito | null
  topTrechosSuspeitos?: TrechoSuspeito[]
  placeholder?: string
}

export function EegSignalChart({
  exameId,
  mapaShapUrl,
  canaisSelecionados = [],
  trechoSuspeito,
  topTrechosSuspeitos = [],
  placeholder = 'Aguardando importacao do exame...',
}: EegSignalChartProps) {
  const [sinais, setSinais] = useState<SinaisExameResponse | undefined>(undefined)
  const [carregando, setCarregando] = useState(false)
  const [erro, setErro] = useState<string | null>(null)
  const [shapExpandido, setShapExpandido] = useState(false)
  const [escalaVertical, setEscalaVertical] = useState(1)

  useEffect(() => {
    if (exameId == null) {
      setSinais(undefined)
      setCarregando(false)
      setErro(null)
      setEscalaVertical(1)
      return
    }

    let cancelado = false
    setCarregando(true)
    setErro(null)
    setSinais(undefined)
    setEscalaVertical(1)

    obterSinaisExame(exameId)
      .then((resposta) => {
        if (!cancelado) setSinais(resposta)
      })
      .catch(() => {
        if (!cancelado) {
          setErro('Nao foi possivel carregar as ondas cerebrais do exame.')
        }
      })
      .finally(() => {
        if (!cancelado) setCarregando(false)
      })

    return () => {
      cancelado = true
    }
  }, [exameId])

  const seriesVisiveis = useMemo(() => {
    if (!sinais) return []
    const disponiveis = sinais.series?.length
      ? sinais.series
      : [{ canal: 'Media dos canais', pontos: sinais.pontos }]
    if (canaisSelecionados.length === 0) return disponiveis
    const filtradas = disponiveis.filter((serie) => canaisSelecionados.includes(serie.canal))
    return filtradas.length > 0 ? filtradas : disponiveis
  }, [sinais, canaisSelecionados])

  const dadosGrafico = useMemo(() => {
    if (seriesVisiveis.length === 0) return []
    const escalas = seriesVisiveis.map((serie) => {
      const absolutos = serie.pontos.map((ponto) => Math.abs(ponto.amplitude)).sort((a, b) => a - b)
      return Math.max(absolutos[Math.floor(absolutos.length * 0.95)] ?? 1, 1)
    })
    return seriesVisiveis[0].pontos.map((ponto, indice) => {
      const linha: Record<string, number> = { tempo: ponto.tempo }
      seriesVisiveis.forEach((serie, canalIndice) => {
        const base = (seriesVisiveis.length - canalIndice - 1) * 4
        const amplitude = serie.pontos[indice]?.amplitude ?? 0
        const normalizada = Math.max(-1.6, Math.min(1.6, amplitude / escalas[canalIndice]))
        linha[`canal_${canalIndice}`] = base + normalizada * escalaVertical
      })
      return linha
    })
  }, [seriesVisiveis, escalaVertical])

  const possuiSerie = dadosGrafico.length > 0
  const dominioY: [number, number] = [-2, Math.max(2, (seriesVisiveis.length - 1) * 4 + 2)]
  const ticksY = seriesVisiveis.map((_, indice) => (seriesVisiveis.length - indice - 1) * 4)

  const mensagem = carregando
    ? 'Carregando ondas cerebrais...'
    : erro ?? placeholder

  return (
    <section className="flex h-full min-h-0 flex-col rounded-xl border border-clinical-200 bg-white shadow-clinical">
      <header className="shrink-0 border-b border-clinical-100 px-5 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-clinical-900">
              Visualizador de Sinais EEG
            </h2>
            <p className="text-xs text-clinical-500">
              Tracados multicanais com zoom horizontal e marcacao dos trechos suspeitos
            </p>
          </div>

          {possuiSerie && (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setEscalaVertical((v) => Math.max(0.5, Number((v - 0.5).toFixed(1))))}
                className="rounded-lg border border-clinical-200 px-2.5 py-1 text-xs font-semibold text-clinical-700 transition hover:border-accent hover:text-accent"
              >
                Y-
              </button>
              <span className="w-12 text-center text-xs font-medium text-clinical-500">
                {escalaVertical.toFixed(1)}x
              </span>
              <button
                type="button"
                onClick={() => setEscalaVertical((v) => Math.min(5, Number((v + 0.5).toFixed(1))))}
                className="rounded-lg border border-clinical-200 px-2.5 py-1 text-xs font-semibold text-clinical-700 transition hover:border-accent hover:text-accent"
              >
                Y+
              </button>
              <button
                type="button"
                onClick={() => setEscalaVertical(1)}
                className="rounded-lg border border-clinical-200 px-2.5 py-1 text-xs font-semibold text-clinical-700 transition hover:border-accent hover:text-accent"
              >
                Reset
              </button>
            </div>
          )}
        </div>
      </header>

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
            <LineChart data={dadosGrafico} margin={{ top: 8, right: 16, left: 24, bottom: 8 }}>
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
                domain={dominioY}
                ticks={ticksY}
                tick={{ fontSize: 11, fill: '#64748b' }}
                tickFormatter={(value) => {
                  const indice = ticksY.indexOf(Number(value))
                  return indice >= 0 ? seriesVisiveis[indice].canal : ''
                }}
                width={82}
              />
              <Tooltip
                contentStyle={{
                  borderRadius: 8,
                  borderColor: '#e2e8f0',
                  fontSize: 12,
                }}
              />
              {topTrechosSuspeitos.slice(1).map((trecho, indice) => (
                <ReferenceArea
                  key={`${trecho.start_seconds}-${trecho.end_seconds}-${indice}`}
                  x1={trecho.start_seconds}
                  x2={trecho.end_seconds}
                  fill="#f59e0b"
                  fillOpacity={0.1}
                  strokeOpacity={0}
                />
              ))}
              {trechoSuspeito && (
                <ReferenceArea
                  x1={trechoSuspeito.start_seconds}
                  x2={trechoSuspeito.end_seconds}
                  fill="#dc2626"
                  fillOpacity={0.16}
                  stroke="#dc2626"
                  strokeOpacity={0.45}
                />
              )}
              {seriesVisiveis.map((serie, indice) => (
                <Line
                  key={serie.canal}
                  type="linear"
                  dataKey={`canal_${indice}`}
                  name={serie.canal}
                  stroke="#334155"
                  strokeWidth={0.85}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
              <Brush
                dataKey="tempo"
                height={24}
                travellerWidth={8}
                stroke="#0d9488"
                tickFormatter={(value) => `${Number(value).toFixed(0)}s`}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {mapaShapUrl && (
        <div className="flex min-h-0 flex-1 flex-col border-t border-clinical-200">
          <div className="flex shrink-0 items-center justify-between border-b border-clinical-100 bg-clinical-50 px-5 py-3">
            <div>
              <h3 className="text-sm font-semibold text-clinical-900">
                Mapa de Explicabilidade (SHAP)
              </h3>
              <p className="text-xs text-clinical-500">
                Features que mais influenciaram a decisao da IA
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
                Mapa SHAP - Visualizacao ampliada
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
