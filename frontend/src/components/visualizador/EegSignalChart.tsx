import { memo, useEffect, useMemo, useState } from 'react'
import { Eye, EyeOff, Maximize2, Minus, Plus, RotateCcw, X } from 'lucide-react'
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
import type { ShapOverlay, SinaisExameResponse, TrechoSuspeito } from '../../types/api'

interface EegSignalChartProps {
  exameId: number | null
  mapaShapUrl?: string | null
  shapOverlay?: ShapOverlay | null
  canaisSelecionados?: string[]
  trechoSuspeito?: TrechoSuspeito | null
  topTrechosSuspeitos?: TrechoSuspeito[]
  placeholder?: string
}

export const EegSignalChart = memo(function EegSignalChart({
  exameId,
  mapaShapUrl,
  shapOverlay,
  canaisSelecionados = [],
  trechoSuspeito,
  topTrechosSuspeitos = [],
  placeholder = 'Aguardando importacao do exame...',
}: EegSignalChartProps) {
  const [sinais, setSinais] = useState<SinaisExameResponse | undefined>(undefined)
  const [carregando, setCarregando] = useState(false)
  const [erro, setErro] = useState<string | null>(null)
  const [shapExpandido, setShapExpandido] = useState(false)
  const [shapVisivel, setShapVisivel] = useState(true)
  const [detalharShap, setDetalharShap] = useState(false)
  const [escalaVertical, setEscalaVertical] = useState(1)
  const intervaloDetalhe = useMemo(() => {
    if (!detalharShap || !shapOverlay?.cells.length) return undefined
    const inicio = Math.min(...shapOverlay.cells.map((celula) => celula.start_seconds))
    const fim = Math.max(...shapOverlay.cells.map((celula) => celula.end_seconds))
    return { start_seconds: Math.max(0, inicio - 8), end_seconds: fim + 8 }
  }, [detalharShap, shapOverlay])
  const detalheInicio = intervaloDetalhe?.start_seconds
  const detalheFim = intervaloDetalhe?.end_seconds

  useEffect(() => {
    if (exameId == null) {
      setSinais(undefined)
      setCarregando(false)
      setErro(null)
      setEscalaVertical(1)
      setDetalharShap(false)
      return
    }

    let cancelado = false
    setCarregando(true)
    setErro(null)
    setSinais(undefined)
    setEscalaVertical(1)

    obterSinaisExame(
      exameId,
      detalheInicio == null || detalheFim == null
        ? undefined
        : { start_seconds: detalheInicio, end_seconds: detalheFim },
    )
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
  }, [exameId, detalheInicio, detalheFim])

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

  const dadosExibidos = useMemo(() => {
    if (detalharShap || dadosGrafico.length <= 600) return dadosGrafico
    const ultimo = dadosGrafico.length - 1
    return Array.from({ length: 600 }, (_, indice) =>
      dadosGrafico[Math.round((indice * ultimo) / 599)],
    )
  }, [dadosGrafico, detalharShap])

  const possuiSerie = dadosExibidos.length > 0
  const dominioY: [number, number] = [-2, Math.max(2, (seriesVisiveis.length - 1) * 4 + 2)]
  const ticksY = seriesVisiveis.map((_, indice) => (seriesVisiveis.length - indice - 1) * 4)
  const indicesCanais = new Map(seriesVisiveis.map((serie, indice) => [serie.canal, indice]))
  const celulasShap = (shapVisivel ? shapOverlay?.cells : undefined)?.filter((celula) =>
    indicesCanais.has(celula.canal) && celula.intensity > 0,
  ) ?? []
  const focoShap = useMemo(() => {
    if (!shapVisivel || !shapOverlay?.cells.length || dadosExibidos.length < 2) return null
    const inicio = Math.min(...shapOverlay.cells.map((celula) => celula.start_seconds))
    const fim = Math.max(...shapOverlay.cells.map((celula) => celula.end_seconds))
    const margem = Math.max(4, fim - inicio)
    const primeiro = dadosExibidos.findIndex((ponto) => ponto.tempo >= inicio - margem)
    const ultimo = dadosExibidos.findIndex((ponto) => ponto.tempo >= fim + margem)
    return {
      startIndex: Math.max(0, primeiro),
      endIndex: ultimo < 0 ? dadosExibidos.length - 1 : ultimo,
    }
  }, [shapVisivel, shapOverlay, dadosExibidos])

  const mensagem = carregando
    ? 'Carregando ondas cerebrais...'
    : erro ?? placeholder

  return (
    <section className="flex h-full min-h-[530px] flex-col overflow-hidden rounded-md border border-clinical-200 bg-white shadow-clinical">
      <header className="shrink-0 border-b border-clinical-200 bg-white px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-clinical-900">Traçado EEG</h2>
            <p className="mt-0.5 text-xs text-clinical-500">
              {possuiSerie ? `${seriesVisiveis.length} ${seriesVisiveis.length === 1 ? 'canal' : 'canais'} · ${detalharShap ? 'recorte detalhado' : 'visão geral amostrada'} · tempo em segundos` : carregando ? 'Carregando sinal' : 'Aguardando sinal'}
            </p>
          </div>
          {possuiSerie && (
            <div className="flex flex-wrap items-center gap-2">
              {shapOverlay?.cells.length ? (
                <div className="flex overflow-hidden rounded-md border border-clinical-200" role="group" aria-label="Área do traçado">
                  <button type="button" onClick={() => setDetalharShap(false)} className={`px-2.5 py-1.5 text-xs font-medium ${!detalharShap ? 'bg-accent-light text-accent-dark' : 'text-clinical-700 hover:bg-clinical-50'}`}>Exame</button>
                  <button type="button" onClick={() => setDetalharShap(true)} className={`border-l border-clinical-200 px-2.5 py-1.5 text-xs font-medium ${detalharShap ? 'bg-accent-light text-accent-dark' : 'text-clinical-700 hover:bg-clinical-50'}`}>Trecho SHAP</button>
                </div>
              ) : null}
              {shapOverlay?.cells.length ? (
                <button
                  type="button"
                  onClick={() => setShapVisivel((visivel) => !visivel)}
                  className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium ${shapVisivel ? 'border-red-200 bg-red-50 text-red-700' : 'border-clinical-200 text-clinical-700'}`}
                  aria-pressed={shapVisivel}
                  title={shapVisivel ? 'Ocultar sobreposição SHAP' : 'Exibir sobreposição SHAP'}
                >
                  {shapVisivel ? <Eye size={14} aria-hidden="true" /> : <EyeOff size={14} aria-hidden="true" />}
                  SHAP
                </button>
              ) : null}
              {mapaShapUrl && (
                <button type="button" onClick={() => setShapExpandido(true)} className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-clinical-200 text-clinical-700 hover:bg-clinical-50" title="Abrir mapa SHAP" aria-label="Abrir mapa SHAP">
                  <Maximize2 size={15} aria-hidden="true" />
                </button>
              )}
              <div className="flex items-center gap-1 border-l border-clinical-200 pl-2">
                <button type="button" onClick={() => setEscalaVertical((v) => Math.max(0.5, Number((v - 0.5).toFixed(1))))} className="flex h-8 w-8 items-center justify-center rounded-md text-clinical-700 hover:bg-clinical-50" aria-label="Reduzir escala vertical" title="Reduzir escala vertical"><Minus size={15} aria-hidden="true" /></button>
                <span className="w-10 text-center text-xs tabular-nums text-clinical-700">{escalaVertical.toFixed(1)}×</span>
                <button type="button" onClick={() => setEscalaVertical((v) => Math.min(5, Number((v + 0.5).toFixed(1))))} className="flex h-8 w-8 items-center justify-center rounded-md text-clinical-700 hover:bg-clinical-50" aria-label="Aumentar escala vertical" title="Aumentar escala vertical"><Plus size={15} aria-hidden="true" /></button>
                <button type="button" onClick={() => setEscalaVertical(1)} className="flex h-8 w-8 items-center justify-center rounded-md text-clinical-700 hover:bg-clinical-50" aria-label="Restaurar escala vertical" title="Restaurar escala vertical"><RotateCcw size={14} aria-hidden="true" /></button>
              </div>
            </div>
          )}
        </div>
        {possuiSerie && (
          <div className="mt-2 text-xs text-clinical-500">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
              {trechoSuspeito && <span><span className="mr-1 inline-block h-2 w-2 rounded-sm bg-amber-500" />Trecho suspeito</span>}
              {celulasShap.length > 0 && <span><span className="mr-1 inline-block h-2 w-2 rounded-sm bg-red-600" />SHAP positivo · sequência de pico</span>}
            </div>
            {celulasShap.length > 0 && <p className="mt-1">Atribuição por janela de features e canal; não por amostra bruta.</p>}
          </div>
        )}
      </header>

      <div className="min-h-[400px] flex-1 p-3 md:p-4">
        {!possuiSerie ? (
          <div className="flex h-full min-h-[380px] items-center justify-center bg-clinical-50">
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
            <LineChart data={dadosExibidos} margin={{ top: 8, right: 16, left: 24, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e4ebf3" />
              <XAxis
                dataKey="tempo"
                type="number"
                domain={['dataMin', 'dataMax']}
                tick={{ fontSize: 11, fill: '#61758b' }}
                label={{
                  value: 'Tempo (s)',
                  position: 'insideBottom',
                  offset: -4,
                  fill: '#61758b',
                  fontSize: 11,
                }}
              />
              <YAxis
                domain={dominioY}
                ticks={ticksY}
                interval={0}
                tick={{ fontSize: 11, fill: '#61758b' }}
                tickFormatter={(value) => {
                  const indice = ticksY.indexOf(Number(value))
                  return indice >= 0 ? seriesVisiveis[indice].canal : ''
                }}
                width={82}
              />
              {seriesVisiveis.length <= 8 && <Tooltip
                content={({ label }) => label == null ? null : (
                  <div className="rounded-md border border-clinical-200 bg-white px-2.5 py-1.5 text-xs text-clinical-800 shadow-clinical">
                    {Number(label).toFixed(1)} s
                  </div>
                )}
                wrapperStyle={{ pointerEvents: 'none' }}
              />}
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
                  fill="#f59e0b"
                  fillOpacity={0.16}
                  stroke="#d97706"
                  strokeOpacity={0.45}
                />
              )}
              {celulasShap.map((celula, indice) => {
                const canalIndice = indicesCanais.get(celula.canal)!
                const centro = (seriesVisiveis.length - canalIndice - 1) * 4
                return (
                  <ReferenceArea
                    key={`${celula.canal}-${celula.start_seconds}-${indice}`}
                    x1={celula.start_seconds}
                    x2={celula.end_seconds}
                    y1={centro - 1.9}
                    y2={centro + 1.9}
                    fill="#dc2626"
                    fillOpacity={0.08 + 0.47 * celula.intensity}
                    strokeOpacity={0}
                    ifOverflow="hidden"
                  />
                )
              })}
              {seriesVisiveis.map((serie, indice) => (
                <Line
                  key={serie.canal}
                  type="linear"
                  dataKey={`canal_${indice}`}
                  name={serie.canal}
                  stroke="#28496c"
                  strokeWidth={0.85}
                  dot={false}
                  activeDot={false}
                  isAnimationActive={false}
                />
              ))}
              <Brush
                key={`${exameId}-${shapVisivel}-${detalharShap}-${shapOverlay?.cells[0]?.start_seconds ?? 'all'}`}
                dataKey="tempo"
                height={24}
                travellerWidth={8}
                stroke="#1d5fc1"
                tickFormatter={(value) => `${Number(value).toFixed(0)}s`}
                startIndex={focoShap?.startIndex ?? 0}
                endIndex={focoShap?.endIndex ?? dadosExibidos.length - 1}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

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
          <div className="relative flex max-h-[95vh] max-w-5xl flex-col rounded-md bg-white shadow-2xl animate-modal-in">
            <div className="flex items-center justify-between border-b border-clinical-100 px-5 py-3">
              <h3 className="text-sm font-semibold text-clinical-900">
                Mapa SHAP · sequência de pico
              </h3>
              <button
                type="button"
                onClick={() => setShapExpandido(false)}
                className="rounded-lg p-1.5 text-clinical-500 transition hover:bg-clinical-100"
                aria-label="Fechar"
              >
                <X size={20} aria-hidden="true" />
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
})
