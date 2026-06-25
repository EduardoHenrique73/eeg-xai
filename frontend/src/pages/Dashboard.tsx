import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { type DashboardStats, obterStats } from '../api/stats'
import { useAuth } from '../contexts/AuthContext'

interface IndicadorCardProps {
  titulo: string
  valor: string | number
  descricao: string
  carregando?: boolean
  cor?: 'default' | 'accent' | 'success' | 'warning'
  icone: React.ReactNode
}

const COR_MAP = {
  default: 'from-clinical-50 to-white border-clinical-200',
  accent: 'from-accent/5 to-white border-accent/30',
  success: 'from-green-50 to-white border-green-200',
  warning: 'from-orange-50 to-white border-orange-200',
}

function IndicadorCard({
  titulo,
  valor,
  descricao,
  carregando = false,
  cor = 'default',
  icone,
}: IndicadorCardProps) {
  return (
    <article
      className={[
        'relative overflow-hidden rounded-xl border bg-gradient-to-br p-5 shadow-clinical animate-slide-up',
        COR_MAP[cor],
      ].join(' ')}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-clinical-500">{titulo}</p>
        <span className="shrink-0 rounded-lg bg-white p-1.5 shadow-clinical">{icone}</span>
      </div>

      {carregando ? (
        <div className="mt-3 h-9 w-16 animate-pulse rounded-lg bg-clinical-200" />
      ) : (
        <p className="mt-3 text-4xl font-bold tabular-nums text-clinical-900">{valor}</p>
      )}

      <p className="mt-1.5 text-xs text-clinical-500">{descricao}</p>
    </article>
  )
}

export function Dashboard() {
  const { medico } = useAuth()
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [carregando, setCarregando] = useState(true)

  useEffect(() => {
    obterStats()
      .then(setStats)
      .catch(() => {})
      .finally(() => setCarregando(false))
  }, [])

  return (
    <div className="p-8">
      <header className="mb-8">
        <p className="text-sm text-clinical-500">Bem-vindo de volta,</p>
        <h2 className="text-2xl font-bold text-clinical-900">
          {medico?.nome ?? 'Plataforma EEG-XAI'}
        </h2>
        <p className="mt-1 text-sm text-clinical-500">
          Visão geral do sistema de diagnóstico neurológico com IA explicável.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <IndicadorCard
          titulo="Total de Pacientes"
          valor={stats?.total_pacientes ?? '—'}
          descricao="Prontuários cadastrados na base"
          carregando={carregando}
          cor="accent"
          icone={
            <svg className="h-4 w-4 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          }
        />
        <IndicadorCard
          titulo="Exames em Análise"
          valor={stats?.exames_pendentes ?? '—'}
          descricao="Aguardando processamento CNN-LSTM"
          carregando={carregando}
          cor="warning"
          icone={
            <svg className="h-4 w-4 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
        />
        <IndicadorCard
          titulo="Laudos Emitidos"
          valor={stats?.laudos_emitidos ?? '—'}
          descricao="Pareceres médicos finalizados"
          carregando={carregando}
          cor="success"
          icone={
            <svg className="h-4 w-4 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
        />
        <IndicadorCard
          titulo="Acurácia Média"
          valor="99,2%"
          descricao="CNN-LSTM — Moreira et al. (2025)"
          cor="default"
          icone={
            <svg className="h-4 w-4 text-clinical-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
        />
      </div>

      {/* Atalhos rápidos */}
      <div className="mt-8">
        <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-clinical-500">
          Acesso Rápido
        </h3>
        <div className="grid gap-4 sm:grid-cols-2">
          <Link
            to="/pacientes"
            className="group flex items-center gap-4 rounded-xl border border-clinical-200 bg-white p-5 shadow-clinical transition hover:border-accent/40 hover:shadow-md"
          >
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent/10 transition group-hover:bg-accent/20">
              <svg className="h-5 w-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
              </svg>
            </div>
            <div className="min-w-0">
              <p className="font-semibold text-clinical-900">Novo Paciente</p>
              <p className="text-sm text-clinical-500">Cadastrar e iniciar exame EEG</p>
            </div>
            <svg className="ml-auto h-4 w-4 text-clinical-300 transition group-hover:text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </Link>

          <Link
            to="/pacientes"
            className="group flex items-center gap-4 rounded-xl border border-clinical-200 bg-white p-5 shadow-clinical transition hover:border-accent/40 hover:shadow-md"
          >
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-clinical-100 transition group-hover:bg-clinical-200">
              <svg className="h-5 w-5 text-clinical-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
            </div>
            <div className="min-w-0">
              <p className="font-semibold text-clinical-900">Prontuários</p>
              <p className="text-sm text-clinical-500">Ver e gerenciar lista de pacientes</p>
            </div>
            <svg className="ml-auto h-4 w-4 text-clinical-300 transition group-hover:text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </Link>
        </div>
      </div>
    </div>
  )
}
