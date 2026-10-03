import { useEffect, useState } from 'react'
import { ArrowRight, ClipboardCheck, Clock3, UsersRound } from 'lucide-react'
import { Link } from 'react-router-dom'
import { type DashboardStats, obterStats } from '../api/stats'
import { useAuth } from '../contexts/AuthContext'

export function Dashboard() {
  const { medico } = useAuth()
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [carregando, setCarregando] = useState(true)

  useEffect(() => {
    obterStats().then(setStats).catch(() => {}).finally(() => setCarregando(false))
  }, [])

  const indicadores = [
    { titulo: 'Pacientes', valor: stats?.total_pacientes, icon: UsersRound, color: 'text-accent' },
    { titulo: 'Exames pendentes', valor: stats?.exames_pendentes, icon: Clock3, color: 'text-amber-600' },
    { titulo: 'Laudos emitidos', valor: stats?.laudos_emitidos, icon: ClipboardCheck, color: 'text-emerald-700' },
  ]

  return (
    <div className="mx-auto max-w-6xl px-5 py-7 md:px-8">
      <header className="mb-7 border-b border-clinical-200 pb-5">
        <p className="text-xs font-semibold uppercase text-accent">EEG-XAI</p>
        <h1 className="mt-1 text-2xl font-semibold text-clinical-900">Visão geral</h1>
        <p className="mt-1 text-sm text-clinical-500">{medico?.nome ?? 'Equipe clínica'}</p>
      </header>

      <section aria-label="Indicadores operacionais" className="grid overflow-hidden rounded-md border border-clinical-200 bg-white sm:grid-cols-3">
        {indicadores.map(({ titulo, valor, icon: Icon, color }, indice) => (
          <div key={titulo} className={`px-5 py-5 ${indice > 0 ? 'border-t border-clinical-200 sm:border-t-0 sm:border-l' : ''}`}>
            <div className="flex items-center gap-2 text-sm font-medium text-clinical-500"><Icon size={17} className={color} aria-hidden="true" />{titulo}</div>
            <p className="mt-3 text-3xl font-semibold tabular-nums text-clinical-900">{carregando ? '…' : valor ?? '—'}</p>
          </div>
        ))}
      </section>

      <section className="mt-8" aria-labelledby="acoes-titulo">
        <h2 id="acoes-titulo" className="mb-3 text-sm font-semibold text-clinical-900">Acesso rápido</h2>
        <div className="divide-y divide-clinical-200 rounded-md border border-clinical-200 bg-white">
          <Link to="/pacientes" className="flex items-center justify-between gap-4 px-5 py-4 text-sm font-medium text-clinical-800 hover:bg-clinical-50">
            Abrir prontuários e exames
            <ArrowRight size={18} className="text-accent" aria-hidden="true" />
          </Link>
          <Link to="/configuracoes" className="flex items-center justify-between gap-4 px-5 py-4 text-sm font-medium text-clinical-800 hover:bg-clinical-50">
            Preferências de análise
            <ArrowRight size={18} className="text-accent" aria-hidden="true" />
          </Link>
        </div>
      </section>
    </div>
  )
}
