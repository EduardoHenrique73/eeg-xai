import { Activity, LayoutDashboard, LogOut, Settings2, UsersRound } from 'lucide-react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

const navItems = [
  { to: '/', label: 'Visão geral', mobileLabel: 'Início', icon: LayoutDashboard, end: true },
  { to: '/pacientes', label: 'Pacientes', mobileLabel: 'Pacientes', icon: UsersRound, end: false },
  { to: '/configuracoes', label: 'Configurações', mobileLabel: 'Ajustes', icon: Settings2, end: false },
]

export function Layout() {
  const { medico, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <div className="flex min-h-screen w-full flex-col bg-clinical-50 md:h-screen md:flex-row">
      <aside className="relative flex w-full shrink-0 flex-col border-b border-clinical-200 bg-white md:w-60 md:border-b-0 md:border-r">
        <div className="flex items-center gap-3 border-b border-clinical-100 px-5 py-4 md:h-20">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-accent text-white">
            <Activity size={22} strokeWidth={2.2} aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <p className="text-base font-bold text-clinical-900">EEG-XAI</p>
            <p className="text-xs text-clinical-500">Ambiente clínico</p>
          </div>
        </div>

        <nav aria-label="Navegação principal" className="grid grid-cols-3 gap-1 px-3 py-2 md:flex md:flex-1 md:flex-col md:gap-1 md:py-5">
          {navItems.map(({ to, label, mobileLabel, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => [
                'flex min-w-0 items-center justify-center gap-1.5 rounded-md px-1 py-2.5 text-xs font-medium transition-colors md:justify-start md:gap-3 md:px-3 md:text-sm',
                isActive
                  ? 'bg-accent-light text-accent-dark'
                  : 'text-clinical-700 hover:bg-clinical-50 hover:text-clinical-900',
              ].join(' ')}
            >
              <Icon size={18} strokeWidth={1.9} className="shrink-0" aria-hidden="true" />
              <span className="md:hidden">{mobileLabel}</span>
              <span className="hidden md:inline">{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="hidden border-t border-clinical-100 px-4 py-4 md:block">
          {medico && (
            <div className="mb-3 min-w-0 px-2">
              <p className="truncate text-sm font-semibold text-clinical-900">{medico.nome}</p>
              <p className="text-xs text-clinical-500">CRM {medico.crm}</p>
            </div>
          )}
          <button
            type="button"
            onClick={handleLogout}
            className="flex w-full items-center gap-3 rounded-md px-2 py-2 text-sm font-medium text-clinical-700 hover:bg-clinical-50 hover:text-clinical-900"
          >
            <LogOut size={18} aria-hidden="true" />
            Sair
          </button>
        </div>
        <button
          type="button"
          onClick={handleLogout}
          className="absolute right-4 top-5 text-clinical-700 md:hidden"
          aria-label="Sair"
          title="Sair"
        >
          <LogOut size={20} aria-hidden="true" />
        </button>
      </aside>

      <main className="min-h-0 min-w-0 flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
