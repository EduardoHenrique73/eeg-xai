import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

const linkClass = ({ isActive }: { isActive: boolean }) =>
  [
    'block rounded-lg px-4 py-3 text-sm font-medium transition-colors',
    isActive
      ? 'bg-white/10 text-white'
      : 'text-slate-300 hover:bg-white/5 hover:text-white',
  ].join(' ')

export function Layout() {
  const { medico, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <div className="flex h-screen w-full bg-clinical-100">
      <aside className="flex w-64 shrink-0 flex-col bg-slate-900 text-white">
        <div className="border-b border-slate-700 px-5 py-6">
          <h1 className="text-lg font-bold tracking-tight">EEG-XAI</h1>
          <p className="mt-1 text-xs text-slate-400">Plataforma Clínica</p>
        </div>

        <nav className="flex flex-1 flex-col gap-1 p-4">
          <NavLink to="/" end className={linkClass}>
            Dashboard
          </NavLink>
          <NavLink to="/pacientes" className={linkClass}>
            Pacientes
          </NavLink>
        </nav>

        <div className="border-t border-slate-700 px-5 py-4">
          {medico && (
            <p className="mb-3 text-xs text-slate-300">
              <span className="block font-semibold text-white">{medico.nome}</span>
              CRM {medico.crm}
            </p>
          )}
          <button
            type="button"
            onClick={handleLogout}
            className="w-full rounded-lg border border-slate-600 px-3 py-2 text-xs font-medium text-slate-200 hover:bg-slate-800"
          >
            Sair
          </button>
        </div>
      </aside>

      <main className="min-w-0 flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}