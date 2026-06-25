import { type FormEvent, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { RecuperarSenhaModal } from '../components/RecuperarSenhaModal'
import { useAuth } from '../contexts/AuthContext'
import { useToast } from '../contexts/ToastContext'

function Spinner() {
  return (
    <svg
      className="h-4 w-4 animate-spin-slow"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      aria-hidden
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2.5}
        d="M12 3v3m6.364 1.636l-2.121 2.121M21 12h-3m-1.636 6.364l-2.121-2.121M12 21v-3m-6.364-1.636l2.121-2.121M3 12h3m1.636-6.364l2.121 2.121"
      />
    </svg>
  )
}

export function Login() {
  const { login, isAuthenticated, isLoading } = useAuth()
  const { sucesso } = useToast()
  const navigate = useNavigate()

  const [email, setEmail] = useState('')
  const [senha, setSenha] = useState('')
  const [erro, setErro] = useState<string | null>(null)
  const [entrando, setEntrando] = useState(false)
  const [agitando, setAgitando] = useState(false)
  const [modalRecuperar, setModalRecuperar] = useState(false)

  if (!isLoading && isAuthenticated) {
    return <Navigate to="/" replace />
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setErro(null)
    setEntrando(true)

    try {
      await login(email, senha)
      sucesso('Bem-vindo à plataforma EEG-XAI!')
      navigate('/', { replace: true })
    } catch {
      setErro('E-mail ou senha inválidos. Verifique suas credenciais.')
      setAgitando(true)
      setTimeout(() => setAgitando(false), 500)
    } finally {
      setEntrando(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 px-4">

      {/* Fundo decorativo */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-32 -top-32 h-96 w-96 rounded-full bg-accent/10 blur-3xl" />
        <div className="absolute -bottom-32 -right-32 h-96 w-96 rounded-full bg-accent/10 blur-3xl" />
      </div>

      <div className={[
        'relative w-full max-w-md animate-slide-up',
        agitando ? 'animate-shake' : '',
      ].join(' ')}>

        {/* Logo */}
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-accent text-xl font-bold text-white shadow-lg ring-4 ring-accent/20">
            EEG
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white">EEG-XAI</h1>
          <p className="mt-2 text-sm text-slate-400">
            Plataforma clínica de diagnóstico neurológico com IA explicável
          </p>
        </div>

        {/* Card */}
        <form
          onSubmit={(e) => void handleSubmit(e)}
          className="rounded-2xl border border-white/10 bg-white/5 p-8 shadow-2xl backdrop-blur-sm"
        >
          <h2 className="text-lg font-semibold text-white">Acesso Médico</h2>
          <p className="mt-1 text-sm text-slate-400">
            Entre com suas credenciais institucionais.
          </p>

          <div className="mt-6 space-y-4">
            <label className="block text-sm">
              <span className="font-medium text-slate-300">E-mail</span>
              <input
                required
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 text-white outline-none placeholder:text-slate-500 transition focus:border-accent focus:bg-white/10 focus:ring-2 focus:ring-accent/30"
                placeholder="ana.silva@hospital.com"
              />
            </label>

            <label className="block text-sm">
              <span className="font-medium text-slate-300">Senha</span>
              <input
                required
                type="password"
                autoComplete="current-password"
                value={senha}
                onChange={(e) => setSenha(e.target.value)}
                className="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 text-white outline-none placeholder:text-slate-500 transition focus:border-accent focus:bg-white/10 focus:ring-2 focus:ring-accent/30"
                placeholder="••••••••"
              />
            </label>
          </div>

          {erro && (
            <div
              role="alert"
              className="mt-4 flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2.5 text-sm text-red-300 animate-fade-in"
            >
              <svg className="h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
              </svg>
              {erro}
            </div>
          )}

          <button
            type="submit"
            disabled={entrando}
            className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg bg-accent py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-accent-dark disabled:opacity-70"
          >
            {entrando && <Spinner />}
            {entrando ? 'Verificando credenciais...' : 'Entrar no Sistema'}
          </button>

          <button
            type="button"
            onClick={() => setModalRecuperar(true)}
            className="mt-4 w-full text-center text-sm text-slate-500 transition hover:text-accent"
          >
            Esqueci minha senha
          </button>
        </form>

        <p className="mt-5 text-center text-xs text-slate-600">
          Dev: ana.silva@hospital.com / senha123
        </p>
      </div>

      <RecuperarSenhaModal
        aberto={modalRecuperar}
        onFechar={() => setModalRecuperar(false)}
      />
    </div>
  )
}
