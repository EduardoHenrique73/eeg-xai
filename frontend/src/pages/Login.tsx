import { type FormEvent, useState } from 'react'
import { Activity, ArrowRight, LoaderCircle } from 'lucide-react'
import { Navigate, useNavigate } from 'react-router-dom'
import { RecuperarSenhaModal } from '../components/RecuperarSenhaModal'
import { useAuth } from '../contexts/AuthContext'
import { useToast } from '../contexts/ToastContext'

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

  if (!isLoading && isAuthenticated) return <Navigate to="/" replace />

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
    <div className="flex min-h-screen items-center justify-center bg-clinical-50 px-4 py-8">
      <div className={`w-full max-w-md ${agitando ? 'animate-shake' : ''}`}>
        <div className="mb-7 flex items-center gap-3">
          <span className="flex h-11 w-11 items-center justify-center rounded-md bg-accent text-white">
            <Activity size={24} aria-hidden="true" />
          </span>
          <div>
            <h1 className="text-xl font-bold text-clinical-900">EEG-XAI</h1>
            <p className="text-xs text-clinical-500">Ambiente clínico</p>
          </div>
        </div>

        <form onSubmit={(event) => void handleSubmit(event)} className="rounded-md border border-clinical-200 border-t-4 border-t-accent bg-white p-7 shadow-clinical sm:p-8">
          <h2 className="text-xl font-semibold text-clinical-900">Acesso médico</h2>
          <p className="mt-1 text-sm text-clinical-500">Entre com sua conta institucional.</p>

          <div className="mt-6 space-y-4">
            <label className="block text-sm">
              <span className="font-medium text-clinical-700">E-mail</span>
              <input required type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="seu.email@hospital.com" className="mt-1 w-full rounded-md border border-clinical-300 bg-white px-3 py-2.5 text-clinical-900 outline-none placeholder:text-clinical-500 transition focus:border-accent focus:ring-2 focus:ring-accent/20" />
            </label>
            <label className="block text-sm">
              <span className="font-medium text-clinical-700">Senha</span>
              <input required type="password" autoComplete="current-password" value={senha} onChange={(event) => setSenha(event.target.value)} className="mt-1 w-full rounded-md border border-clinical-300 bg-white px-3 py-2.5 text-clinical-900 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20" />
            </label>
          </div>

          {erro && <p role="alert" className="mt-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{erro}</p>}

          <button type="submit" disabled={entrando} className="mt-6 flex w-full items-center justify-center gap-2 rounded-md bg-accent py-3 text-sm font-semibold text-white transition hover:bg-accent-dark disabled:opacity-70">
            {entrando ? <LoaderCircle size={16} className="animate-spin" aria-hidden="true" /> : null}
            {entrando ? 'Verificando credenciais...' : 'Entrar no sistema'}
            {!entrando && <ArrowRight size={16} aria-hidden="true" />}
          </button>
          <button type="button" onClick={() => setModalRecuperar(true)} className="mt-4 w-full text-center text-sm text-clinical-500 hover:text-accent">Esqueci minha senha</button>
        </form>
      </div>

      <RecuperarSenhaModal aberto={modalRecuperar} onFechar={() => setModalRecuperar(false)} />
    </div>
  )
}
