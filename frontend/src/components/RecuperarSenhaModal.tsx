import { type FormEvent, useState } from 'react'
import { recuperarSenha } from '../api/auth'
import { useToast } from '../contexts/ToastContext'

interface RecuperarSenhaModalProps {
  aberto: boolean
  onFechar: () => void
}

export function RecuperarSenhaModal({ aberto, onFechar }: RecuperarSenhaModalProps) {
  const { sucesso, erro: toastErro } = useToast()
  const [email, setEmail] = useState('')
  const [enviando, setEnviando] = useState(false)
  const [enviado, setEnviado] = useState(false)

  if (!aberto) return null

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setEnviando(true)

    try {
      await recuperarSenha(email)
      setEnviado(true)
      sucesso('Instruções enviadas! Verifique seu e-mail.')
      setTimeout(() => {
        handleFechar()
      }, 1800)
    } catch {
      toastErro('Não foi possível processar a solicitação. Tente novamente.')
    } finally {
      setEnviando(false)
    }
  }

  const handleFechar = () => {
    setEmail('')
    setEnviado(false)
    onFechar()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-backdrop-in">
      {/* Overlay */}
      <div
        className="absolute inset-0 bg-slate-900/70 backdrop-blur-sm"
        onClick={handleFechar}
        aria-hidden
      />

      {/* Card */}
      <div
        className="relative w-full max-w-md rounded-2xl bg-white p-7 shadow-2xl animate-modal-in"
        role="dialog"
        aria-modal="true"
        aria-labelledby="recuperar-senha-titulo"
      >
        <button
          type="button"
          onClick={handleFechar}
          className="absolute right-4 top-4 rounded-lg p-1.5 text-clinical-400 transition hover:bg-clinical-100 hover:text-clinical-700"
          aria-label="Fechar"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <div className="mb-5 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent/10">
            <svg className="h-5 w-5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
            </svg>
          </div>
          <div>
            <h2 id="recuperar-senha-titulo" className="text-lg font-semibold text-clinical-900">
              Recuperar Senha
            </h2>
            <p className="text-xs text-clinical-500">
              Enviaremos instruções de redefinição por e-mail.
            </p>
          </div>
        </div>

        {enviado ? (
          <div className="flex flex-col items-center gap-3 py-4 animate-fade-in">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-50">
              <svg className="h-6 w-6 text-alert-normal" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <p className="text-sm font-medium text-clinical-700">Instruções enviadas!</p>
            <p className="text-xs text-clinical-500">Verifique sua caixa de entrada.</p>
          </div>
        ) : (
          <form onSubmit={(e) => void handleSubmit(e)} className="space-y-4">
            <label className="block text-sm">
              <span className="font-medium text-clinical-700">E-mail corporativo</span>
              <input
                required
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="mt-1 w-full rounded-lg border border-clinical-200 px-3 py-2.5 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                placeholder="seu.email@hospital.com"
              />
            </label>

            <div className="flex justify-end gap-3 pt-1">
              <button
                type="button"
                onClick={handleFechar}
                className="rounded-lg border border-clinical-300 px-4 py-2 text-sm font-medium text-clinical-700 transition hover:bg-clinical-50"
              >
                Cancelar
              </button>
              <button
                type="submit"
                disabled={enviando}
                className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-white transition hover:bg-accent-dark disabled:opacity-60"
              >
                {enviando && (
                  <svg className="h-3.5 w-3.5 animate-spin-slow" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 3v3m6.364 1.636l-2.121 2.121M21 12h-3m-1.636 6.364l-2.121-2.121M12 21v-3m-6.364-1.636l2.121 2.121M3 12h3m1.636-6.364l2.121 2.121" />
                  </svg>
                )}
                {enviando ? 'Enviando...' : 'Enviar Instruções'}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  )
}
