import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from 'react'

type ToastTipo = 'sucesso' | 'erro' | 'info'

interface ToastItem {
  id: string
  tipo: ToastTipo
  mensagem: string
}

interface ToastContextValue {
  toasts: ToastItem[]
  sucesso: (mensagem: string) => void
  erro: (mensagem: string) => void
  info: (mensagem: string) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

const DURACAO_MS = 4500

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([])
  const idRef = useRef(0)

  const adicionar = useCallback((tipo: ToastTipo, mensagem: string) => {
    const id = String(++idRef.current)
    setToasts((prev) => [...prev, { id, tipo, mensagem }])
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, DURACAO_MS)
  }, [])

  const sucesso = useCallback((m: string) => adicionar('sucesso', m), [adicionar])
  const erro = useCallback((m: string) => adicionar('erro', m), [adicionar])
  const info = useCallback((m: string) => adicionar('info', m), [adicionar])

  return (
    <ToastContext.Provider value={{ toasts, sucesso, erro, info }}>
      {children}
    </ToastContext.Provider>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast deve ser usado dentro de <ToastProvider>')
  return ctx
}
