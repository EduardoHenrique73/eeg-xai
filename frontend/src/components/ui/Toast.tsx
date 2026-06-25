import { useToast } from '../../contexts/ToastContext'

const CONFIG = {
  sucesso: {
    bg: 'bg-green-50 border-green-200',
    text: 'text-green-800',
    icon: (
      <svg className="h-5 w-5 shrink-0 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    ),
  },
  erro: {
    bg: 'bg-red-50 border-red-200',
    text: 'text-red-800',
    icon: (
      <svg className="h-5 w-5 shrink-0 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    ),
  },
  info: {
    bg: 'bg-blue-50 border-blue-200',
    text: 'text-blue-800',
    icon: (
      <svg className="h-5 w-5 shrink-0 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
}

export function ToastContainer() {
  const { toasts } = useToast()

  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed bottom-6 right-6 z-[9999] flex flex-col gap-3"
    >
      {toasts.map((toast) => {
        const c = CONFIG[toast.tipo]
        return (
          <div
            key={toast.id}
            role="status"
            className={[
              'pointer-events-auto flex max-w-sm items-start gap-3 rounded-xl border px-4 py-3',
              'shadow-lg animate-toast-in',
              c.bg,
            ].join(' ')}
          >
            {c.icon}
            <p className={`text-sm font-medium leading-snug ${c.text}`}>{toast.mensagem}</p>
          </div>
        )
      })}
    </div>
  )
}
