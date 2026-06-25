interface AnaliseLoadingOverlayProps {
  visivel: boolean
  nCanais: number
}

export function AnaliseLoadingOverlay({ visivel, nCanais }: AnaliseLoadingOverlayProps) {
  if (!visivel) return null

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-900/70 backdrop-blur-sm animate-backdrop-in"
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <div className="mx-4 max-w-md rounded-2xl bg-white p-8 text-center shadow-2xl animate-modal-in">
        <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-accent/10">
          <svg
            className="h-8 w-8 animate-spin-slow text-accent"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            aria-hidden
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 3v3m6.364 1.636l-2.121 2.121M21 12h-3m-1.636 6.364l-2.121-2.121M12 21v-3m-6.364-1.636l2.121 2.121M3 12h3m1.636-6.364l2.121 2.121"
            />
          </svg>
        </div>

        <h2 className="text-lg font-bold text-clinical-900">Análise neurológica em andamento</h2>
        <p className="mt-2 text-sm leading-relaxed text-clinical-600">
          Processando{' '}
          <span className="font-semibold text-accent">
            {nCanais} canal{nCanais !== 1 ? 'is' : ''}
          </span>{' '}
          neurológico{nCanais !== 1 ? 's' : ''}… Por favor, aguarde.
        </p>
        <p className="mt-4 text-xs text-clinical-400">
          CNN-LSTM + mapa SHAP · processamento assíncrono no servidor
        </p>

        <div className="mt-5 h-1.5 overflow-hidden rounded-full bg-clinical-100">
          <div className="h-full w-2/3 animate-pulse-bar rounded-full bg-accent" />
        </div>
      </div>
    </div>
  )
}
