import { LoaderCircle } from 'lucide-react'

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
      <div className="mx-4 max-w-md rounded-md bg-white p-8 text-center shadow-2xl animate-modal-in">
        <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-accent/10">
          <LoaderCircle className="h-8 w-8 animate-spin text-accent" aria-hidden="true" />
        </div>
        <h2 className="text-lg font-bold text-clinical-900">Análise neurológica em andamento</h2>
        <p className="mt-2 text-sm leading-relaxed text-clinical-600">
          Processando <span className="font-semibold text-accent">{nCanais} {nCanais === 1 ? 'canal' : 'canais'}</span> do EEG. Aguarde a conclusão do processamento.
        </p>
        <p className="mt-4 text-xs text-clinical-400">Análise assíncrona no servidor</p>
        <div className="mt-5 h-1.5 overflow-hidden rounded-full bg-clinical-100">
          <div className="h-full w-2/3 animate-pulse-bar rounded-full bg-accent" />
        </div>
      </div>
    </div>
  )
}
