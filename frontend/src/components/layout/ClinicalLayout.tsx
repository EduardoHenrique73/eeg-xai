import type { ReactNode } from 'react'

interface ClinicalLayoutProps {
  left: ReactNode
  center: ReactNode
  right: ReactNode
}

export function ClinicalLayout({ left, center, right }: ClinicalLayoutProps) {
  return (
    <div className="flex h-full min-h-0 w-full flex-col bg-clinical-100">
      <header className="flex shrink-0 items-center justify-between border-b border-clinical-200 bg-white px-6 py-3 shadow-sm">
        <div>
          <h1 className="text-lg font-bold tracking-tight text-clinical-900">
            Visualizador EEG
          </h1>
          <p className="text-xs text-clinical-500">
            Análise de sinais com IA explicável (XAI)
          </p>
        </div>
        <span className="rounded-full bg-accent-light px-3 py-1 text-xs font-semibold text-accent-dark">
          Ambiente Clínico
        </span>
      </header>

      <main className="grid min-h-0 flex-1 grid-cols-1 gap-4 p-4 lg:grid-cols-[minmax(280px,22%)_1fr_minmax(300px,26%)]">
        <aside className="flex min-h-0 flex-col gap-4 overflow-y-auto">
          {left}
        </aside>

        <section className="min-h-0 overflow-y-auto">{center}</section>

        <aside className="min-h-0 overflow-y-auto">{right}</aside>
      </main>
    </div>
  )
}
