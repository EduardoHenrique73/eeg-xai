import type { ReactNode } from 'react'

interface ClinicalLayoutProps {
  left: ReactNode
  center: ReactNode
  right: ReactNode
}

export function ClinicalLayout({ left, center, right }: ClinicalLayoutProps) {
  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_312px] xl:gap-0 xl:p-0 2xl:grid-cols-[250px_minmax(0,1fr)_340px]">
      <aside className="flex min-w-0 flex-col gap-4 xl:col-span-2 xl:flex-row xl:p-4 2xl:col-span-1 2xl:flex-col 2xl:overflow-y-auto">
        {left}
      </aside>

      <section className="min-h-[530px] min-w-0 xl:min-h-0 xl:p-4">{center}</section>

      <aside className="min-w-0 xl:overflow-y-auto xl:p-4">
        {right}
      </aside>
    </div>
  )
}
