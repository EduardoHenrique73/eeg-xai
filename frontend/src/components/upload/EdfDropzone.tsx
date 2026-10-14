import { useCallback, useRef, useState, type DragEvent } from 'react'
import { FileUp } from 'lucide-react'

interface EdfDropzoneProps {
  onFileSelected: (file: File) => void
  selectedFile?: File | null
  disabled?: boolean
  compacto?: boolean
}

function isEdfFile(file: File): boolean {
  return file.name.toLowerCase().endsWith('.edf')
}

export function EdfDropzone({
  onFileSelected,
  selectedFile,
  disabled = false,
  compacto = false,
}: EdfDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [erro, setErro] = useState<string | null>(null)

  const processarArquivo = useCallback(
    (file: File | undefined) => {
      if (!file) return

      if (!isEdfFile(file)) {
        setErro('Apenas arquivos com extensão .edf são aceitos.')
        return
      }

      setErro(null)
      onFileSelected(file)
    },
    [onFileSelected],
  )

  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault()
      setIsDragging(false)
      if (disabled) return
      processarArquivo(event.dataTransfer.files[0])
    },
    [disabled, processarArquivo],
  )

  return (
    <section className={`min-w-0 rounded-md border border-clinical-200 bg-white shadow-clinical xl:flex-1 2xl:flex-none ${compacto ? 'p-3' : 'p-4'}`}>
      {!compacto && (
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-clinical-500">
          Upload de Exame EEG
        </h3>
      )}

      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault()
            if (!disabled) inputRef.current?.click()
          }
        }}
        onDragOver={(event) => {
          event.preventDefault()
          if (!disabled) setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={[
          compacto
            ? 'flex cursor-pointer items-center gap-3 rounded-md border border-dashed px-3 py-2 text-left transition-colors'
            : 'flex min-h-40 cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed px-4 py-6 text-center transition-colors xl:min-h-28 xl:py-3 2xl:min-h-40 2xl:py-6',
          disabled
            ? 'cursor-not-allowed border-clinical-200 bg-clinical-50 opacity-60'
            : isDragging
              ? 'border-accent bg-accent-light'
              : 'border-clinical-300 bg-clinical-50 hover:border-accent hover:bg-white',
        ].join(' ')}
      >
        <FileUp size={compacto ? 20 : 30} strokeWidth={1.6} className={compacto ? 'shrink-0 text-accent' : 'mb-3 text-accent xl:mb-1 2xl:mb-3'} aria-hidden="true" />

        <div className="min-w-0">
          <p className="text-sm font-medium text-clinical-800">
            {compacto ? 'Carregar outro exame EDF' : 'Arraste o arquivo .edf ou clique para selecionar'}
          </p>
          <p className="mt-0.5 text-xs text-clinical-500">
            {compacto ? 'Substitui o exame exibido na tela' : 'Formato aceito: European Data Format (.edf)'}
          </p>
        </div>

        {selectedFile && (
          <p className={`${compacto ? 'ml-auto truncate' : 'mt-3'} rounded-md bg-white px-3 py-1 font-mono text-xs text-clinical-700 shadow-sm`}>
            {selectedFile.name}
          </p>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".edf,application/octet-stream"
        className="hidden"
        disabled={disabled}
        onChange={(event) => processarArquivo(event.target.files?.[0])}
      />

      {erro && (
        <p className="mt-3 text-sm font-medium text-alert-crisis" role="alert">
          {erro}
        </p>
      )}
    </section>
  )
}
