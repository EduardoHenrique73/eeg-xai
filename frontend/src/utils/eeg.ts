import type { DiagnosticoConcluido } from '../types/api'

export function isDiagnosticoConcluido(
  diagnostico: { status: string } | null,
): diagnostico is DiagnosticoConcluido {
  return diagnostico?.status === 'concluido'
}
