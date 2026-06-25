import type {
  AnaliseIAResponse,
  DiagnosticoConcluido,
  DiagnosticoEmProcessamento,
  DiagnosticoResponse,
  ExameUploadResponse,
  LaudoExameResponse,
  SinaisExameResponse,
} from '../types/api'
import { apiClient } from './client'

export async function uploadExame(
  arquivo: File,
  pacienteId: number,
  taxaAmostragem?: number,
): Promise<ExameUploadResponse> {
  const formData = new FormData()
  formData.append('arquivo', arquivo)
  formData.append('paciente_id', String(pacienteId))
  if (taxaAmostragem != null) {
    formData.append('taxa_amostragem', String(taxaAmostragem))
  }

  const { data } = await apiClient.post<ExameUploadResponse>(
    '/api/exames/upload',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
    },
  )

  return data
}

export async function obterDiagnostico(
  exameId: number,
): Promise<DiagnosticoResponse> {
  const response = await apiClient.get<DiagnosticoResponse>(
    `/api/exames/${exameId}/diagnostico`,
    {
      validateStatus: (status) => status === 200 || status === 206,
    },
  )

  if (response.status === 206) {
    return response.data as DiagnosticoEmProcessamento
  }

  return response.data as DiagnosticoConcluido
}

export async function obterSinaisExame(
  exameId: number,
): Promise<SinaisExameResponse> {
  const { data } = await apiClient.get<SinaisExameResponse>(
    `/api/exames/${exameId}/sinais`,
  )
  return data
}

export async function solicitarAnaliseIA(
  exameId: number,
  canaisSelecionados: string[],
): Promise<AnaliseIAResponse> {
  const { data } = await apiClient.post<AnaliseIAResponse>(
    `/api/exames/${exameId}/analise`,
    { canais_selecionados: canaisSelecionados },
  )
  return data
}

export async function salvarLaudo(
  exameId: number,
  laudoTexto: string,
): Promise<LaudoExameResponse> {
  const { data } = await apiClient.patch<LaudoExameResponse>(
    `/api/exames/${exameId}/laudo`,
    { laudo_texto: laudoTexto },
  )
  return data
}
