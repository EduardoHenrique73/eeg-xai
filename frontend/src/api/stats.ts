import { apiClient } from './client'

export interface DashboardStats {
  total_pacientes: number
  exames_pendentes: number
  laudos_emitidos: number
}

export async function obterStats(): Promise<DashboardStats> {
  const { data } = await apiClient.get<DashboardStats>('/api/stats')
  return data
}
