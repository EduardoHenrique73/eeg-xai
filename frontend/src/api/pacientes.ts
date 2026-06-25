import type { Paciente, PacienteCreate } from '../types/api'
import { apiClient } from './client'

export async function listarPacientes(): Promise<Paciente[]> {
  const { data } = await apiClient.get<Paciente[]>('/api/pacientes')
  return data
}

export async function obterPaciente(pacienteId: number): Promise<Paciente> {
  const { data } = await apiClient.get<Paciente>(`/api/pacientes/${pacienteId}`)
  return data
}

export async function criarPaciente(payload: PacienteCreate): Promise<Paciente> {
  const { data } = await apiClient.post<Paciente>('/api/pacientes', payload)
  return data
}
