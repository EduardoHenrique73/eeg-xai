import type {
  LoginRequest,
  LoginResponse,
  MedicoAuth,
  MedicoConfigUpdate,
  RecuperarSenhaResponse,
} from '../types/auth'
import { apiClient } from './client'

export async function login(credenciais: LoginRequest): Promise<LoginResponse> {
  const { data } = await apiClient.post<LoginResponse>(
    '/api/auth/login',
    credenciais,
  )
  return data
}

export async function recuperarSenha(email: string): Promise<RecuperarSenhaResponse> {
  const { data } = await apiClient.post<RecuperarSenhaResponse>(
    '/api/auth/recuperar-senha',
    { email },
  )
  return data
}

export async function obterPerfilMedico(): Promise<MedicoAuth> {
  const { data } = await apiClient.get<MedicoAuth>('/api/auth/me')
  return data
}

export async function atualizarPerfilMedico(
  payload: MedicoConfigUpdate,
): Promise<MedicoAuth> {
  const { data } = await apiClient.patch<MedicoAuth>('/api/auth/me', payload)
  return data
}
