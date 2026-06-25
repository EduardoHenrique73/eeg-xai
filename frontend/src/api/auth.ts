import type {
  LoginRequest,
  LoginResponse,
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
