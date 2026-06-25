export interface MedicoAuth {
  id: number
  nome: string
  email: string
  crm: string
}

export interface LoginRequest {
  email: string
  senha: string
}

export interface LoginResponse {
  access_token: string
  token_type: string
  medico: MedicoAuth
}

export interface RecuperarSenhaResponse {
  message: string
}
