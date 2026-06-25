/** Tipos alinhados aos schemas Pydantic do backend FastAPI. */

export interface Paciente {
  id: number
  nome: string
  data_nascimento: string
  sexo: 'M' | 'F' | string
  cpf?: string | null
  telefone?: string | null
  observacoes?: string | null
  id_usuario: number
  created_at?: string
  updated_at?: string
}

export interface PacienteCreate {
  nome: string
  data_nascimento: string
  sexo: 'M' | 'F'
  cpf?: string | null
  telefone?: string | null
  observacoes?: string | null
}

export interface ExameUploadResponse {
  message: string
  exame_id: number
  arquivo_path: string
  status_exame: string
  laudo_texto?: string | null
}

export interface LaudoExameResponse {
  exame_id: number
  laudo_texto: string
  status_exame: string
  message: string
}

export interface DiagnosticoExameBase {
  exame_id: number
  id_paciente: number
  taxa_amostragem: number
  data_upload: string
  status_exame: string
  laudo_texto?: string | null
}

export interface DiagnosticoEmProcessamento extends DiagnosticoExameBase {
  status: 'em_processamento'
  message: string
}

export interface DiagnosticoConcluido extends DiagnosticoExameBase {
  status: 'concluido'
  resultado_score: number
  classificacao_clinica: string
  mapa_shap_url: string
  data_analise: string
}

export type DiagnosticoResponse =
  | DiagnosticoEmProcessamento
  | DiagnosticoConcluido

export interface EegPonto {
  tempo: number
  amplitude: number
}

export interface SinaisExameResponse {
  exame_id: number
  pontos: EegPonto[]
  taxa_amostragem_hz: number
  n_canais_eeg: number
  canais_eeg: string[]
  n_pontos_original: number
  n_pontos_retornados: number
}

export interface AnaliseIARequest {
  canais_selecionados?: string[] | null
}

export interface AnaliseIAResponse {
  exame_id: number
  status: 'em_processamento'
  canais_processados: string[]
  message: string
}
