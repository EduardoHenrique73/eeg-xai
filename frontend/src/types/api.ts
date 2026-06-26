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

export type PacienteUpdate = Partial<PacienteCreate>

export interface ExameUploadResponse {
  message: string
  exame_id: number
  arquivo_path: string
  taxa_amostragem: number
  canais_eeg: string[]
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
  mapa_shap_url?: string | null
  threshold_confianca: number
  feature_mode?: string | null
  canais_processados: string[]
  canais_omitidos: string[]
  canais_destaque: Array<{
    canal: string
    score: number
    impacto?: number
    score_sem_canal?: number
  }>
  n_janelas_analisadas?: number | null
  janela_pico?: {
    start_seconds: number
    end_seconds: number
    score?: number
  } | null
  janelas_top: Array<{
    start_seconds: number
    end_seconds: number
    score: number
  }>
  score_agregacao?: string | null
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
