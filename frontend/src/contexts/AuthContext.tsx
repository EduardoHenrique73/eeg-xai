import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { login as loginApi } from '../api/auth'
import { setAuthToken } from '../api/client'
import type { MedicoAuth } from '../types/auth'

const TOKEN_KEY = 'eeg_xai_token'
const MEDICO_KEY = 'eeg_xai_medico'

interface AuthContextValue {
  medico: MedicoAuth | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (email: string, senha: string) => Promise<void>
  logout: () => void
  atualizarMedico: (medico: MedicoAuth) => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

function lerMedicoStorage(): MedicoAuth | null {
  const raw = localStorage.getItem(MEDICO_KEY)
  if (!raw) return null
  try {
    const dados = JSON.parse(raw) as MedicoAuth
    return {
      ...dados,
      threshold_confianca: dados.threshold_confianca ?? 0.5,
      montagem_padrao: dados.montagem_padrao ?? [],
      exibir_shap: dados.exibir_shap ?? true,
    }
  } catch {
    return null
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY))
  const [medico, setMedico] = useState<MedicoAuth | null>(() => lerMedicoStorage())
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    setAuthToken(token)
    setIsLoading(false)
  }, [token])

  useEffect(() => {
    const handleUnauthorized = () => {
      setToken(null)
      setMedico(null)
      localStorage.removeItem(TOKEN_KEY)
      localStorage.removeItem(MEDICO_KEY)
      setAuthToken(null)
    }

    window.addEventListener('eeg-xai:unauthorized', handleUnauthorized)
    return () => window.removeEventListener('eeg-xai:unauthorized', handleUnauthorized)
  }, [])

  const logout = useCallback(() => {
    setToken(null)
    setMedico(null)
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(MEDICO_KEY)
    setAuthToken(null)
  }, [])

  const atualizarMedico = useCallback((dados: MedicoAuth) => {
    setMedico(dados)
    localStorage.setItem(MEDICO_KEY, JSON.stringify(dados))
  }, [])

  const login = useCallback(async (email: string, senha: string) => {
    const resposta = await loginApi({ email, senha })
    setToken(resposta.access_token)
    setMedico(resposta.medico)
    localStorage.setItem(TOKEN_KEY, resposta.access_token)
    localStorage.setItem(MEDICO_KEY, JSON.stringify(resposta.medico))
    setAuthToken(resposta.access_token)
  }, [])

  const value = useMemo<AuthContextValue>(
    () => ({
      medico,
      token,
      isAuthenticated: Boolean(token),
      isLoading,
      login,
      logout,
      atualizarMedico,
    }),
    [medico, token, isLoading, login, logout, atualizarMedico],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth deve ser usado dentro de AuthProvider')
  }
  return context
}
