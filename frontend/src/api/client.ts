import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE_URL ?? ''

export const apiClient = axios.create({
  baseURL,
  headers: {
    Accept: 'application/json',
  },
})

export function setAuthToken(token: string | null): void {
  if (token) {
    apiClient.defaults.headers.common.Authorization = `Bearer ${token}`
  } else {
    delete apiClient.defaults.headers.common.Authorization
  }
}

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && !error.config?.url?.includes('/api/auth/login')) {
      window.dispatchEvent(new CustomEvent('eeg-xai:unauthorized'))
    }
    return Promise.reject(error)
  },
)
