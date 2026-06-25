import { useCallback, useEffect, useRef, useState } from 'react'
import { obterDiagnostico, solicitarAnaliseIA } from '../api/exames'
import type { DiagnosticoConcluido, DiagnosticoResponse } from '../types/api'
import { isDiagnosticoConcluido } from '../utils/eeg'

const INTERVALO_POLL_MS = 2500
const MAX_TENTATIVAS = 40

export function useDiagnosticoPolling() {
  const [exameId, setExameId] = useState<number | null>(null)
  const [diagnostico, setDiagnostico] = useState<DiagnosticoResponse | null>(
    null,
  )
  const [carregando, setCarregando] = useState(false)
  const [erro, setErro] = useState<string | null>(null)
  const [ativo, setAtivo] = useState(false)

  const tentativasRef = useRef(0)
  const pollGenerationRef = useRef(0)
  const concluidoRef = useRef(false)

  const parar = useCallback(() => {
    pollGenerationRef.current += 1
    setAtivo(false)
    setCarregando(false)
  }, [])

  const resetar = useCallback(() => {
    pollGenerationRef.current += 1
    concluidoRef.current = false
    setAtivo(false)
    setCarregando(false)
    setExameId(null)
    setDiagnostico(null)
    setErro(null)
  }, [])

  const aplicarDiagnostico = useCallback((resultado: DiagnosticoResponse) => {
    if (isDiagnosticoConcluido(resultado)) {
      concluidoRef.current = true
      setDiagnostico(resultado)
      return
    }

    if (concluidoRef.current) {
      return
    }

    setDiagnostico(resultado)
  }, [])

  const iniciar = useCallback(async (id: number, canaisSelecionados: string[]) => {
    pollGenerationRef.current += 1
    concluidoRef.current = false
    setExameId(id)
    setDiagnostico(null)
    setErro(null)
    tentativasRef.current = 0
    setCarregando(true)
    setAtivo(false)

    try {
      await solicitarAnaliseIA(id, canaisSelecionados)
      setAtivo(true)
    } catch {
      setErro('Falha ao enfileirar a análise IA. Verifique os canais selecionados.')
      setCarregando(false)
    }
  }, [])

  const retomar = useCallback(
    (id: number) => {
      if (concluidoRef.current && exameId === id) {
        return
      }
      pollGenerationRef.current += 1
      concluidoRef.current = false
      setExameId(id)
      setErro(null)
      tentativasRef.current = 0
      setCarregando(true)
      setAtivo(true)
    },
    [exameId],
  )

  useEffect(() => {
    if (!ativo || exameId == null) return

    const geracao = pollGenerationRef.current
    let cancelado = false
    let timeoutId: ReturnType<typeof setTimeout>

    const consultar = async () => {
      if (cancelado || geracao !== pollGenerationRef.current) return

      try {
        const resultado = await obterDiagnostico(exameId)
        if (cancelado || geracao !== pollGenerationRef.current) return

        aplicarDiagnostico(resultado)

        if (isDiagnosticoConcluido(resultado)) {
          parar()
          return
        }

        tentativasRef.current += 1
        if (tentativasRef.current >= MAX_TENTATIVAS) {
          setErro('Tempo limite aguardando conclusão da análise IA.')
          parar()
          return
        }

        timeoutId = setTimeout(consultar, INTERVALO_POLL_MS)
      } catch {
        if (!cancelado && geracao === pollGenerationRef.current) {
          setErro('Não foi possível consultar o diagnóstico do exame.')
          parar()
        }
      }
    }

    void consultar()

    return () => {
      cancelado = true
      clearTimeout(timeoutId)
    }
  }, [ativo, exameId, parar, aplicarDiagnostico])

  const concluido: DiagnosticoConcluido | null = isDiagnosticoConcluido(
    diagnostico,
  )
    ? diagnostico
    : null

  return {
    diagnostico,
    concluido,
    exameId,
    carregando,
    erro,
    iniciar,
    retomar,
    parar,
    resetar,
  }
}
