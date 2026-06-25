import { useCallback, useEffect, useState } from 'react'

import { Link, useParams } from 'react-router-dom'

import { obterSinaisExame, uploadExame } from '../api/exames'

import { obterPaciente } from '../api/pacientes'

import { ClinicalLayout } from '../components/layout/ClinicalLayout'

import { IaLaudoPanel } from '../components/laudo/IaLaudoPanel'

import { PacienteCard } from '../components/paciente/PacienteCard'

import { EdfDropzone } from '../components/upload/EdfDropzone'

import { EegSignalChart } from '../components/visualizador/EegSignalChart'

import { useDiagnosticoPolling } from '../hooks/useDiagnosticoPolling'

import type { Paciente } from '../types/api'

const TAXA_AMOSTRAGEM_PADRAO = 256

export function VisualizadorClinico() {
  const { pacienteId } = useParams<{ pacienteId: string }>()
  const pacienteIdNum = Number(pacienteId)

  const [paciente, setPaciente] = useState<Paciente | null>(null)
  const [carregandoPaciente, setCarregandoPaciente] = useState(true)
  const [erroPaciente, setErroPaciente] = useState<string | null>(null)

  const [arquivoEdf, setArquivoEdf] = useState<File | null>(null)
  const [exameIdUpload, setExameIdUpload] = useState<number | null>(null)
  const [erroUpload, setErroUpload] = useState<string | null>(null)
  const [enviandoArquivo, setEnviandoArquivo] = useState(false)

  const [canaisEeg, setCanaisEeg] = useState<string[]>([])
  const [canaisSelecionados, setCanaisSelecionados] = useState<string[]>([])
  const [carregandoCanais, setCarregandoCanais] = useState(false)

  const {
    diagnostico,
    concluido,
    exameId: exameIdAnalise,
    carregando: analiseEmAndamento,
    erro: erroDiagnostico,
    iniciar: iniciarPolling,
    retomar: retomarPolling,
    resetar: resetarPolling,
  } = useDiagnosticoPolling()

  const exameId = exameIdUpload ?? exameIdAnalise

  useEffect(() => {
    if (!Number.isFinite(pacienteIdNum) || pacienteIdNum <= 0) {
      setErroPaciente('Identificador de paciente inválido na URL.')
      setCarregandoPaciente(false)
      return
    }

    let cancelado = false
    setCarregandoPaciente(true)
    setErroPaciente(null)

    obterPaciente(pacienteIdNum)
      .then((dados) => {
        if (!cancelado) setPaciente(dados)
      })
      .catch(() => {
        if (!cancelado) {
          setErroPaciente('Paciente não encontrado.')
          setPaciente(null)
        }
      })
      .finally(() => {
        if (!cancelado) setCarregandoPaciente(false)
      })

    return () => {
      cancelado = true
    }
  }, [pacienteIdNum])

  useEffect(() => {
    if (exameIdUpload == null) {
      setCanaisEeg([])
      setCanaisSelecionados([])
      setCarregandoCanais(false)
      return
    }

    let cancelado = false
    setCarregandoCanais(true)
    setCanaisEeg([])
    setCanaisSelecionados([])

    obterSinaisExame(exameIdUpload)
      .then((resposta) => {
        if (cancelado) return
        const canais = resposta.canais_eeg ?? []
        setCanaisEeg(canais)
        setCanaisSelecionados(canais)
      })
      .catch(() => {
        if (!cancelado) {
          setCanaisEeg([])
          setCanaisSelecionados([])
        }
      })
      .finally(() => {
        if (!cancelado) setCarregandoCanais(false)
      })

    return () => {
      cancelado = true
    }
  }, [exameIdUpload])

  const enviarArquivo = useCallback(
    async (file: File, pacienteAtual: Paciente) => {
      setEnviandoArquivo(true)
      setErroUpload(null)
      resetarPolling()
      setExameIdUpload(null)
      setCanaisEeg([])
      setCanaisSelecionados([])

      try {
        const resposta = await uploadExame(
          file,
          pacienteAtual.id,
          TAXA_AMOSTRAGEM_PADRAO,
        )
        setExameIdUpload(resposta.exame_id)
      } catch {
        setErroUpload(
          'Falha ao enviar o exame. Verifique se o backend está em execução.',
        )
        setArquivoEdf(null)
      } finally {
        setEnviandoArquivo(false)
      }
    },
    [resetarPolling],
  )

  const handleArquivoSelecionado = useCallback(
    (file: File) => {
      if (!paciente) return
      setArquivoEdf(file)
      void enviarArquivo(file, paciente)
    },
    [paciente, enviarArquivo],
  )

  const handleSolicitarAnalise = useCallback(async () => {
    if (!paciente) return

    if (!exameIdUpload) {
      setErroUpload('Aguarde o envio do arquivo .edf ou selecione um exame válido.')
      return
    }

    if (canaisSelecionados.length === 0) {
      setErroUpload('Selecione ao menos um canal EEG para análise.')
      return
    }

    if (concluido && exameIdAnalise === exameIdUpload) return

    setErroUpload(null)

    if (exameIdAnalise === exameIdUpload && analiseEmAndamento) {
      return
    }

    if (exameIdAnalise === exameIdUpload && !concluido) {
      retomarPolling(exameIdUpload)
      return
    }

    await iniciarPolling(exameIdUpload, canaisSelecionados)
  }, [
    analiseEmAndamento,
    canaisSelecionados,
    concluido,
    exameIdAnalise,
    exameIdUpload,
    iniciarPolling,
    paciente,
    retomarPolling,
  ])

  if (carregandoPaciente) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-clinical-500">
        Carregando dados do paciente...
      </div>
    )
  }

  if (erroPaciente || !paciente) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 p-8">
        <p className="text-alert-crisis">{erroPaciente ?? 'Paciente indisponível.'}</p>
        <Link to="/pacientes" className="text-sm font-semibold text-accent hover:underline">
          Voltar para Gestão de Pacientes
        </Link>
      </div>
    )
  }

  const placeholderGrafico = enviandoArquivo
    ? 'Enviando arquivo .edf e preparando visualização...'
    : arquivoEdf && exameId
      ? 'Carregando ondas cerebrais do exame...'
      : 'Arraste um arquivo .edf para visualizar os sinais EEG.'

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-clinical-200 bg-white px-6 py-3">
        <Link
          to="/pacientes"
          className="text-sm font-medium text-accent hover:underline"
        >
          ← Voltar para Pacientes
        </Link>
        <p className="mt-1 text-xs text-clinical-500">
          Visualizador de exame — Paciente #{paciente.id}
          {exameId != null && ` · Exame #${exameId}`}
        </p>
      </div>

      <div className="min-h-0 flex-1">
        <ClinicalLayout
          left={
            <>
              <PacienteCard paciente={paciente} />
              <EdfDropzone
                selectedFile={arquivoEdf}
                onFileSelected={handleArquivoSelecionado}
                disabled={enviandoArquivo || analiseEmAndamento}
              />
              {enviandoArquivo && (
                <p className="text-center text-xs font-medium text-accent-dark animate-pulse">
                  Enviando exame para o servidor...
                </p>
              )}
            </>
          }
          center={
            <EegSignalChart
              exameId={exameId}
              mapaShapUrl={concluido?.mapa_shap_url}
              placeholder={placeholderGrafico}
            />
          }
          right={
            <IaLaudoPanel
              exameId={exameId}
              canaisEeg={canaisEeg}
              canaisSelecionados={canaisSelecionados}
              onCanaisChange={setCanaisSelecionados}
              carregandoCanais={carregandoCanais}
              onSolicitarAnalise={() => void handleSolicitarAnalise()}
              analiseEmAndamento={analiseEmAndamento}
              analiseConcluida={Boolean(concluido)}
              solicitarDesabilitado={!exameIdUpload || Boolean(concluido)}
              score={concluido?.resultado_score}
              classificacao={concluido?.classificacao_clinica}
              laudoTextoInicial={diagnostico?.laudo_texto}
              statusExameInicial={diagnostico?.status_exame}
              erro={erroUpload ?? erroDiagnostico}
            />
          }
        />
      </div>
    </div>
  )
}
