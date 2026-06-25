import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'
import {
  atualizarPaciente,
  criarPaciente,
  excluirPaciente,
  listarPacientes,
} from '../api/pacientes'
import { useToast } from '../contexts/ToastContext'
import type { Paciente, PacienteCreate } from '../types/api'

function formatarCpf(cpf: string | null | undefined): string {
  if (!cpf) return '-'
  const digitos = cpf.replace(/\D/g, '')
  if (digitos.length !== 11) return cpf
  return `${digitos.slice(0, 3)}.${digitos.slice(3, 6)}.${digitos.slice(6, 9)}-${digitos.slice(9)}`
}

function pacienteParaForm(paciente: Paciente): PacienteCreate {
  return {
    nome: paciente.nome,
    data_nascimento: paciente.data_nascimento,
    sexo: paciente.sexo === 'F' ? 'F' : 'M',
    cpf: paciente.cpf ?? '',
    telefone: paciente.telefone ?? '',
    observacoes: paciente.observacoes ?? '',
  }
}

const formInicial: PacienteCreate = {
  nome: '',
  data_nascimento: '',
  sexo: 'M',
  cpf: '',
  telefone: '',
  observacoes: '',
}

export function GestaoPacientes() {
  const { sucesso, erro: toastErro } = useToast()
  const [pacientes, setPacientes] = useState<Paciente[]>([])
  const [carregando, setCarregando] = useState(true)
  const [modalAberto, setModalAberto] = useState(false)
  const [salvando, setSalvando] = useState(false)
  const [removendoId, setRemovendoId] = useState<number | null>(null)
  const [pacienteEditando, setPacienteEditando] = useState<Paciente | null>(null)
  const [form, setForm] = useState<PacienteCreate>(formInicial)
  const [busca, setBusca] = useState('')

  const carregar = useCallback(async () => {
    setCarregando(true)
    try {
      const lista = await listarPacientes()
      setPacientes(lista)
    } catch {
      toastErro('Nao foi possivel carregar a lista de pacientes.')
    } finally {
      setCarregando(false)
    }
  }, [toastErro])

  useEffect(() => {
    void carregar()
  }, [carregar])

  const pacientesFiltrados = useMemo(() => {
    const termo = busca.trim().toLowerCase()
    if (!termo) return pacientes
    return pacientes.filter(
      (p) =>
        p.nome.toLowerCase().includes(termo) ||
        (p.cpf?.replace(/\D/g, '') ?? '').includes(termo.replace(/\D/g, '')),
    )
  }, [pacientes, busca])

  const abrirNovo = () => {
    setPacienteEditando(null)
    setForm(formInicial)
    setModalAberto(true)
  }

  const abrirEdicao = (paciente: Paciente) => {
    setPacienteEditando(paciente)
    setForm(pacienteParaForm(paciente))
    setModalAberto(true)
  }

  const normalizarForm = (): PacienteCreate => ({
    ...form,
    nome: form.nome.trim(),
    cpf: form.cpf?.replace(/\D/g, '') || null,
    telefone: form.telefone?.trim() || null,
    observacoes: form.observacoes?.trim() || null,
  })

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setSalvando(true)

    try {
      const payload = normalizarForm()
      if (pacienteEditando) {
        await atualizarPaciente(pacienteEditando.id, payload)
        sucesso(`Paciente "${payload.nome}" atualizado com sucesso.`)
      } else {
        await criarPaciente(payload)
        sucesso(`Paciente "${payload.nome}" cadastrado com sucesso.`)
      }
      setModalAberto(false)
      setPacienteEditando(null)
      await carregar()
    } catch {
      toastErro('Falha ao salvar paciente. Verifique os dados e tente novamente.')
    } finally {
      setSalvando(false)
    }
  }

  const handleExcluir = async (paciente: Paciente) => {
    const confirmar = window.confirm(
      `Excluir o paciente "${paciente.nome}" e todos os exames vinculados?`,
    )
    if (!confirmar) return

    setRemovendoId(paciente.id)
    try {
      await excluirPaciente(paciente.id)
      sucesso(`Paciente "${paciente.nome}" excluido com sucesso.`)
      await carregar()
    } catch {
      toastErro('Nao foi possivel excluir o paciente.')
    } finally {
      setRemovendoId(null)
    }
  }

  const tituloModal = pacienteEditando ? 'Editar Paciente' : 'Novo Paciente'

  return (
    <div className="p-8">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-clinical-900">Gestao de Pacientes</h2>
          <p className="mt-1 text-sm text-clinical-500">
            Cadastro, edicao e encaminhamento para novos exames EEG.
          </p>
        </div>
        <button
          type="button"
          onClick={abrirNovo}
          className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-accent-dark"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Novo Paciente
        </button>
      </header>

      <div className="relative mb-4">
        <svg
          className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-clinical-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          type="search"
          placeholder="Buscar por nome ou CPF..."
          value={busca}
          onChange={(e) => setBusca(e.target.value)}
          className="w-full max-w-sm rounded-lg border border-clinical-200 bg-white py-2.5 pl-9 pr-4 text-sm outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
        />
        {busca && (
          <span className="ml-4 text-sm text-clinical-500">
            {pacientesFiltrados.length} resultado{pacientesFiltrados.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      <div className="overflow-hidden rounded-xl border border-clinical-200 bg-white shadow-clinical">
        <table className="min-w-full text-left text-sm">
          <thead className="border-b border-clinical-100 bg-clinical-50 text-xs uppercase tracking-wide text-clinical-500">
            <tr>
              <th className="px-5 py-3 font-semibold">Nome</th>
              <th className="px-5 py-3 font-semibold">CPF</th>
              <th className="px-5 py-3 font-semibold text-right">Acoes</th>
            </tr>
          </thead>
          <tbody>
            {carregando ? (
              <tr>
                <td colSpan={3} className="px-5 py-10 text-center text-clinical-500">
                  Carregando pacientes...
                </td>
              </tr>
            ) : pacientesFiltrados.length === 0 ? (
              <tr>
                <td colSpan={3} className="px-5 py-10 text-center text-clinical-500">
                  {busca ? `Nenhum resultado para "${busca}".` : 'Nenhum paciente cadastrado.'}
                </td>
              </tr>
            ) : (
              pacientesFiltrados.map((paciente) => (
                <tr
                  key={paciente.id}
                  className="border-b border-clinical-100 transition-colors last:border-0 hover:bg-clinical-50"
                >
                  <td className="px-5 py-4 font-medium text-clinical-900">{paciente.nome}</td>
                  <td className="px-5 py-4 font-mono text-clinical-700">{formatarCpf(paciente.cpf)}</td>
                  <td className="px-5 py-4">
                    <div className="flex justify-end gap-2">
                      <Link
                        to={`/pacientes/${paciente.id}/exame`}
                        className="inline-flex items-center gap-1.5 rounded-lg bg-accent/10 px-3 py-1.5 text-xs font-semibold text-accent transition hover:bg-accent hover:text-white"
                      >
                        Novo Exame
                      </Link>
                      <button
                        type="button"
                        onClick={() => abrirEdicao(paciente)}
                        className="rounded-lg border border-clinical-200 px-3 py-1.5 text-xs font-semibold text-clinical-700 transition hover:border-accent hover:text-accent"
                      >
                        Editar
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleExcluir(paciente)}
                        disabled={removendoId === paciente.id}
                        className="rounded-lg border border-red-200 px-3 py-1.5 text-xs font-semibold text-alert-crisis transition hover:bg-red-50 disabled:opacity-50"
                      >
                        {removendoId === paciente.id ? 'Excluindo...' : 'Excluir'}
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {modalAberto && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-backdrop-in">
          <div
            className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
            onClick={() => setModalAberto(false)}
            aria-hidden
          />
          <div
            className="relative w-full max-w-lg rounded-2xl bg-white p-7 shadow-2xl animate-modal-in"
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-titulo"
          >
            <button
              type="button"
              onClick={() => setModalAberto(false)}
              className="absolute right-4 top-4 rounded-lg p-1.5 text-clinical-400 transition hover:bg-clinical-100 hover:text-clinical-700"
            >
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            <div className="mb-5">
              <h3 id="modal-titulo" className="text-lg font-semibold text-clinical-900">
                {tituloModal}
              </h3>
              <p className="text-xs text-clinical-500">Preencha os dados cadastrais.</p>
            </div>

            <form onSubmit={(e) => void handleSubmit(e)} className="space-y-4">
              <label className="block text-sm">
                <span className="font-medium text-clinical-700">Nome completo</span>
                <input
                  required
                  value={form.nome}
                  onChange={(e) => setForm({ ...form, nome: e.target.value })}
                  className="mt-1 w-full rounded-lg border border-clinical-200 px-3 py-2.5 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                  placeholder="Nome do paciente"
                />
              </label>

              <div className="grid grid-cols-2 gap-4">
                <label className="block text-sm">
                  <span className="font-medium text-clinical-700">Nascimento</span>
                  <input
                    required
                    type="date"
                    value={form.data_nascimento}
                    onChange={(e) => setForm({ ...form, data_nascimento: e.target.value })}
                    className="mt-1 w-full rounded-lg border border-clinical-200 px-3 py-2.5 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                  />
                </label>
                <label className="block text-sm">
                  <span className="font-medium text-clinical-700">Sexo</span>
                  <select
                    value={form.sexo}
                    onChange={(e) => setForm({ ...form, sexo: e.target.value as 'M' | 'F' })}
                    className="mt-1 w-full rounded-lg border border-clinical-200 px-3 py-2.5 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                  >
                    <option value="M">Masculino</option>
                    <option value="F">Feminino</option>
                  </select>
                </label>
              </div>

              <label className="block text-sm">
                <span className="font-medium text-clinical-700">CPF</span>
                <input
                  value={form.cpf ?? ''}
                  onChange={(e) => setForm({ ...form, cpf: e.target.value })}
                  placeholder="Somente numeros"
                  className="mt-1 w-full rounded-lg border border-clinical-200 px-3 py-2.5 font-mono outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                />
              </label>

              <label className="block text-sm">
                <span className="font-medium text-clinical-700">Telefone</span>
                <input
                  value={form.telefone ?? ''}
                  onChange={(e) => setForm({ ...form, telefone: e.target.value })}
                  className="mt-1 w-full rounded-lg border border-clinical-200 px-3 py-2.5 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                />
              </label>

              <label className="block text-sm">
                <span className="font-medium text-clinical-700">Observacoes</span>
                <textarea
                  rows={3}
                  value={form.observacoes ?? ''}
                  onChange={(e) => setForm({ ...form, observacoes: e.target.value })}
                  className="mt-1 w-full resize-none rounded-lg border border-clinical-200 px-3 py-2.5 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                />
              </label>

              <div className="flex justify-end gap-3 pt-1">
                <button
                  type="button"
                  onClick={() => setModalAberto(false)}
                  className="rounded-lg border border-clinical-300 px-4 py-2 text-sm font-medium text-clinical-700 transition hover:bg-clinical-50"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  disabled={salvando}
                  className="rounded-lg bg-accent px-5 py-2 text-sm font-semibold text-white transition hover:bg-accent-dark disabled:opacity-60"
                >
                  {salvando ? 'Salvando...' : pacienteEditando ? 'Salvar Alteracoes' : 'Cadastrar Paciente'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
