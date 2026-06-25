import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { atualizarPerfilMedico, obterPerfilMedico } from '../api/auth'
import { useAuth } from '../contexts/AuthContext'
import { useToast } from '../contexts/ToastContext'
import type { MedicoConfigUpdate } from '../types/auth'

const CANAIS_COMUNS = [
  'FP1-F7',
  'F7-T7',
  'T7-P7',
  'P7-O1',
  'FP2-F8',
  'F8-T8',
  'T8-P8',
  'P8-O2',
  'FP1-F3',
  'F3-C3',
  'C3-P3',
  'P3-O1',
  'FP2-F4',
  'F4-C4',
  'C4-P4',
  'P4-O2',
]

function formInicial(medico: ReturnType<typeof useAuth>['medico']): MedicoConfigUpdate {
  return {
    nome: medico?.nome ?? '',
    email: medico?.email ?? '',
    crm: medico?.crm ?? '',
    threshold_confianca: medico?.threshold_confianca ?? 0.5,
    montagem_padrao: medico?.montagem_padrao ?? [],
    exibir_shap: medico?.exibir_shap ?? true,
  }
}

export function Configuracoes() {
  const { medico, atualizarMedico } = useAuth()
  const { sucesso, erro: toastErro } = useToast()
  const [form, setForm] = useState<MedicoConfigUpdate>(() => formInicial(medico))
  const [canalLivre, setCanalLivre] = useState('')
  const [carregando, setCarregando] = useState(true)
  const [salvando, setSalvando] = useState(false)

  useEffect(() => {
    let cancelado = false
    setCarregando(true)
    obterPerfilMedico()
      .then((dados) => {
        if (cancelado) return
        atualizarMedico(dados)
        setForm(formInicial(dados))
      })
      .catch(() => {
        if (!cancelado) toastErro('Nao foi possivel carregar as configuracoes.')
      })
      .finally(() => {
        if (!cancelado) setCarregando(false)
      })

    return () => {
      cancelado = true
    }
  }, [atualizarMedico, toastErro])

  const thresholdPercentual = useMemo(
    () => Math.round(form.threshold_confianca * 100),
    [form.threshold_confianca],
  )

  const toggleCanal = (canal: string) => {
    const existe = form.montagem_padrao.includes(canal)
    setForm({
      ...form,
      montagem_padrao: existe
        ? form.montagem_padrao.filter((item) => item !== canal)
        : [...form.montagem_padrao, canal],
    })
  }

  const adicionarCanalLivre = () => {
    const canal = canalLivre.trim().toUpperCase()
    if (!canal || form.montagem_padrao.includes(canal)) return
    setForm({ ...form, montagem_padrao: [...form.montagem_padrao, canal] })
    setCanalLivre('')
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setSalvando(true)

    try {
      const atualizado = await atualizarPerfilMedico({
        ...form,
        nome: form.nome.trim(),
        email: form.email.trim().toLowerCase(),
        crm: form.crm.trim(),
        montagem_padrao: form.montagem_padrao.map((canal) => canal.trim()).filter(Boolean),
      })
      atualizarMedico(atualizado)
      setForm(formInicial(atualizado))
      sucesso('Configuracoes salvas com sucesso.')
    } catch {
      toastErro('Nao foi possivel salvar as configuracoes.')
    } finally {
      setSalvando(false)
    }
  }

  return (
    <div className="p-8">
      <header className="mb-6">
        <h2 className="text-2xl font-bold text-clinical-900">Configuracoes</h2>
        <p className="mt-1 text-sm text-clinical-500">
          Perfil medico, preferencias de analise e exibicao da explicabilidade.
        </p>
      </header>

      {carregando ? (
        <div className="rounded-xl border border-clinical-200 bg-white p-6 text-sm text-clinical-500 shadow-clinical">
          Carregando configuracoes...
        </div>
      ) : (
        <form onSubmit={(event) => void handleSubmit(event)} className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="space-y-5">
            <section className="rounded-xl border border-clinical-200 bg-white p-5 shadow-clinical">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-clinical-500">
                Dados do Perfil
              </h3>

              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <label className="block text-sm">
                  <span className="font-medium text-clinical-700">Nome</span>
                  <input
                    required
                    value={form.nome}
                    onChange={(e) => setForm({ ...form, nome: e.target.value })}
                    className="mt-1 w-full rounded-lg border border-clinical-200 px-3 py-2.5 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                  />
                </label>

                <label className="block text-sm">
                  <span className="font-medium text-clinical-700">CRM</span>
                  <input
                    required
                    value={form.crm}
                    onChange={(e) => setForm({ ...form, crm: e.target.value })}
                    className="mt-1 w-full rounded-lg border border-clinical-200 px-3 py-2.5 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                  />
                </label>
              </div>

              <label className="mt-4 block text-sm">
                <span className="font-medium text-clinical-700">E-mail</span>
                <input
                  required
                  type="email"
                  value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })}
                  className="mt-1 w-full rounded-lg border border-clinical-200 px-3 py-2.5 outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                />
              </label>
            </section>

            <section className="rounded-xl border border-clinical-200 bg-white p-5 shadow-clinical">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-wide text-clinical-500">
                    Montagem Padrao
                  </h3>
                  <p className="mt-1 text-sm text-clinical-500">
                    Canais preferenciais usados como referencia para novas analises.
                  </p>
                </div>
                <span className="rounded-full bg-clinical-100 px-3 py-1 text-xs font-semibold text-clinical-700">
                  {form.montagem_padrao.length} selecionado{form.montagem_padrao.length !== 1 ? 's' : ''}
                </span>
              </div>

              <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                {CANAIS_COMUNS.map((canal) => {
                  const ativo = form.montagem_padrao.includes(canal)
                  return (
                    <button
                      type="button"
                      key={canal}
                      onClick={() => toggleCanal(canal)}
                      className={[
                        'rounded-lg border px-3 py-2 text-left font-mono text-xs font-semibold transition',
                        ativo
                          ? 'border-accent bg-accent/10 text-accent-dark'
                          : 'border-clinical-200 bg-white text-clinical-700 hover:border-accent/60',
                      ].join(' ')}
                    >
                      {canal}
                    </button>
                  )
                })}
              </div>

              <div className="mt-4 flex gap-2">
                <input
                  value={canalLivre}
                  onChange={(e) => setCanalLivre(e.target.value)}
                  placeholder="Canal personalizado"
                  className="min-w-0 flex-1 rounded-lg border border-clinical-200 px-3 py-2 text-sm outline-none transition focus:border-accent focus:ring-2 focus:ring-accent/20"
                />
                <button
                  type="button"
                  onClick={adicionarCanalLivre}
                  className="rounded-lg border border-clinical-300 px-4 py-2 text-sm font-semibold text-clinical-700 transition hover:border-accent hover:text-accent"
                >
                  Adicionar
                </button>
              </div>
            </section>
          </div>

          <aside className="space-y-5">
            <section className="rounded-xl border border-clinical-200 bg-white p-5 shadow-clinical">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-clinical-500">
                IA e XAI
              </h3>

              <label className="mt-5 block text-sm">
                <span className="flex items-center justify-between gap-3 font-medium text-clinical-700">
                  Threshold de confianca
                  <span className="rounded-full bg-accent/10 px-2.5 py-1 text-xs font-bold text-accent-dark">
                    {thresholdPercentual}%
                  </span>
                </span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={form.threshold_confianca}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      threshold_confianca: Number(e.target.value),
                    })
                  }
                  className="mt-3 w-full accent-teal-600"
                />
              </label>

              <label className="mt-5 flex items-start gap-3 rounded-lg border border-clinical-200 bg-clinical-50 p-3 text-sm">
                <input
                  type="checkbox"
                  checked={form.exibir_shap}
                  onChange={(e) => setForm({ ...form, exibir_shap: e.target.checked })}
                  className="mt-0.5 h-4 w-4 rounded border-clinical-300 text-accent focus:ring-accent/30"
                />
                <span>
                  <span className="block font-medium text-clinical-800">Exibir mapas SHAP</span>
                  <span className="mt-1 block text-xs text-clinical-500">
                    Quando desligado, o diagnostico mostra score e classificacao sem a imagem XAI.
                  </span>
                </span>
              </label>
            </section>

            <button
              type="submit"
              disabled={salvando}
              className="w-full rounded-lg bg-accent px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-accent-dark disabled:opacity-60"
            >
              {salvando ? 'Salvando...' : 'Salvar Configuracoes'}
            </button>
          </aside>
        </form>
      )}
    </div>
  )
}
