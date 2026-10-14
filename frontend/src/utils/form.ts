import axios from 'axios'

export function somenteDigitos(valor: string, limite?: number): string {
  const digitos = valor.replace(/\D/g, '')
  return limite ? digitos.slice(0, limite) : digitos
}

export function somenteNome(valor: string): string {
  return valor.replace(/[^A-Za-zÀ-ÖØ-öø-ÿ' -]/g, '').replace(/\s{2,}/g, ' ')
}

export function formatarCpf(valor: string | null | undefined): string {
  const digitos = somenteDigitos(valor ?? '', 11)
  if (!digitos) return ''
  return digitos
    .replace(/^(\d{3})(\d)/, '$1.$2')
    .replace(/^(\d{3})\.(\d{3})(\d)/, '$1.$2.$3')
    .replace(/\.(\d{3})(\d)/, '.$1-$2')
}

export function formatarTelefone(valor: string | null | undefined): string {
  const digitos = somenteDigitos(valor ?? '', 11)
  if (!digitos) return ''
  if (digitos.length <= 10) {
    return digitos
      .replace(/^(\d{2})(\d)/, '($1) $2')
      .replace(/(\d{4})(\d)/, '$1-$2')
  }
  return digitos
    .replace(/^(\d{2})(\d)/, '($1) $2')
    .replace(/(\d{5})(\d)/, '$1-$2')
}

export function mensagemErroApi(erro: unknown, fallback: string): string {
  if (!axios.isAxiosError(erro)) return fallback
  const detalhe = erro.response?.data?.detail
  if (typeof detalhe === 'string') return detalhe
  if (Array.isArray(detalhe)) {
    const mensagens = detalhe
      .map((item) => (typeof item?.msg === 'string' ? item.msg.replace(/^Value error,\s*/i, '') : null))
      .filter(Boolean)
    if (mensagens.length) return mensagens.join(' ')
  }
  return fallback
}

export function cpfValido(valor: string): boolean {
  const cpf = somenteDigitos(valor)
  if (cpf.length !== 11 || /^(\d)\1{10}$/.test(cpf)) return false
  for (let posicao = 9; posicao <= 10; posicao += 1) {
    let soma = 0
    for (let indice = 0; indice < posicao; indice += 1) {
      soma += Number(cpf[indice]) * (posicao + 1 - indice)
    }
    const digito = ((soma * 10) % 11) % 10
    if (digito !== Number(cpf[posicao])) return false
  }
  return true
}
