# Auditoria de estabilidade da v40 por evento

## Escopo

As seeds 42 e 44 foram reproduzidas com o manifesto, sampler, arquitetura,
loss, checkpoint e calibracao da v40. A auditoria usou somente `chb16`, `chb19`,
`chb20` e `chb24`. Os pacientes reservados `chb09`, `chb15` e `chb18` nao
foram carregados.

O artefato `modelos/v40_seed_stability_audit.json` registra, para cada uma das
20 crises de calibracao, score ictal, deteccao, sobreposicao, erro de inicio e
duracao prevista nos tetos FROC de 0,50 e 0,75 FA/h.

## Resultado no teto de 0,50 FA/h

- Seed 42: 11/20 eventos.
- Seed 44: 14/20 eventos.
- Onze eventos foram detectados pelas duas seeds.
- Seis eventos foram perdidos pelas duas seeds.
- Tres eventos foram detectados apenas pela seed 44.

### Eventos perdidos pelas duas seeds

| EDF/evento | Duracao | Score max seed 42 | Score max seed 44 | Diagnostico |
|---|---:|---:|---:|---|
| `chb16_11`/1 | 12 s | 0,778 | 0,965 | evidencia alta, mas sem continuidade/duracao valida |
| `chb16_17`/1 | 12 s | 0,333 | 0,527 | evidencia moderada e insuficiente |
| `chb16_17`/2 | 10 s | 0,047 | 0,261 | representacao fraca nas duas seeds |
| `chb16_17`/3 | 12 s | 0,018 | 0,087 | representacao fraca nas duas seeds |
| `chb16_17`/4 | 12 s | 0,169 | 0,780 | representacao muito instavel |
| `chb24_04`/3 | 22 s | 0,999 | 1,000 | evidencia alta, bloqueada pela regra temporal de 26 s |

A duracao minima escolhida no ponto estrito e 26 segundos. Portanto, eventos
reais de 10 a 22 segundos dependem de o score permanecer alto fora da anotacao
para formar um candidato valido. Isso explica por que aumentar apenas o peso de
crises curtas na v41 nao resolveu o problema.

### Eventos que mudaram entre seeds

| EDF/evento | Duracao | Seed 42 | Seed 44 | Score 42/44 | Interpretacao |
|---|---:|---|---|---:|---|
| `chb20_13`/1 | 34 s | perdido | detectado | 0,975 / 0,999 | score existe; continuidade/localizacao muda |
| `chb20_14`/1 | 40 s | perdido | detectado | 0,434 / 0,988 | representacao depende fortemente da seed |
| `chb24_01`/2 | 28 s | perdido | detectado | 0,999 / 1,000 | score existe; candidato fica fragmentado na seed 42 |

Dois dos tres eventos variaveis ja possuem pico quase unitario na seed 42. O
problema nesses casos nao e ausencia de evidencia, mas sustentar e agrupar essa
evidencia no intervalo correto.

## Resultado no teto de 0,75 FA/h

Com duracao minima de 18 segundos, a seed 42 detecta 13/20 e a seed 44 detecta
15/20. `chb24_04`/3 passa a ser detectado nas duas seeds. Permanecem perdidos
nas duas seeds `chb16_11` e os quatro eventos de `chb16_17`, todos com 10-12
segundos. Apenas `chb20_12` e `chb20_14` mudam entre seeds.

## Conclusao

O gargalo nao e unico:

1. **Regra temporal incompativel com crises curtas:** `chb16_11` e
   `chb24_04`/3 apresentam scores altos, mas a duracao minima global elimina os
   candidatos.
2. **Representacao realmente fraca:** dois eventos de `chb16_17` permanecem
   praticamente invisiveis nas duas seeds.
3. **Instabilidade de otimizacao:** `chb20_14` e `chb16_17`/4 mudam muito de
   score conforme a seed.
4. **Fragmentacao:** `chb20_13`/1 e `chb24_01`/2 tem pico alto nas duas seeds,
   mas nem sempre formam um trecho valido.

Assim, trocar toda a arquitetura agora misturaria problemas distintos. O
proximo experimento controlado deve manter o modelo v40 e testar uma regra de
dois caminhos somente na calibracao:

- caminho sustentado: ponto atual, com duracao de 18-26 s;
- caminho transiente de alta confianca: limiar alto e duracao de 6-10 s.

Os dois caminhos devem ser unidos antes da contagem de eventos e avaliados por
FROC. A regra so deve continuar se recuperar crises curtas sem ultrapassar o
teto predefinido de FA/h. Depois disso, os eventos de score realmente baixo em
`chb16_17` justificam uma mudanca de representacao separada.
