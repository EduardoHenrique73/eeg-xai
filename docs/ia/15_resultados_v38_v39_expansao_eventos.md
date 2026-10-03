# v38-v39: expansao de eventos e sampler por crise

## Dados adicionados

A base de treino recebeu sete EDFs novos, alem dos dois ja adicionados na v37:

| EDF | Eventos | Duracao ictal total |
|---|---:|---:|
| `chb12_27.edf` | 6 | 229 s |
| `chb03_34.edf` | 1 | 47 s |
| `chb05_17.edf` | 1 | 120 s |
| `chb08_13.edf` | 1 | 160 s |
| `chb14_27.edf` | 1 | 16 s |
| `chb22_38.edf` | 1 | 72 s |
| `chb21_22.edf` | 1 | 12 s |

O manifesto v38 adiciona 14 eventos em nove EDFs em relacao a baseline v23.
Calibracao e pacientes reservados permaneceram inalterados.

## Compatibilidade de montagem

`chb12_27` usa canais referenciais como `F7-CS2`, enquanto a rede espera
derivacoes bipolares como `F7-T7`. O pipeline passou a reconstruir uma
derivacao bipolar quando os dois eletrodos possuem referencia comum:

`F7-T7 = (F7-CS2) - (T7-CS2)`

No arquivo, 20 dos 23 canais da montagem foram reconstruidos e tres foram
explicitamente omitidos. O comportamento possui teste automatizado e nao muda
arquivos que ja contem diretamente a montagem bipolar.

## Problema encontrado na v38

Com 32 sequencias ictais permitidas por EDF, o sampler anterior escolhia quatro
blocos uniformes no arquivo inteiro. Em `chb12_27`, isso eliminou completamente
as crises 2 e 4 do treino. Portanto, adicionar dados sem amostragem por evento
reduziu o resultado:

| Variante | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h |
|---|---:|---:|---:|---:|---:|
| Baseline v32 | 12,67/20 | - | 70,2% | 87,9% | 0,89 |
| v37, dois eventos novos | 12,67/20 | 59,7% | 72,5% | 89,6% | 0,76 |
| v38, 14 eventos novos | 11,33/20 | 53,4% | 67,5% | 87,9% | 0,86 |
| v39, sampler por evento | 12,33/20 | 64,7% | 75,8% | 90,9% | 0,47 |

## v39 por seed

| Seed | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h |
|---:|---:|---:|---:|---:|---:|
| 42 | 12/20 | 57,1% | 73,7% | 90,9% | 0,78 |
| 43 | 12/20 | 66,7% | 80,0% | 90,9% | 0,31 |
| 44 | 13/20 | 70,3% | 73,7% | 90,9% | 0,31 |

O sampler v39 divide a cota positiva entre blocos temporais ictais distintos.
No `chb12_27`, todas as seis crises passaram a contribuir com 5–6 sequencias,
incluindo inicio, centro e fim quando a duracao permite.

## Decisao

**PROMISSOR MAS NAO SUPERIOR.** A v39 melhora claramente precisao temporal,
F1 por EDF e falsos alarmes, mas sua media de 12,33 eventos ainda fica pouco
abaixo dos 12,67 da baseline. Ela nao sera promovida nem avaliada nos pacientes
reservados nesta etapa.

O proximo gargalo e recuperar sensibilidade sem perder o ganho de FA/h. A
proxima comparacao deve manter os dados e o sampler v39 e alterar somente a
ponderacao das crises curtas ou a selecao do checkpoint. Nao ha justificativa
para nova arquitetura antes dessa ablação.

Artefatos: `modelos/v38_expanded_train_calibration.json` e
`modelos/v39_event_balanced_sampling_calibration.json`.
