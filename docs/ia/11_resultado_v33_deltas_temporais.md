# v33: deltas temporais por canal

## Hipotese e controle

A v33 manteve a CNN-BiLSTM, o split, a amostragem por sequencia, a weighted
BCE e o pos-processamento da baseline. A unica mudanca foi anexar, para cada
canal e feature, a diferenca entre passos temporais consecutivos. A diferenca
do primeiro passo e zero e nunca atravessa a fronteira entre sequencias.

Treino: 68 EDFs, 4.053 sequencias, 789 positivas. Calibracao: `chb16`,
`chb19`, `chb20` e `chb24`, com 13 EDFs, 11.587 sequencias e 20 eventos.
`chb09`, `chb15` e `chb18` nao participaram da escolha. O Random Forest
permaneceu desligado.

## Resultado na calibracao

Media de tres seeds pareadas:

| Entrada | F1 EDF | F1 localizado | Eventos | FA/h |
|---|---:|---:|---:|---:|
| Baseline | 87,9% | 70,2% | 12,67/20 | 0,89 |
| Baseline + deltas | 94,1% | 77,9% | 13,33/20 | 0,81 |

Os deltas melhoraram todas as medias, mas a seed 44 caiu de 13 para 12
eventos. O candidato foi classificado como promissor, ainda sem promocao.

## Teste de desenvolvimento reservado

A variante foi congelada antes do acesso e executada com as seeds historicas
43, 44 e 45. Este conjunto ja foi consultado em experimentos anteriores e
nao representa validacao externa intocada.

| Metrica | v23b | v29 | v33 |
|---|---:|---:|---:|
| F1 EDF | 86,7% | 86,7% | 86,7% |
| F1 localizado | 66,7% | 61,1% | 55,6% |
| Eventos | 6/9 | 7/9 | 6/9 |
| FA/h | 1,36 | 1,30 | 1,69 |

Decisao: **REJEITAR**. A v33 nao deve substituir o modelo ativo.

## Evidencia por crise

- `chb18_30`: o pico ictal subiu de aproximadamente 0,243 na v29 para 0,54,
  mas continuou sem bloco temporal valido;
- `chb15_17`: apareceu um candidato correto de 28 s com pico 0,924 e 73,7%
  de cobertura da crise, mas a duracao calibrada era 30 s; falsos trechos mais
  longos ocuparam as primeiras posicoes;
- `chb15_10`: permaneceu praticamente invisivel, com pico ictal 0,258 e apenas
  4 s de evidencia;
- `chb09_06` e `chb15_20`: a crise apareceu em candidato secundario, mas um
  falso trecho recebeu o primeiro lugar.

Os deltas recuperam parte da transicao ictal, mas tambem amplificam transientes
nao ictais. O problema nao pode ser resolvido promovendo esta entrada nem
reduzindo globalmente a duracao minima, pois isso aumentaria os falsos alarmes.

## Proximo experimento

O proximo candidato deve preservar a branch de features atual e adicionar uma
branch curta de morfologia temporal regularizada, em vez de anexar deltas de
todas as features. A comparacao deve ocorrer primeiro em validacao agrupada por
paciente, com as mesmas metricas e sem novo ajuste em `chb09/chb15/chb18`.
Uma frontend EEGNet-like pequena ou convolucao depthwise por canal e adequada;
transformer ou aumento indiscriminado de contexto nao e justificado pelos
dados atuais.

Artefato: `modelos/v33_temporal_delta_calibration.json`.
Implementacao: `app/ai_engine/temporal_representation.py` e
`scripts/experiment_temporal_delta_v33.py`.
