# Resultado v44: estabilidade de otimizacao da fusao tardia

Atualizado em: 2026-10-07

## Hipotese

A v44 manteve integralmente arquitetura, dados, sampler, loss e
pos-processamento da v43. Foram alterados somente:

- learning rate inicial: `0,001` para `0,0005`;
- `ReduceLROnPlateau`, fator 0,5, paciencia 2 e minimo `0,00005`;
- checkpoint: F1 EDF passou a desempatar antes da precisao de evento;
- paciencia do checkpoint: 8; minimo de 10 e maximo de 25 epocas.

O experimento usou as seeds 42-46, calibracao fixa e ponto FROC com
`FA/h <= 0,75`. Os pacientes historicamente reservados nao foram acessados.

## Resultado por seed

| Seed | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h | Epoca escolhida |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 14/20 | 66,7% | 80,0% | 95,7% | 0,63 | 9 |
| 43 | 13/20 | 65,0% | 80,0% | 95,7% | 0,55 | 7 |
| 44 | 13/20 | 63,4% | 73,7% | 95,7% | 0,63 | 7 |
| 45 | 12/20 | 60,0% | 73,7% | 90,9% | 0,63 | 2 |
| 46 | 15/20 | 68,2% | 76,2% | 96,0% | 0,71 | 3 |
| **Media** | **13,4/20** | **64,7%** | **76,7%** | **94,8%** | **0,63** | - |

## Comparacao com v43

| Metrica | v43 | v44 | Diferenca |
|---|---:|---:|---:|
| Eventos | 13,8/20 | 13,4/20 | -0,4 |
| F1 evento | 65,7% | 64,7% | -1,0 pp |
| F1 localizado | 81,9% | 76,7% | -5,2 pp |
| F1 EDF | 96,4% | 94,8% | -1,7 pp |
| FA/h | 0,64 | 0,63 | -0,02 |

A reducao pequena de falsos alarmes nao compensou as perdas de recall,
localizacao e F1 EDF. A pior seed permaneceu em 12/20.

## Analise por evento

- `chb20_13` caiu de deteccao em 4/5 para 2/5 seeds; score medio de 0,987
  para 0,939;
- `chb20_14` caiu de 2/5 para 1/5; score medio de 0,800 para 0,623;
- `chb24_01`, evento 2, melhorou de 3/5 para 4/5;
- os eventos 2-4 de `chb16_17` continuaram sem solucao.

Nas seeds 42 e 43, o checkpoint escolhido ja usava learning rate reduzido para
aproximadamente `0,000063` e `0,000125`. A agenda orientada por PR-AUC de
janela reduziu o passo antes de consolidar todos os eventos. Isso mostra que
menor learning rate nao equivale automaticamente a maior estabilidade por
evento.

## Decisao

**REJEITAR.** A v44 nao atingiu nenhuma das metas principais de promocao e
deve permanecer somente como resultado negativo documentado. A v43 continua
sendo o candidato experimental mais promissor.

## Proximo gargalo

Nao repetir ajustes globais de learning rate, threshold ou duracao. O proximo
experimento deve atacar os eventos persistentemente invisiveis, especialmente
`chb16_17` eventos 2-4, por meio de auditoria de representacao e dados de treino
com morfologia semelhante, mantendo a v43 como baseline congelada.

Artefato: `modelos/v44_late_fusion_stable_calibration.json`.
