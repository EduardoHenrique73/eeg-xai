# v57: comparacao pareada na calibracao

Teto FROC: 0.75 FA/h. Pacientes historicos nao carregados.

| Seed | Modelo | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h |
|---|---|---:|---:|---:|---:|---:|
| 42 | baseline | 15/20 | 71.4% | 90.9% | 95.7% | 0.549 |
| 42 | candidate | 15/20 | 71.4% | 90.9% | 100.0% | 0.549 |
| 43 | baseline | 14/20 | 66.7% | 85.7% | 100.0% | 0.628 |
| 43 | candidate | 14/20 | 66.7% | 80.0% | 95.7% | 0.628 |
| 44 | baseline | 13/20 | 61.9% | 73.7% | 95.7% | 0.706 |
| 44 | candidate | 13/20 | 66.7% | 80.0% | 95.7% | 0.471 |

## Pontos escolhidos somente na calibracao

| Seed | Modelo | Epoca | Threshold | Duracao minima (s) | Gap (s) |
|---|---|---:|---:|---:|---:|
| 42 | baseline | 14 | 0.60 | 18 | 8 |
| 42 | candidate | 6 | 0.30 | 26 | 8 |
| 43 | baseline | 14 | 0.30 | 26 | 6 |
| 43 | candidate | 10 | 0.10 | 26 | 6 |
| 44 | baseline | 1 | 0.60 | 26 | 0 |
| 44 | candidate | 6 | 0.50 | 26 | 0 |

## Eventos alterados

| Seed | Arquivo | Evento | Baseline | Candidata | Score max baseline | Score max candidata |
|---|---|---:|---|---|---:|---:|
| 42 | chb20_14.edf | 1 | False | True | 0.2571 | 0.8004 |
| 42 | chb24_04.edf | 3 | True | False | 1.0000 | 1.0000 |
| 43 | chb20_12.edf | 1 | True | False | 0.7210 | 0.6786 |
| 43 | chb20_13.edf | 1 | False | True | 0.9938 | 0.9980 |
| 44 | chb20_13.edf | 1 | True | False | 0.9730 | 0.9962 |
| 44 | chb24_01.edf | 2 | False | True | 0.9951 | 0.9994 |

## Decisao

PROMISSOR MAS NAO SUPERIOR

Melhora sem perdas nas quatro metricas principais: 2 de 3 seeds.

Confirmacao de ganho na calibracao: True.

O resultado nao autoriza alegar generalizacao melhor. A v43 ativa permanece congelada.
