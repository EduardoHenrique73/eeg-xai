# v57: comparacao pareada na calibracao

Teto FROC: 0.75 FA/h. Pacientes historicos nao carregados.

| Seed | Modelo | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h |
|---|---|---:|---:|---:|---:|---:|
| 42 | baseline | 15/20 | 71.4% | 90.9% | 95.7% | 0.549 |
| 42 | candidate | 15/20 | 71.4% | 90.9% | 100.0% | 0.549 |

## Eventos alterados

| Seed | Arquivo | Evento | Baseline | Candidata | Score max baseline | Score max candidata |
|---|---|---:|---|---|---:|---:|
| 42 | chb20_14.edf | 1 | False | True | 0.2571 | 0.8004 |
| 42 | chb24_04.edf | 3 | True | False | 1.0000 | 1.0000 |

## Decisao

PROMISSOR MAS NAO SUPERIOR

Melhora sem perdas nas quatro metricas principais: 1 de 1 seeds.

O resultado nao autoriza alegar generalizacao melhor. A v43 ativa permanece congelada.
