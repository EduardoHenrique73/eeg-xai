# v58: resultado controlado de dropout por canal

Comparacao somente na calibracao. Teto principal: 0,75 FA/h.

| Seed | Modelo | Eventos | F1 localizado | F1 EDF | FA/h |
|---|---|---:|---:|---:|---:|
| 42 | v43 | 15/20 | 90.9% | 95.7% | 0.549 |
| 42 | v58 | 13/20 | 80.0% | 95.7% | 0.706 |

Checkpoint v58: epoca 16.
Criterio para continuar: NAO ATENDIDO.

| Arquivo | Evento | Detectado v43/v58 | Pico v43 | Pico v58 |
|---|---:|---|---:|---:|
| chb16_10.edf | 1 | True/True | 0.9891 | 0.9939 |
| chb16_11.edf | 1 | False/False | 0.3522 | 0.9866 |
| chb16_17.edf | 1 | True/False | 0.9089 | 0.9736 |
| chb16_17.edf | 2 | False/False | 0.0377 | 0.0958 |
| chb16_17.edf | 3 | False/False | 0.0221 | 0.0015 |
| chb16_17.edf | 4 | False/False | 0.0513 | 0.0959 |
| chb19_28.edf | 1 | True/True | 0.9999 | 1.0 |
| chb19_29.edf | 1 | True/True | 1.0 | 1.0 |
| chb19_30.edf | 1 | True/True | 0.9999 | 1.0 |
| chb20_12.edf | 1 | True/True | 0.9972 | 0.9869 |
| chb20_13.edf | 1 | True/False | 0.9944 | 0.9923 |
| chb20_13.edf | 2 | True/True | 0.9985 | 0.9999 |
| chb20_14.edf | 1 | False/False | 0.2571 | 0.2984 |
| chb24_01.edf | 1 | True/True | 1.0 | 1.0 |
| chb24_01.edf | 2 | True/True | 0.9998 | 0.9997 |
| chb24_03.edf | 1 | True/True | 0.9999 | 1.0 |
| chb24_03.edf | 2 | True/True | 1.0 | 1.0 |
| chb24_04.edf | 1 | True/True | 1.0 | 1.0 |
| chb24_04.edf | 2 | True/True | 1.0 | 1.0 |
| chb24_04.edf | 3 | True/True | 1.0 | 1.0 |

FROC mostra pontos selecionados da grade de calibracao; nao e validacao externa.
Nenhum modelo foi promovido automaticamente.
