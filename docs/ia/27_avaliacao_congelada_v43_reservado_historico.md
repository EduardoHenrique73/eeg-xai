# Avaliacao congelada da v43 no reservado historico

## Status metodologico

O modelo `v43_final_seed42` foi avaliado com o ponto operacional congelado da calibracao: threshold `0,60`, duracao minima `18 s` e fechamento de lacunas de `8 s`. Nenhum parametro foi escolhido ou alterado olhando esta execucao.

Os pacientes `chb09`, `chb15` e `chb18`, entretanto, ja haviam sido consultados em experimentos anteriores. O resultado abaixo e um teste de desenvolvimento historico comparavel, nao uma avaliacao final intocada.

## Resultado

| Nivel | Metrica | Resultado |
|---|---|---:|
| EDF | Precisao | 87,5% |
| EDF | Sensibilidade | 77,8% |
| EDF | F1 | 82,4% |
| Temporal | F1 localizado | 46,2% |
| Evento | Eventos detectados | 4/9 |
| Evento | Precisao | 11,1% |
| Evento | Sensibilidade | 44,4% |
| Evento | F1 | 17,8% |
| Evento | Falsos alarmes | 32 |
| Evento | FA/h | 2,08 |
| Temporal | IoU medio nos detectados | 73,7% |
| Temporal | Erro absoluto medio de inicio | 3,0 s |
| Temporal | Erro absoluto medio de fim | 23,5 s |

## Comparacao por crise

| Arquivo | Intervalo real | Resultado |
|---|---:|---|
| `chb09_06` | 12230-12296 s | detectada, IoU 1,00 |
| `chb09_19` | 5298-5362 s | detectada, IoU 0,84 |
| `chb15_06` | 270-398 s | perdida |
| `chb15_10` | 1080-1114 s | perdida |
| `chb15_17` | 1924-1962 s | perdida; falso trecho principal em 1058-1106 s |
| `chb15_20` | 606-664 s | perdida; falso trecho principal em 0-18 s |
| `chb15_31` | 1750-1872 s | detectada em candidato secundario, IoU 0,36 |
| `chb18_29` | 3476-3528 s | detectada, IoU 0,74 |
| `chb18_30` | 540-572 s | perdida; falso trecho principal em 2922-2986 s |

O EDF normal `chb15_01` concentrou dez falsos alarmes. Os EDFs `chb15_17`, `chb15_31` e `chb15_20` tambem geraram varios runs normais sustentados. O problema combina baixa representacao de algumas crises com alta resposta a padroes interictais do paciente 15.

## Controle do artefato

O mesmo arquivo `.keras` foi reavaliado nos pacientes de calibracao `chb16`, `chb19`, `chb20` e `chb24` e reproduziu exatamente:

- F1 EDF: 95,7%;
- F1 localizado: 90,9%;
- eventos: 15/20;
- F1 de evento: 71,4%;
- FA/h: 0,55.

Isso descarta corrupcao dos pesos ou erro causado pela conversao da camada serializavel. A diferenca observada e uma falha de generalizacao interpaciente.

## Decisao

Classificacao: **PROMISSOR NA CALIBRACAO, MAS NAO SUFICIENTE PARA GENERALIZACAO**.

Nao se deve recalibrar a v43 em `chb09/chb15/chb18`. O proximo experimento deve atacar os erros usando apenas treino e calibracao: minerar hard negatives de pacientes permitidos com morfologia semelhante, aumentar diversidade de crises curtas e reservar uma nova coorte externa para a avaliacao final.

Artefatos completos:

- `modelos/v43_final_seed42_reserved_historical_eval.json`;
- `modelos/v43_final_seed42_calibration_artifact_verification.json`.
