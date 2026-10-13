# Resultados v55-v56: hard negatives no primeiro estagio

## Hipotese

Reduzir falsos alarmes da v43 adicionando sequencias normais de treino que o modelo congelado classificava com score alto. Os pacientes historicamente reservados `chb09`, `chb15` e `chb18` nao foram carregados pela mineracao nem pelo treino.

O minerador:

- aceita apenas sequencias cujo alvo temporal completo e normal;
- usa apenas EDFs do split de treino;
- ordena pelo maior score temporal da v43;
- exige separacao minima de 30 s para evitar duplicar o mesmo alarme;
- limita exemplos por EDF e paciente;
- registra arquivo, paciente e score de origem.

Arquitetura, weighted BCE, amostragem ictal por evento, calibracao e criterio de checkpoint permaneceram iguais aos da v43.

## v55: configuracao forte

- threshold de mineracao: 0,80;
- ate 8 hard negatives por EDF;
- ate 40 por paciente;
- peso adicional: 1,5;
- 454 sequencias mineradas;
- seeds 42, 43 e 44.

Media de calibracao:

| Metrica | v43 (5 seeds) | v55 (3 seeds) |
|---|---:|---:|
| Eventos detectados | 13,8/20 | 12,67/20 |
| F1 de evento | 65,7% | 62,3% |
| F1 localizado | 81,9% | 71,3% |
| F1 EDF | 96,4% | 90,9% |
| FA/h | 0,64 | 0,63 |

O peso forte reduziu sensibilidade sem produzir reducao relevante de falsos alarmes. Resultado: **REJEITAR**.

## v56: configuracao leve

- threshold de mineracao: 0,80;
- ate 4 hard negatives por EDF;
- ate 20 por paciente;
- sem peso adicional;
- 252 sequencias mineradas;
- seed exploratoria 42.

| Metrica | v43 seed 42 | v56 seed 42 |
|---|---:|---:|
| Eventos detectados | 15/20 | 15/20 |
| F1 de evento | 71,4% | 71,4% |
| F1 localizado | 90,9% | 85,7% |
| F1 EDF | 95,7% | 95,7% |
| FA/h | 0,55 | 0,55 |

A configuracao leve recuperou a sensibilidade, mas nao reduziu alarmes e piorou localizacao. Como nao superou a baseline nem na calibracao, nao foram executadas seeds adicionais nem nova consulta ao reservado historico. Resultado: **REJEITAR**.

## Conclusao

Hard negatives adicionados diretamente ao primeiro estagio nao resolveram o gargalo. A versao forte tornou o detector conservador demais; a versao leve apenas reproduziu a classificacao da v43 com localizacao inferior.

A v43 seed 42 permanece ativa. O proximo experimento deve manter o primeiro estagio congelado e testar um confirmador de candidatos separado. Isso permite rejeitar falsos eventos usando duracao, estabilidade, distribuicao entre canais e evolucao espectral sem enfraquecer a representacao ictal da CNN-BiLSTM.

Artefatos:

- `modelos/v55_hard_negative_event_balanced_calibration.json`;
- `modelos/v56_hard_negative_light_seed42_calibration.json`.
