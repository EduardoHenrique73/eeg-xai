# v57: influencia dos pacientes na loss temporal

Protocolo registrado em 20/10/2026, antes dos resultados desta rodada.

## Hipotese

A v43 ja possui sampler ictal por evento, peso de borda e selecao de checkpoint
por evento. A v55/v56 mostrou que acrescentar negativos dificeis ao primeiro
estagio nao melhorou a calibracao. Esta rodada altera somente a influencia
relativa de cada paciente na loss, separadamente para passos normais e ictais.

Para cada classe, calcula-se a soma dos pesos temporais por pessoa. O fator
de ajuste e proporcional a raiz quadrada do inverso dessa soma (expoente 0,5).
Os fatores sao normalizados para preservar o peso total original de cada
classe. CHB01 e CHB21 sao agrupados como uma unica pessoa.

Isso reduz dominio de pacientes com mais passos ponderados sem duplicar
sequencias, acrescentar EDFs ou alterar a proporcao global da loss das classes.
A possibilidade de melhorar a generalizacao e uma hipotese a ser testada.

## Controles

- Arquitetura, scaler, features, sampler, bordas e BCE da v43.
- Manifesto v38: 77 EDFs de treino e 13 de calibracao.
- Calibracao: CHB16, CHB19, CHB20 e CHB24.
- CHB09, CHB15 e CHB18 nao sao carregados nesta rodada.
- 20 epocas no maximo; checkpoint por evento sob FA/h <= 0,75.
- Comparacao inicial pareada com seed 42: expoente 0 versus 0,5.
- Se houver ganho, confirmar com seeds 43 e 44 antes de decidir.
- O controle seed 42 foi retreinado nesta rodada. Seus 22 arrays de pesos sao
  exatamente iguais aos da v43 final (diferenca absoluta maxima 0,0).
- Para seeds 43 e 44, a referencia e o relatorio v43 ja registrado no mesmo
  protocolo. Defaults antigos ausentes no JSON sao explicitados pelo comparador
  somente para o experimento identificado `v43_late_channel_fusion`; sua origem
  e a lista de defaults ficam registradas no novo resultado.
- Modelos sao salvos em novos caminhos; o modelo ativo e o TCC nao sao substituidos.

## Regra de decisao

Comparar o ponto FROC no teto 0,75 de ambos os modelos (nao confundir com o
campo calibration_best do script, que usa teto 1,0). Exigir ao menos o mesmo
numero de eventos e F1 EDF, com melhora de eventos, localizacao ou FA/h sem
piora relevante das demais metricas. Um ganho isolado na seed 42 e exploratorio.
Sem confirmacao em seeds adicionais, nao promover. A confirmacao exige ganho
em pelo menos duas das tres seeds, sem reduzir as medias de eventos, F1
localizado e F1 EDF, sob o teto 0,75. Mesmo uma confirmacao na calibracao nao
comprova melhora de generalizacao.

Resultados historicos anteriores (como 7/9) nao sao comparacoes pareadas com
a v43 final: envolvem diferentes checkpoints, seeds e agregacoes. A queda
para 4/9 nao permite atribuir uma causa unica, nem comprova que evitar consultas
ao historico por si so causou a queda.

## Reproducao

Usar `scripts/experiment_expanded_train_v37.py` com arquitetura
`late_channel_fusion`, `--event-balanced-sampling`, `--checkpoint-monitor event_recall`,
`--checkpoint-fa-cap 0.75`, manifesto v38 e 20 epocas.

Comparar `--patient-balance-exponent 0` com `--patient-balance-exponent 0.5`.
O JSON de cada execucao inclui a auditoria da massa da loss por paciente e classe.

## Resultados

Rodada concluida em 20/10/2026: uma reproducao da baseline e tres treinos da
candidata. O experimento nao alterou o modelo ativo nem as metricas oficiais do TCC.

Resultado completo inicial, seed 42:

| Metrica de calibracao (teto 0,75 FA/h) | V43 reproduzida | V57 |
|---|---:|---:|
| Eventos | 15/20 | 15/20 |
| F1 evento | 71,4% | 71,4% |
| F1 localizado | 90,9% | 90,9% |
| F1 EDF | 95,7% | 100,0% |
| FA/h | 0,549 | 0,549 |
| Epoca selecionada | 14 | 6 |
| Threshold / duracao / gap | 0,60 / 18 / 8 | 0,30 / 26 / 8 |

A candidata recuperou `chb20_14`, mas perdeu o terceiro evento de `chb24_04`.
O ganho de F1 EDF nesta seed nao representa aumento do total de crises encontradas.

### Confirmacao nas tres seeds

| Seed | Modelo | Eventos | F1 localizado | F1 EDF | FA/h |
|---|---|---:|---:|---:|---:|
| 42 | v43 | 15/20 | 90,9% | 95,7% | 0,549 |
| 42 | v57 | 15/20 | 90,9% | 100,0% | 0,549 |
| 43 | v43 | 14/20 | 85,7% | 100,0% | 0,628 |
| 43 | v57 | 14/20 | 80,0% | 95,7% | 0,628 |
| 44 | v43 | 13/20 | 73,7% | 95,7% | 0,706 |
| 44 | v57 | 13/20 | 80,0% | 95,7% | 0,471 |
| Media | v43 | 14,00/20 | 83,44% | 97,10% | 0,628 |
| Media | v57 | 14,00/20 | 83,64% | 97,10% | 0,549 |

O F1 de evento medio passou de 66,67% para 68,25%. Houve reducao media de
12,5% em FA/h. A melhora media de localizacao foi apenas 0,20 ponto percentual.
Nao houve aumento do total de eventos em nenhuma seed pareada.

A candidata cumpriu o criterio para uma verificacao historica exploratoria:
ganho sem perdas nas metricas principais em duas seeds, preservando as medias
de eventos, F1 localizado e F1 EDF sob o teto. A seed 42 foi escolhida somente
pela calibracao e congelada antes dessa verificacao. O manifesto registra os
hashes do modelo, scaler, metadados e calibracao. Seu ponto e 0,30 / 26 s / 8 s;
o ponto oficial da v43 continua 0,60 / 18 s / 8 s.

### Verificacao historica unica apos o congelamento

| Metrica | V43 seed 42 | V57 seed 42 |
|---|---:|---:|
| Precisao EDF | 87,5% | 71,4% |
| Sensibilidade EDF | 77,8% | 55,6% |
| F1 EDF | 82,4% | 62,5% |
| FP EDF | 1 | 2 |
| FN EDF | 2 | 4 |
| Eventos detectados | 4/9 | 4/9 |
| Precisao de evento | 11,1% | 20,0% |
| Recall de evento | 44,4% | 44,4% |
| F1 de evento | 17,8% | 27,6% |
| F1 localizado | 46,2% | 53,3% |
| Principais corretamente localizados | 3/9 | 4/9 |
| Falsos alarmes | 32 | 16 |
| FA/h | 2,076 | 1,038 |
| IoU media dos eventos detectados | 73,7% | 72,8% |
| Erro absoluto medio de inicio | 3,0 s | 16,0 s |
| Erro absoluto medio de fim | 23,5 s | 13,0 s |
| Latencia media de inicio do candidato | 3,0 s | 15,5 s |

A latencia acima e retrospectiva (inicio do intervalo previsto menos inicio
anotado, limitado a zero), nao atraso de um detector em tempo real. IoU e erros
medios consideram somente os quatro eventos detectados.

### Comparacao por crise historica

| EDF | V43 | V57 | IoU v43 | IoU v57 | Inicio v57 - real (s) |
|---|---|---|---:|---:|---:|
| chb09_06 | detectada | detectada | 1,000 | 0,971 | 0 |
| chb09_19 | detectada | detectada | 0,844 | 0,970 | -2 |
| chb15_06 | perdida | perdida | 0 | 0 | -- |
| chb15_10 | perdida | perdida | 0 | 0 | -- |
| chb15_17 | perdida | perdida | 0 | 0 | -- |
| chb15_20 | perdida | perdida | 0 | 0 | -- |
| chb15_31 | detectada | detectada | 0,361 | 0,393 | 62 |
| chb18_29 | detectada | detectada | 0,743 | 0,578 | 0 |
| chb18_30 | perdida | perdida | 0 | 0 | -- |

Os mesmos quatro eventos foram encontrados. Em `chb15_31`, um candidato que
era secundario passou a ser o principal, explicando a melhora do F1 localizado,
mas o inicio previsto ficou 62 s atrasado. Em `chb09_19`, a sobreposicao cresceu
de 54 s para 64 s. Em `chb18_29`, o intervalo ficou mais longo que a referencia.

A queda do F1 EDF exige interpretacao: `chb15_17` tinha sete alarmes sem crise
correspondente e `chb18_30` tinha um. A v57 removeu esses alarmes, mas como os
arquivos contem crise real, eles passaram de positivos por EDF para negativos,
mesmo sem perder um evento que a v43 detectava. Alem disso, `chb18_02`, normal,
passou a receber um alarme indevido. Portanto, o F1 EDF isolado nao resume a
utilidade temporal da mudanca; a queda e real, mas nao significa perda de duas
crises anteriormente localizadas.

### Diagnostico e decisao final

**PROMISSOR PARA REDUZIR ALARMES, MAS NAO SUPERIOR EM SENSIBILIDADE.**

- A v57 reduziu pela metade os alarmes historicos e melhorou a escolha do
  candidato principal em um arquivo.
- Nao recuperou nenhuma das cinco crises historicas perdidas.
- `chb15_10` continuou com score ictal medio baixo: 0,056 na v43 e 0,078 na v57.
- `chb18_30` piorou de score ictal medio 0,083 para 0,057.
- Nao foi feita nova selecao de threshold, duracao, seed ou pesos apos o teste
  historico. Todos os resultados historicos continuam exploratorios.
- A v43 permanece ativa e a v57 fica disponivel apenas como artefato experimental.

A ponderacao por paciente isolada nao resolveu a representacao das crises
dificeis. O proximo estudo deve buscar novos candidatos dentro dos eventos,
avaliando representacao/dados de treino com controle pareado e verificacao de
continuidade ictal. Repetir a ponderacao ou reduzir o limiar a partir destes
pacientes historicos nao constitui evidencia de generalizacao.

Verificacao tecnica: 21 testes focados aprovados. Os artefatos oficiais da v43
permanecem com os hashes originais. Nenhuma configuracao de producao foi alterada.

## Auditoria previa dos dados carregados

- 4.648 sequencias de treino, preservando as 952 sequencias positivas da v43.
- 17 grupos CHB-MIT de treino correspondem a 16 pessoas ao unir CHB01/CHB21.
- CHB08 contribui com 13,56% e CHB13 com 12,79% da massa ictal ponderada original.
- CHB06 contribui com 1,06% dessa massa.
- Com expoente 0,5, essas participacoes passam para 9,59%, 9,32% e 2,68%.
- A soma de pesos normais e a soma de pesos ictais sao preservadas separadamente.

## Arquivos desta rodada

- `scripts/patient_balancing.py`: ponderacao e auditoria.
- `scripts/compare_patient_balancing.py`: verificacao pareada, tabela por evento e FROC.
- `modelos/v57_baseline_seed42_training.json`: reproducao de controle.
- `modelos/v57_patient_balanced_seed42_training.json`: candidata inicial.
- `modelos/v57_patient_balancing_comparison/`: comparacao e figura derivadas.
- `modelos/v57_patient_balanced_seed43_training.json`: confirmacao seed 43.
- `modelos/v57_patient_balanced_seed44_training.json`: confirmacao seed 44.
- `modelos/v57_patient_balancing_comparison/selected_candidate_manifest.json`:
  selecao e hashes registrados antes da verificacao historica.
- `modelos/v57_patient_balanced_seed42_historical_eval.json`: verificacao unica.

## Comandos principais

```powershell
.venv\Scripts\python.exe scripts\compare_patient_balancing.py --baseline modelos\v43_late_channel_fusion_calibration.json --candidate modelos\v57_patient_balanced_seed42_training.json modelos\v57_patient_balanced_seed43_training.json modelos\v57_patient_balanced_seed44_training.json --output-dir modelos\v57_patient_balancing_comparison
.venv\Scripts\python.exe scripts\evaluate_frozen_v43_reserved.py --model-path modelos\v57_patient_balanced_seed42.keras --output modelos\v57_patient_balanced_seed42_historical_eval.json
.venv\Scripts\python.exe -m pytest tests\test_patient_balancing.py tests\test_compare_patient_balancing.py tests\test_event_checkpoint.py -q
```
