# v37: dois eventos independentes adicionais no treino

## Objetivo

Medir isoladamente o efeito de ampliar a diversidade ictal do treino, sem
alterar arquitetura, features, loss, pos-processamento ou pacientes de
calibracao. Foram adicionados somente ao treino:

- `chb02_19.edf`: crise curta de 9 segundos;
- `chb17a_04.edf`: crise de 115 segundos.

Os arquivos foram validados com MNE e possuem 3.600 segundos. O manifesto novo
e `dataset_amostra/manifests/v37_expanded_train.txt`; o manifesto v23 original
permaneceu inalterado. `chb09`, `chb15` e `chb18` nao foram carregados.

## Comparacao na calibracao fixa

Mesma CNN-BiLSTM, weighted BCE, amostragem por sequencia, seeds 42/43/44 e
criterio FROC com `FA/h <= 1` da baseline v32.

| Seed | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h |
|---:|---:|---:|---:|---:|---:|
| 42 | 11/20 | 55,0% | 73,7% | 90,9% | 0,71 |
| 43 | 14/20 | 65,1% | 70,0% | 87,0% | 0,71 |
| 44 | 13/20 | 59,1% | 73,7% | 90,9% | 0,86 |
| **Media v37** | **12,67/20** | **59,7%** | **72,5%** | **89,6%** | **0,76** |
| Baseline v32 | 12,67/20 | - | 70,2% | 87,9% | 0,89 |

O treino passou de 4.053 para 4.181 sequencias e de 789 para 821 sequencias
positivas. A sensibilidade media por evento nao mudou. Houve melhora simultanea
de localizacao, F1 por EDF e falsos alarmes, mas a seed 42 perdeu dois eventos
em relacao a sua baseline pareada. Portanto, o ganho ainda nao e robusto para
promocao.

## Decisao

**PROMISSOR MAS NAO SUPERIOR.** Nao avaliar no conjunto reservado e nao alterar
o modelo de producao. O resultado sustenta a hipotese de que novos eventos
independentes ajudam mais que adicionar descritores manuais ou uma branch bruta,
mas dois eventos sao insuficientes para aumentar consistentemente o recall.

A auditoria dos summaries encontrou 45 EDFs ictais de pacientes de treino ainda
ausentes, com 73 eventos anotados. O proximo passo e ampliar o treino em lotes
retomaveis, repetir a mesma comparacao e exigir aumento consistente de eventos
nas tres seeds antes de abrir nova avaliacao reservada. Durante esta rodada, o
PhysioNet apresentou transferencia muito lenta; por isso o lote foi encerrado
apos os dois EDFs validados, sem arquivos incompletos no manifesto.

Artefato: `modelos/v37_expanded_train_calibration.json`.
