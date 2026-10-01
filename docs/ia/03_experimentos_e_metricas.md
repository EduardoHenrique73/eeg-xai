# Experimentos e metricas

Atualizado em: 2026-07-03

## Principio de avaliacao

O objetivo do projeto nao e apenas acertar se um arquivo tem crise. O objetivo real e:

1. detectar se existe crise;
2. evitar falso positivo em arquivo normal;
3. localizar o trecho temporal correto da crise;
4. generalizar para paciente novo.

Por isso, as metricas mais importantes sao:

- F1 geral por EDF;
- falsos positivos;
- falsos negativos;
- F1 localizado;
- falsos negativos de localizacao;
- predicoes positivas sem sobreposicao com a crise real.

## Dataset local

Estado da base local:

```text
93 EDFs completos em dataset_amostra
```

Manifests relevantes:

- `dataset_amostra/manifests/current_patient_split_train.txt`
- `dataset_amostra/manifests/current_patient_split_calibration.txt`
- `dataset_amostra/manifests/current_patient_split_test.txt`
- `dataset_amostra/manifests/expanded_20260702_patient_split_train.txt`
- `dataset_amostra/manifests/expanded_20260702_patient_split_calibration.txt`
- `dataset_amostra/manifests/expanded_20260702_patient_split_test.txt`
- `dataset_amostra/manifests/fp_reduction_patient_split_train.txt`
- `dataset_amostra/manifests/mixed_calibration_patient_split_train.txt`

Split expandido de 2026-07-02:

```text
Treino:      54 arquivos
Calibracao: 13 arquivos
Teste:      25 arquivos
```

Pacientes de calibracao usados no split expandido:

```text
chb06, chb17, chb21, chb22, chb23
```

Pacientes de teste usados no split expandido:

```text
chb03, chb11, chb13, chb14
```

## Comparacao de familias de features

Arquivo de referencia:

```text
modelos/compare_features_summary_2026-07-01.json
```

### symbolic_mean

```text
EDF accuracy:           0.6650
EDF precision:          0.6810
EDF recall:             0.9688
EDF F1:                 0.7919
Localized accuracy:     0.0682
Localized precision:    0.1250
Localized recall:       0.0938
Localized F1:           0.1071
Window F1:              0.0329
Falsos positivos:       6
Falsos negativos:       1
FN localizacao:         14
Crises localizadas:     3
```

Conclusao: detecta muitos arquivos com crise, mas localiza muito mal.

### time_frequency

```text
EDF accuracy:           0.7256
EDF precision:          0.7893
EDF recall:             0.8750
EDF F1:                 0.8013
Localized accuracy:     0.4659
Localized precision:    0.5417
Localized recall:       0.5625
Localized F1:           0.5076
Window F1:              0.1154
Falsos positivos:       3
Falsos negativos:       4
FN localizacao:         12
Crises localizadas:     5
```

Conclusao: melhor familia testada entre as tres para localizacao temporal.

### raw_signal

```text
EDF accuracy:           0.5590
EDF precision:          0.7143
EDF recall:             0.7188
EDF F1:                 0.6588
Localized accuracy:     0.1039
Localized precision:    0.1250
Localized recall:       0.0417
Localized F1:           0.0625
Window F1:              0.0275
Falsos positivos:       3
Falsos negativos:       6
FN localizacao:         16
Crises localizadas:     1
```

Conclusao: pior desempenho entre os caminhos comparados. Nao e recomendavel insistir no sinal bruto sem mais arquitetura/dados/preprocessamento.

## Modelo `localized_w1`

Artefatos principais:

```text
modelos/sequence_cnn_lstm_tf_per_channel_localized_w1.keras
modelos/calibration_sequence_cnn_lstm_tf_per_channel_localized_w1_balanced.json
modelos/promoted_sequence_tf_per_channel_localized_w1_test_limited.json
```

Metricas no teste separado:

```text
Accuracy geral:          0.7391
Precision geral:         0.8235
Recall geral:            0.8235
F1 geral:                0.8235

Accuracy localizacao:    0.4348
Precision localizacao:   0.7000
Recall localizacao:      0.4118
F1 localizacao:          0.5185

Falsos positivos:        3
Falsos negativos:        3
FN localizacao:          10
```

Conclusao: melhor candidato conhecido ate agora para localizacao temporal.

## Modelo `context_w1`

Foi testado um modelo com contexto temporal.

Resultado conhecido:

```text
Localized F1: 0.25
```

Conclusao: piorou a localizacao. Nao promover.

## Modelo `expanded_20260702`

Objetivo:

- baixar mais EDFs;
- reconstruir manifests por paciente;
- treinar com mais dados;
- calibrar threshold/duracao;
- testar em pacientes separados.

Artefatos:

```text
modelos/sequence_cnn_lstm_tf_per_channel_expanded_20260702.keras
modelos/sequence_cnn_lstm_tf_per_channel_expanded_20260702_metadata.json
modelos/sequence_cnn_lstm_tf_per_channel_expanded_20260702_scaler.pkl
modelos/metrics_sequence_cnn_lstm_tf_per_channel_expanded_20260702.json
modelos/calibration_sequence_cnn_lstm_tf_per_channel_expanded_20260702_balanced.json
modelos/promoted_sequence_tf_per_channel_expanded_20260702_test_limited.json
```

Treino:

```text
Feature mode: time_frequency_per_channel
Epochs configuradas: 10
Early stopping: parou no epoch 4
Sequencias: 777
Classe 0: 660
Classe 1: 117
```

Metricas de sequencia no holdout interno do treino:

```text
Accuracy:  0.7297
Precision: 0.3071
Recall:    0.6325
F1:        0.4134
```

Metricas por EDF no holdout interno do treino:

```text
Accuracy:  0.6667
Precision: 0.6000
Recall:    1.0000
F1:        0.7500
```

Calibracao:

```text
Melhor threshold:            0.7
Melhor duracao minima:       26s
Accuracy calibracao:         0.9231
Precision calibracao:        0.8750
Recall calibracao:           1.0000
F1 calibracao:               0.9333
F1 localizado calibracao:    0.9333
Falsos positivos:            1
Falsos negativos:            0
FN localizacao:              0
```

Teste separado com regra fixa `threshold=0.7` e `duration=26s`:

```text
Accuracy geral:                0.7200
Precision geral:               0.7895
Recall geral:                  0.8333
F1 geral:                      0.8108

Accuracy localizacao:          0.2800
Precision localizacao:         0.5000
Recall localizacao:            0.2222
F1 localizacao:                0.3077

Falsos positivos:              4
Falsos negativos:              3
FN localizacao:                14
Predicoes positivas sem overlap: 11
```

Conclusao:

O modelo pareceu bom na calibracao, mas nao generalizou bem no teste separado. Ele nao deve ser promovido.

## Comparacao final relevante

| Modelo | F1 geral | F1 localizacao | FP | FN | FN localizacao |
| --- | ---: | ---: | ---: | ---: | ---: |
| `localized_w1` | 0.8235 | 0.5185 | 3 | 3 | 10 |
| `expanded_20260702` | 0.8108 | 0.3077 | 4 | 3 | 14 |

Conclusao:

O modelo `expanded_20260702` nao substitui o `localized_w1`.

## Testes automatizados

Ultima bateria principal executada:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_training_pipeline.py tests\test_sequence_inference.py tests\test_exame_pipeline.py -q
```

Resultado mais recente:

```text
106 passed, 5 warnings
```

## Experimento `center_v2` (2026-07-20)

Objetivo: corrigir o alinhamento temporal entre a entrada da CNN-LSTM e a
janela prevista. Cada sequencia continua usando 8 janelas como contexto, mas o
rotulo e o intervalo de saida passam a representar apenas a janela central.

Correcoes aplicadas antes do treino:

- sobreposicao ictal minima de 50% para rotular uma janela como crise;
- janelas de borda tratadas como negativos dificeis;
- amostragem preservando blocos com pelo menos 8 janelas consecutivas;
- validacao de early stopping em pacientes de holdout, sem split aleatorio de janelas;
- pesos de classe e de borda combinados em um unico `sample_weight`;
- cache de dataset por EDF e configuracao de extracao.

Treino:

```text
Pacientes treino:      15
EDFs treino:           54
Sequencias treino:     2230
Sequencias normais:    1904
Sequencias crise:      326
Epocas executadas:     5 (early stopping)
```

Calibracao em 5 pacientes separados:

```text
Threshold:             0.15
Duracao minima:        6s
F1 geral:              0.6667
F1 localizado:         0.6667
FP:                    1
FN:                    3
FN localizacao:        3
```

Teste congelado em 4 pacientes nunca vistos:

```text
Accuracy geral:        0.3600
Precision geral:       0.5833
Recall geral:          0.3889
F1 geral:              0.4667
F1 localizado:         0.0833
FP:                    5
FN:                    11
FN localizacao:        17
Positivos sem overlap: 6
```

Conclusao: a correcao temporal e necessaria, mas o candidato `center_v2` nao
generalizou entre pacientes e nao deve ser promovido. O modelo ativo continua
sendo o `localized_w1`.

| Modelo | F1 geral | F1 localizacao | FP | FN | FN localizacao |
| --- | ---: | ---: | ---: | ---: | ---: |
| `localized_w1` | 0.8235 | 0.5185 | 3 | 3 | 10 |
| `expanded_20260702` | 0.8108 | 0.3077 | 4 | 3 | 14 |
| `center_v2` | 0.4667 | 0.0833 | 5 | 11 | 17 |

## Features relativas e normalizacao robusta (2026-07-20)

Foram avaliados tres refinamentos posteriores ao `center_v2`, mantendo o
mesmo split por paciente e a regra calibrada congelada no teste.

### `relative_center_v3`

- remove potencias absolutas;
- usa potencias relativas, frequencias normalizadas, entropia normalizada e log-razoes;
- F1 geral no teste: `0.7895`;
- F1 localizado: `0.2308`;
- FP: 5; FN: 3; FN localizacao: 15.

Conclusao: melhorou deteccao geral, mas continuou marcando o trecho errado.

### `relative_robust_v4`

- adiciona normalizacao por mediana/IQR de cada EDF;
- F1 geral no teste: `0.4000`;
- F1 localizado: `0.3333`;
- FP: 2; FN: 13; FN localizacao: 14.

Conclusao: reduziu FP, mas a duracao calibrada de 60s perdeu muitas crises.

### `relative_robust_normalref_v5`

- no treino, estima mediana/IQR usando apenas janelas normais rotuladas;
- na inferencia, usa o EDF completo, normalmente dominado por sinal normal;
- calibracao: F1 localizado `0.9231`, FP 0, FN 1;
- teste: F1 geral `0.5333`;
- teste: F1 localizado `0.4828`;
- teste: FP 4, FN 10, FN localizacao 11.

Conclusao: foi o melhor dos novos candidatos para localizacao, mas ainda nao
superou o `localized_w1` (`0.5185`) e piorou FP/FN. Nao promover.

| Modelo | F1 geral | F1 localizacao | FP | FN | FN localizacao |
| --- | ---: | ---: | ---: | ---: | ---: |
| `localized_w1` | 0.8235 | 0.5185 | 3 | 3 | 10 |
| `relative_center_v3` | 0.7895 | 0.2308 | 5 | 3 | 15 |
| `relative_robust_v4` | 0.4000 | 0.3333 | 2 | 13 | 14 |
| `relative_robust_normalref_v5` | 0.5333 | 0.4828 | 4 | 10 | 11 |

As features relativas e a normalizacao robusta permanecem implementadas para
novos experimentos, mas nenhum desses modelos substitui a baseline ativa.

## Reavaliacao e candidatos `v6`/`v7` (2026-09-30)

A baseline `localized_w1` foi recalculada com o avaliador temporal corrigido.
No conjunto de teste ja consultado, o resultado reproduzido foi:

```text
F1 geral:              0.8235
F1 localizado:         0.4000
Recall localizado:     0.2941
FP:                    3
FN:                    3
FN localizacao:        12
```

O valor historico de F1 localizado `0.5185` nao deve mais ser usado como
resultado final. Das 14 predicoes positivas, apenas 5 trechos principais
sobrepuseram uma crise real e 9 nao tiveram sobreposicao.

O candidato focal `v6` foi treinado em 37 EDFs de treino e 10 EDFs de holdout:

```text
Precisao por sequencia: 0.1859
Recall por sequencia:   0.3222
F1 por sequencia:       0.2358
F1 por EDF:             0.5000
```

Conclusao: forte sobreajuste; candidato rejeitado.

Durante a auditoria foi encontrada uma incompatibilidade metodologica na
normalizacao robusta: o treino usava somente janelas normais conhecidas, mas a
inferencia precisa usar todas as janelas, pois nao conhece os rotulos. O treino
e a validacao foram corrigidos para usar todas as janelas amostradas e a chave
de cache foi versionada.

O candidato `v7`, com features tempo-frequencia relativas por canal e
normalizacao robusta consistente, obteve em 13 EDFs de holdout de pacientes
separados:

```text
Precisao por sequencia: 0.5158
Recall por sequencia:   0.4016
F1 por sequencia:       0.4516
Precisao por EDF:       1.0000
Recall por EDF:         0.2857
F1 por EDF:             0.4444
```

Uma busca exploratoria, ainda com janelas amostradas, escolheu threshold `0.5`
e duracao minima `26s`, com precisao localizada `0.75`, recall `0.4286`, F1
`0.5455`, 1 FP, 4 FN e sensibilidade por evento `0.3333`. Esses numeros nao sao
metricas finais e nao autorizam promocao: calibracao e teste completos ainda
precisam ocorrer em pacientes separados e nunca usados na escolha do modelo.

O modelo ativo nao foi substituido.

## Validacao completa `v8`, `v9` e `v10` (2026-09-30)

A extracao multicanal foi otimizada para carregar os canais de referencia uma
unica vez por EDF. Isso tornou viavel avaliar EDFs completos sem alterar as
features. Tambem foi corrigido o early stopping para restaurar o melhor
`val_pr_auc`, em vez do menor `val_loss`.

Em tres pacientes de teste separados (`chb09`, `chb15` e `chb18`), a `v8`
obteve F1 medio por EDF `0.7905`, mas localizou apenas 2 de 9 crises (F1
localizado `0.3333`). A `v9`, com mais negativos, aumentou a precisao do
holdout por sequencia para `0.6667`, mas piorou na validacao completa: F1 por
EDF `0.5556` e apenas 1 de 9 crises localizada. A `v9` foi rejeitada.

Foi entao corrigida outra divergencia entre treino e producao: a normalizacao
robusta era calculada depois da amostragem balanceada no treino, enquanto a
inferencia usa todas as janelas do EDF. A `v10` normaliza com todas as janelas
antes de amostrar e versiona o cache como
`all_edf_windows_before_sampling_v4`.

Com calibracao interpaciente balanceada, a `v10` obteve:

```text
Precisao media por EDF:       0.7667
Recall medio por EDF:         0.7667
F1 medio por EDF:             0.7667
F1 localizado medio:          0.3889
Crises localizadas:           4/9
Falsos positivos:             2
Falsos negativos:             2
Falsos negativos localizacao: 6
```

A calibracao conservadora anterior priorizava zerar falsos positivos antes de
F1 e recall. No `chb15`, isso escolheu threshold `0.5` e produziu F1 `0.2857`.
O modo balanceado escolheu threshold `0.025`, elevando o F1 desse paciente para
`0.8` e as crises localizadas de 1 para 3. O criterio agora e explicito em
`--calibration-selection-mode`.

Conclusao: a `v10` melhorou a localizacao em relacao a `v8`, mas ainda nao
superou o criterio minimo de F1 localizado `0.4000` nem foi validada em um
teste final intocado. Nenhum novo modelo foi promovido.

## Features hibridas e evidencia acumulada `v11` (2026-10-01)

A `v11` voltou a combinar potencias absolutas e relativas por canal, mantendo
a normalizacao robusta calculada no EDF completo. No holdout amostrado, ela
obteve F1 por sequencia `0.7805` e F1 por EDF `0.7273`. Uma calibracao
exploratoria encontrou F1 por EDF `0.9231`, mas esses valores nao sao resultado
final porque usam dados amostrados.

Na validacao interpaciente completa com dois pacientes de calibracao, a `v11`
elevou o recall medio por EDF para `0.9333` e o F1 para `0.8222`, mas ainda
localizou apenas 4 de 9 crises. A auditoria do `chb18_29` mostrou que um trecho
correto de 52 segundos perdia a posicao principal para um pico falso de 12
segundos com media ligeiramente maior.

O ranqueamento foi corrigido para priorizar evidencia acumulada acima do
threshold entre trechos que atingem a duracao minima. Com quatro pacientes de
calibracao por dobra, o resultado foi:

```text
Precisao media por EDF:       0.9333
Recall medio por EDF:         0.7667
F1 medio por EDF:             0.8222
F1 localizado medio:          0.6111
Crises localizadas:           4/9
Falsos positivos:             1
Falsos negativos:             2
Falsos negativos localizacao: 5
```

Esse e o melhor equilibrio interpaciente registrado ate agora, mas nao deve ser
tratado como resultado final: `chb09`, `chb15` e `chb18` ja foram consultados
durante o desenvolvimento. O modelo ativo nao foi substituido.

### Teste de focal alpha `0.65`

Como `BinaryFocalCrossentropy` atribui `alpha` a classe positiva, foi testado
`alpha=0.65` para aumentar o peso das crises. Mantendo features hibridas,
calibracao com quatro pacientes e os demais parametros da `v11`, o resultado
piorou: F1 por EDF `0.7460`, F1 localizado `0.4667`, 2 falsos positivos e 3
falsos negativos. A calibracao compensou o aumento dos scores escolhendo
thresholds mais altos. Essa configuracao foi rejeitada e nao deve substituir a
`v11`.

## Segmentacao temporal `v13` e `v14` (2026-10-01)

Foi implementado um modo `segmentation` que preserva a familia CNN-LSTM, mas
usa CNN-BiLSTM com uma probabilidade para cada passo temporal. As previsoes de
sequencias sobrepostas sao consolidadas por janela antes da calibracao.

A `v13`, com focal loss, obteve F1 por EDF `0.7460` e F1 localizado `0.5778`.
Ela melhorou o `chb09`, mas perdeu tres EDFs com crise no `chb15`.

A `v14` trocou a focal por BCE com pesos de classe calculados no treino:

```text
Precisao media por EDF:       0.7778
Recall medio por EDF:         1.0000
F1 medio por EDF:             0.8586
F1 localizado medio:          0.5222
Crises localizadas principais: 4/9
Falsos positivos:             3
Falsos negativos:             0
Falsos negativos localizacao: 5
```

Esse foi o primeiro experimento com zero falsos negativos no nivel de EDF. Por
evento, 6/9 crises apareceram em algum candidato temporal. Ainda assim, o
`chb15_10`, `chb15_17` e `chb18_30` nao produziram um trecho continuo valido, e
o `chb18` chegou a 6.493 falsos alarmes por hora. A `v14` e promissora para
sensibilidade, mas nao deve ser promovida antes de reduzir fragmentacao e
falsos alarmes em calibracao independente.

## Calibracao por evento e pos-processamento `v15` a `v19` (2026-10-01)

As `v15` e `v16` passaram a calibrar sensibilidade/precisao por evento, unir
lacunas temporais curtas e calcular falsos alarmes por todas as horas nao
ictais, inclusive dentro de EDFs positivos. A `v16` corrigiu o denominador de
falsos alarmes por hora e obteve o melhor equilibrio da segmentacao:

```text
Precisao media por EDF:        0.8889
Recall medio por EDF:          0.9333
F1 medio por EDF:              0.8963
F1 localizado medio:           0.5794
Crises principais localizadas: 4/9
Eventos detectados:             6/9
Falsos positivos por EDF:       1
Falsos negativos por EDF:       1
```

A `v17` equalizou o peso total de crises curtas e longas. O resultado piorou
para F1 por EDF `0.7222`, F1 localizado `0.5556`, 2 FP e 3 FN; por isso o modo
continua opcional e nao foi adotado.

A `v18` adicionou histerese temporal calibravel. Ela selecionou razao `0.6`
somente no `chb15`, eliminando um falso positivo desse fold, mas repetiu o
agregado da `v16`: F1 por EDF `0.8963`, F1 localizado `0.5794` e 4/9 crises
principais localizadas. A histerese foi mantida como ferramenta de calibracao,
nao como ganho comprovado de localizacao.

A `v19` reduziu de `0.50` para `0.25` a sobreposicao ictal minima das janelas,
aumentou o peso das bordas e o limite de exemplos positivos. Isso adicionou
rotulos ambiguos e piorou o teste: F1 por EDF `0.7460`, F1 localizado `0.4667`,
3/9 crises localizadas, 2 FP e 3 FN. A configuracao foi rejeitada.

Nenhuma dessas rodadas promoveu um novo modelo. `chb09`, `chb15` e `chb18`
continuam sendo coorte de desenvolvimento, nao teste final intocado.

## Detector em dois estagios `v20` a `v22` (2026-10-01)

Foi implementado um confirmador Random Forest treinado apenas com candidatos
dos pacientes de calibracao. Candidatos sem overlap real viram hard negatives.
O threshold do confirmador e escolhido com predicoes leave-one-patient-out na
calibracao, e os artefatos registram a curva FROC.

A `v20`, com geracao em score `0.10`, melhorou o `chb09` isoladamente, mas no
agregado detectou 4/9 eventos, com 15 falsos alarmes (`0.9730/h`) e apenas 3
trechos principais localizados. A `v21` baixou o gerador para `0.05`: detectou
3/9 eventos, gerou 19 falsos alarmes (`1.2325/h`) e localizou 2 trechos
principais. A `v22` passou a escolher o ponto de operacao por F1 de evento e
FA/h, mas repetiu o resultado da `v21`.

Conclusao: o segundo estagio e a avaliacao FROC estao funcionais, mas o gerador
atual nao fornece candidatos transferiveis para as crises de `chb15`. Baixar o
threshold aumenta alarmes sem recuperar esses eventos. `v20` a `v22` foram
rejeitadas e nao substituem a `v16/v18`.

## Correcao do split e expansao `v23` a `v25` (2026-10-01)

Uma auditoria mostrou que as avaliacoes historicas `v16/v18` excluiam do treino
somente o paciente da dobra corrente. Assim, ao testar `chb09`, arquivos de
`chb15` e `chb18` ainda podiam participar de treino ou calibracao. Esses valores
foram preservados como historico de desenvolvimento, mas nao sao mais uma
baseline valida para a coorte reservada completa.

A baseline antiga foi recalculada mantendo `chb09`, `chb15` e `chb18` fora de
todas as dobras e usando `chb16`, `chb19`, `chb20` e `chb24` apenas para
calibracao. O resultado justo foi F1 por EDF `0.8056`, F1 localizado `0.5460`,
4/9 crises principais localizadas, 6/9 eventos detectados e `1.1027` falsos
alarmes por hora.

O manifesto expandido passou de 54 para 92 EDFs completos. A `v23b` isolou o
efeito desses novos dados, preservando a mesma coorte de calibracao e os mesmos
hiperparametros:

```text
Precisao media por EDF:        0.8222
Recall medio por EDF:          0.9333
F1 medio por EDF:              0.8667
F1 localizado medio:           0.6667
Crises principais localizadas: 5/9
Eventos detectados:             6/9
Falsos positivos por EDF:       2
Falsos negativos por EDF:       1
Falsos alarmes por hora:        1.3622
```

A expansao melhorou classificacao e localizacao em relacao a baseline
recalculada, mas aumentou os falsos alarmes. A auditoria mostrou tres causas:
`chb15_10` continua com score baixo durante toda a crise; `chb15_17` e
`chb18_30` agora apresentam scores ictais altos, mas curtos ou fragmentados;
`chb15_20` e detectado, embora o candidato principal possa ficar deslocado.

A `v24` combinou tres CNN-BiLSTM por media de scores. Ela piorou para F1 por EDF
`0.8500`, F1 localizado `0.5000` e 3/9 crises principais localizadas; o ensemble
foi rejeitado porque diluiu eventos reconhecidos por apenas um dos membros.

A `v25` reaplicou o Random Forest confirmador sobre o primeiro estagio da
`v23b`. O primeiro estagio reproduziu a `v23b`, mas o confirmador manteve apenas
6/9 eventos, gerou 27 falsos alarmes (`1.7514/h`), obteve F1 localizado macro
`0.5619` e F1 por EDF macro `0.8222`. Portanto, `v25` foi rejeitada e a `v23b`
permanece como melhor referencia controlada de desenvolvimento. Nenhum desses
artefatos foi promovido para producao.

## Contexto e amostragem por sequencia `v26-v28` (2026-10-01)

A `v26` aumentou o comprimento de 8 para 16 passos, ampliando o contexto de
aproximadamente 18 para 34 segundos. A mudanca reduziu exemplos de crises
curtas e piorou para F1 por EDF `0.8222`, F1 localizado `0.4841`, 3/9 crises
principais, 5/9 eventos e `2.1406` falsos alarmes por hora. Contexto longo unico
foi rejeitado.

Durante esse teste foi identificado que o pipeline limitava janelas antes de
formar sequencias. Muitas janelas selecionadas deixavam de formar blocos
continuos e eram descartadas. Foi adicionado `--sampling-level sequence`, que
forma todas as sequencias validas e aplica os limites por classe depois. O modo
antigo `window` foi mantido como padrao para reproducao historica.

A `v27`, com 8 passos e amostragem por sequencia, aumentou os exemplos validos
sem duplicacao artificial. Ela obteve F1 por EDF `0.7667`, F1 localizado
`0.5556`, 4/9 crises principais e 26 falsos alarmes (`1.6865/h`). O resultado
agregado foi inferior a `v23b`, mas a sensibilidade por evento subiu para 7/9,
contra 6/9 na `v23b`. Portanto, ela e um gerador sensivel promissor, nao um
modelo final.

A `v28` aplicou o confirmador ao gerador da `v27`. O segundo estagio voltou a
6/9 eventos, mas reduziu os falsos alarmes para 9 (`0.5838/h`), com F1
localizado macro `0.6111` e F1 por EDF macro `0.8222`. A calibracao OOF do
confirmador apresentou baixa sensibilidade em algumas dobras, especialmente
`chb15`, e descartou o evento adicional recuperado pela `v27`. A configuracao
nao foi promovida.

## Limite de sequencias normais `v29` (2026-10-01)

Foi criada uma selecao separada, sem carregar os pacientes reservados, para
comparar 32, 48, 64 e 96 sequencias normais por EDF. A seed foi fixada em 42 e
o criterio priorizou F1 e sensibilidade por evento, depois F1 localizado e
falsos alarmes por hora. A calibracao escolheu 48:

| Limite | F1 evento | Sensibilidade | F1 localizado | FA/h |
|---:|---:|---:|---:|---:|
| 32 | 0.522 | 0.600 | 0.588 | 1.099 |
| 48 | 0.600 | 0.600 | 0.737 | 0.628 |
| 64 | 0.526 | 0.500 | 0.667 | 0.628 |
| 96 | 0.564 | 0.550 | 0.737 | 0.628 |

A avaliacao reservada `v29` alterou somente o limite de 96 para 48 em relacao a
`v27`. O resultado foi F1 por EDF `0.8667`, F1 localizado `0.6111`, 5/9 trechos
principais, 7/9 eventos e 20 falsos alarmes (`1.2973/h`). Ela recuperou o F1 por
EDF da `v23b` e preservou a sensibilidade de evento da `v27`, mas nao superou o
F1 localizado da `v23b` nem atingiu a meta de menos de 1 FA/h. A `v29` foi
rejeitada para promocao, embora seja a melhor evidencia ate agora de que reduzir
o excesso de normais melhora o equilibrio do gerador sensivel.
