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

Resultado:

```text
32 passed, 1 warning
```

