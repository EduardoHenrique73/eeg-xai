# Comandos e arquivos importantes

Atualizado em: 2026-07-03

## Comandos de treino e avaliacao

### K-Fold simples inicial

```powershell
.venv\Scripts\python.exe scripts\simple_kfold_windows.py
```

Resultado exemplo registrado:

```text
accuracy: 0.9281
precision: 0.7667
recall: 0.7778
f1: 0.7534
```

Observacao: usado apenas como validacao inicial. Nao substitui teste por paciente.

### Construir manifests por paciente

```powershell
.venv\Scripts\python.exe scripts\build_patient_split_manifests.py --prefix expanded_20260702_patient_split --calibration-patients chb06,chb17,chb21,chb22,chb23 --test-patients chb03,chb11,chb13,chb14
```

Saida esperada:

```text
dataset_amostra/manifests/expanded_20260702_patient_split_train.txt
dataset_amostra/manifests/expanded_20260702_patient_split_calibration.txt
dataset_amostra/manifests/expanded_20260702_patient_split_test.txt
dataset_amostra/manifests/expanded_20260702_patient_split_summary.json
```

### Treinar modelo sequencial expandido

```powershell
.venv\Scripts\python.exe scripts\train_sequence_cnn_lstm.py --manifest dataset_amostra\manifests\expanded_20260702_patient_split_train.txt --holdout-files chb06_01.edf chb06_02.edf chb17a_03.edf chb17a_05.edf chb21_01.edf chb21_02.edf chb21_19.edf chb21_20.edf chb22_01.edf chb22_20.edf chb23_06.edf chb23_07.edf --feature-mode time_frequency_per_channel --channel-reference-edf chb01_01.edf --max-normal-windows-per-file 96 --max-seizure-windows-per-file 32 --max-holdout-normal-windows-per-file 144 --max-holdout-seizure-windows-per-file 48 --epochs 10 --batch-size 16 --class-weight-mode manual --positive-class-weight 1.0 --early-stopping-patience 3 --threshold 0.5 --min-duration-seconds 30 --model-output modelos\sequence_cnn_lstm_tf_per_channel_expanded_20260702.keras --metrics-output modelos\metrics_sequence_cnn_lstm_tf_per_channel_expanded_20260702.json
```

### Calibrar modelo expandido

```powershell
.venv\Scripts\python.exe scripts\calibrate_sequence_cnn_lstm.py --model-path modelos\sequence_cnn_lstm_tf_per_channel_expanded_20260702.keras --files chb06_01.edf chb06_02.edf chb17a_03.edf chb17a_05.edf chb21_01.edf chb21_02.edf chb21_19.edf chb21_20.edf chb22_01.edf chb22_20.edf chb22_25.edf chb23_06.edf chb23_07.edf --max-normal-windows-per-file 144 --max-seizure-windows-per-file 48 --thresholds "0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9" --durations "18,20,26,30,45,60,90,120" --max-suspicious-coverage 0.7 --min-overlap-seconds 1 --min-overlap-ratio 0.25 --selection-mode balanced --output modelos\calibration_sequence_cnn_lstm_tf_per_channel_expanded_20260702_balanced.json
```

Melhor regra encontrada na calibracao:

```text
threshold: 0.7
min_duration_seconds: 26
```

### Testar regra fixa no teste separado

```powershell
.venv\Scripts\python.exe scripts\calibrate_sequence_cnn_lstm.py --model-path modelos\sequence_cnn_lstm_tf_per_channel_expanded_20260702.keras --files chb03_01.edf chb03_02.edf chb03_05.edf chb11_01.edf chb11_02.edf chb11_82.edf chb11_92.edf chb13_02.edf chb13_19.edf chb13_21.edf chb13_22.edf chb13_40.edf chb13_47.edf chb13_55.edf chb13_58.edf chb13_59.edf chb13_60.edf chb13_62.edf chb14_01.edf chb14_03.edf chb14_04.edf chb14_06.edf chb14_11.edf chb14_17.edf chb14_18.edf --max-normal-windows-per-file 144 --max-seizure-windows-per-file 48 --thresholds "0.7" --durations "26" --max-suspicious-coverage 0.7 --min-overlap-seconds 1 --min-overlap-ratio 0.25 --selection-mode balanced --output modelos\promoted_sequence_tf_per_channel_expanded_20260702_test_limited.json
```

### Testes automatizados principais

```powershell
.venv\Scripts\python.exe -m pytest tests\test_training_pipeline.py tests\test_sequence_inference.py tests\test_exame_pipeline.py -q
```

Resultado registrado:

```text
32 passed, 1 warning
```

## Arquivos de modelo importantes

### Melhor candidato atual

```text
modelos/sequence_cnn_lstm_tf_per_channel_localized_w1.keras
modelos/sequence_cnn_lstm_tf_per_channel_localized_w1_metadata.json
modelos/sequence_cnn_lstm_tf_per_channel_localized_w1_scaler.pkl
modelos/calibration_sequence_cnn_lstm_tf_per_channel_localized_w1_balanced.json
modelos/promoted_sequence_tf_per_channel_localized_w1_test_limited.json
```

### Candidato expandido nao promovido

```text
modelos/sequence_cnn_lstm_tf_per_channel_expanded_20260702.keras
modelos/sequence_cnn_lstm_tf_per_channel_expanded_20260702_metadata.json
modelos/sequence_cnn_lstm_tf_per_channel_expanded_20260702_scaler.pkl
modelos/metrics_sequence_cnn_lstm_tf_per_channel_expanded_20260702.json
modelos/calibration_sequence_cnn_lstm_tf_per_channel_expanded_20260702_balanced.json
modelos/promoted_sequence_tf_per_channel_expanded_20260702_test_limited.json
```

## Configuracao recomendada atual

Para usar o melhor candidato conhecido:

```env
AI_MODEL_TYPE=sequence_cnn_lstm
AI_SEQUENCE_MODEL_PATH=C:/Users/Foco/Documents/DEV/eeg-xai/modelos/sequence_cnn_lstm_tf_per_channel_localized_w1.keras
AI_SEQUENCE_CALIBRATION_PATH=C:/Users/Foco/Documents/DEV/eeg-xai/modelos/calibration_sequence_cnn_lstm_tf_per_channel_localized_w1_balanced.json
```

Observacao:

Se `AI_SEQUENCE_SCALER_PATH` e `AI_SEQUENCE_METADATA_PATH` ficarem vazios, o sistema tenta derivar do caminho do `.keras`, desde que os nomes sigam o padrao.

## Arquivos de codigo importantes

Backend:

```text
app/config.py
app/ai_engine/sequence_inference.py
app/ai_engine/feature_extractor.py
app/ai_engine/training.py
app/services/exame_pipeline.py
app/api/routes/exames.py
app/schemas/exame.py
```

Frontend:

```text
frontend/src/components/laudo/IaLaudoPanel.tsx
frontend/src/pages/VisualizadorClinico.tsx
frontend/src/types/api.ts
```

Scripts:

```text
scripts/download_chbmit_balanced.py
scripts/build_patient_split_manifests.py
scripts/train_sequence_cnn_lstm.py
scripts/calibrate_sequence_cnn_lstm.py
scripts/evaluate_promoted_sequence_holdouts.py
scripts/validate_sequence_by_patient.py
scripts/validate_tabular_by_patient.py
```

## Cuidados

- Nao avaliar modelo no mesmo paciente usado no treino.
- Nao escolher threshold usando o conjunto de teste.
- Nao promover modelo baseado apenas em F1 geral.
- Sempre conferir F1 localizado.
- Sempre conferir falsos positivos e falsos negativos.
- Sempre registrar pacientes usados em treino, calibracao e teste.

