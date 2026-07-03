# Estado atual da IA

Atualizado em: 2026-07-03

## Resumo

A IA esta integrada no sistema, mas o modelo ainda nao deve ser tratado como versao final.

Status por camada:

| Camada | Estado |
| --- | --- |
| Backend | Integrado com modelo legado e modelo sequencial |
| Frontend | Exibe informacoes do modelo sequencial |
| API | Retorna campos extras de trecho, score e metadados |
| Fallback | Se o modelo sequencial faltar, usa fluxo legado |
| Testes | Testes principais passaram |
| Qualidade do modelo | Ainda insuficiente para producao clinica |

## Como o fluxo funciona hoje

1. O usuario envia/seleciona um EDF.
2. O backend processa o exame.
3. Se `AI_MODEL_TYPE=sequence_cnn_lstm` e os artefatos existem, usa o modelo sequencial.
4. Se faltar modelo/scaler/metadata, o backend cai para o fluxo legado.
5. O pipeline extrai janelas do EEG.
6. O modelo gera scores por sequencia.
7. O sistema agrega sequencias acima do threshold em trechos suspeitos.
8. A API retorna score geral, trecho principal, top trechos e metadados.
9. O frontend mostra esses dados no painel de laudo.

## Arquivos principais do backend

- `app/config.py`: configuracoes da IA e variaveis de ambiente.
- `app/ai_engine/sequence_inference.py`: carregamento e inferencia do modelo sequencial.
- `app/ai_engine/feature_extractor.py`: extracao de features.
- `app/ai_engine/training.py`: construcao de dataset, janelas e treino.
- `app/services/exame_pipeline.py`: decide entre fluxo sequencial e legado.
- `app/api/routes/exames.py`: rota de diagnostico.
- `app/schemas/exame.py`: schema de resposta do diagnostico.

## Arquivos principais do frontend

- `frontend/src/components/laudo/IaLaudoPanel.tsx`: painel do laudo da IA.
- `frontend/src/pages/VisualizadorClinico.tsx`: repassa dados para o painel.
- `frontend/src/types/api.ts`: tipos da resposta da API.

## Campos novos na resposta da API

A resposta de diagnostico pode incluir:

- `model_type`
- `trecho_suspeito`
- `top_trechos_suspeitos`
- `janela_pico`
- `n_sequences_analisadas`
- `threshold`
- `min_duration_seconds`
- `score_agregacao`
- `feature_mode`
- `canais_processados`
- `canais_omitidos`

## Variaveis de ambiente relevantes

Exemplo para ativar o modelo sequencial:

```env
AI_MODEL_TYPE=sequence_cnn_lstm
AI_SEQUENCE_MODEL_PATH=C:/Users/Foco/Documents/DEV/eeg-xai/modelos/sequence_cnn_lstm_tf_per_channel_localized_w1.keras
AI_SEQUENCE_CALIBRATION_PATH=C:/Users/Foco/Documents/DEV/eeg-xai/modelos/calibration_sequence_cnn_lstm_tf_per_channel_localized_w1_balanced.json
```

Variaveis importantes:

- `AI_MODEL_TYPE`: `legacy` ou `sequence_cnn_lstm`.
- `AI_SEQUENCE_MODEL_PATH`: caminho do `.keras`.
- `AI_SEQUENCE_SCALER_PATH`: caminho do scaler. Pode ser derivado do nome do modelo.
- `AI_SEQUENCE_METADATA_PATH`: metadata do modelo. Pode ser derivado do nome do modelo.
- `AI_SEQUENCE_CALIBRATION_PATH`: calibracao com threshold/duracao.
- `AI_SEQUENCE_DEFAULT_THRESHOLD`: fallback quando nao ha calibracao.
- `AI_SEQUENCE_DEFAULT_MIN_DURATION_SECONDS`: fallback de duracao minima.
- `AI_SEQUENCE_TOP_SEGMENTS_LIMIT`: quantidade de trechos suspeitos retornados.

## O que significa o resultado

O modelo retorna apoio clinico, nao diagnostico final.

Interpretacao correta:

- score alto indica trecho suspeito;
- trecho suspeito indica janela temporal que merece revisao;
- top trechos ajudam a revisar outras regioes do exame;
- o medico deve confirmar visualmente no EEG;
- o sistema nao deve dizer sozinho que o paciente "tem epilepsia".

## Melhor modelo recomendado atualmente

Com base nos testes registrados, o melhor candidato conhecido para localizacao temporal e:

```text
sequence_cnn_lstm_tf_per_channel_localized_w1
```

Motivo:

- melhor F1 localizado que o modelo mais recente expandido;
- menos falsos positivos que o novo modelo expandido;
- menos erros de localizacao.

O modelo expandido `expanded_20260702` nao deve ser promovido ainda.

