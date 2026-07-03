# Historico do trabalho

Atualizado em: 2026-07-03

## Contexto inicial

O projeto `eeg-xai` foi comparado com o TCC informado pelo usuario. O objetivo central do sistema, conforme discutido, e analisar exames EEG em EDF, detectar sinais compativeis com crise epileptica e apoiar o medico com explicabilidade, sem apresentar a saida como diagnostico final automatico.

Pontos citados no TCC e tratados no sistema:

- analise de EEG em janelas;
- uso de rede neural para detectar padroes de crise;
- explicabilidade por SHAP;
- configuracoes clinicas, como threshold de confianca;
- configuracao de canais/montagem;
- validacao com K-Fold ou separacao controlada;
- uso do dataset CHB-MIT;
- comparacao entre resultados preditivos e anotacoes reais de crise.

## Primeiras correcoes de sistema

Foram implementadas ou revisadas partes de backend e frontend para deixar o sistema mais alinhado ao TCC:

- tela de configuracoes;
- threshold de confianca configuravel;
- montagem padrao de canais;
- ativar/desativar SHAP;
- dados de perfil medico;
- ajustes em textos da interface para linguagem clinica cautelosa;
- revisao do painel de laudo da IA;
- exposicao de dados de analise no frontend.

## K-Fold simples

Foi criado/rodado um fluxo simples de K-Fold no Windows:

```powershell
.venv\Scripts\python.exe scripts\simple_kfold_windows.py
```

Exemplo de saida obtida:

```text
accuracy: 0.9281
precision: 0.7667
recall: 0.7778
f1: 0.7534
```

Esse resultado foi considerado apenas uma validacao inicial, porque foi feito com poucos arquivos e nao resolve o problema mais importante: generalizar para paciente novo.

## Download de dados CHB-MIT

Foram baixados varios EDFs do CHB-MIT para ampliar a base local. A pasta `dataset_amostra` chegou a 93 EDFs completos.

O entendimento confirmado foi:

- cada prefixo `chbXX` representa um paciente;
- `chb01_01.edf`, `chb01_03.edf`, etc. sao exames/arquivos do mesmo paciente `chb01`;
- o objetivo real do projeto e detectar crise em paciente novo, entao o teste precisa separar pacientes, nao apenas arquivos aleatorios.

Arquivos baixados em uma das rodadas mais recentes:

- `chb24_03.edf`
- `chb18_30.edf`
- `chb21_20.edf`
- `chb10_20.edf`
- `chb04_08.edf`
- `chb18_02.edf`
- `chb21_02.edf`
- `chb08_05.edf`

Alguns downloads demoraram ou deram timeout. A decisao foi baixar em lotes menores e evitar depender de um unico arquivo lento.

## Evolucao do modelo

No inicio, o sistema ainda dependia de um modelo legado/antigo. Foi discutido que esse modelo vinha de uma estrutura anterior e nao era o ideal para o novo projeto.

Decisao tomada:

- seguir com o projeto novo;
- retreinar modelos dentro da estrutura atual;
- manter fallback legado apenas para nao quebrar o sistema;
- buscar um modelo sequencial CNN-LSTM mais alinhado ao TCC.

## Data augmentation

Foi discutido se o TCC citava data augmentation. O que ficou definido:

- havia referencias a validacao e treinamento, mas augmentation forte nao estava necessariamente detalhado no TCC;
- augmentation mais forte como ruido gaussiano, shift temporal, escala de amplitude e jitter poderia melhorar o modelo, mas deve ser documentado como melhoria adicional caso nao esteja explicitamente no TCC;
- o foco principal passou a ser melhorar a base de EDFs e a separacao treino/calibracao/teste antes de adicionar tecnicas mais agressivas.

## Random Forest x CNN-LSTM

Foi observado que Random Forest poderia performar melhor em alguns testes com features tabulares. A explicacao registrada:

- Random Forest nao usa o sinal bruto inteiro;
- ele usa features extraidas das janelas/canais;
- pode ser mais estavel com pouca base;
- porem o TCC indicava CNN-LSTM, entao a decisao foi manter CNN-LSTM como caminho principal;
- Random Forest pode servir como baseline, nao como substituto principal.

## Sinal bruto, features e dinamica simbolica

Foram comparados tres caminhos:

- `raw_signal`: sinal bruto ou representacao mais direta;
- `symbolic_mean`: features ligadas a dinamica simbolica agregadas por media;
- `time_frequency`: features de tempo-frequencia.

Conclusao:

- sinal bruto foi pior nos testes limitados;
- dinamica simbolica agregada por media tambem apresentou localizacao fraca;
- features de tempo-frequencia foram melhores para localizacao temporal;
- `time_frequency_per_channel` virou o caminho mais promissor, pois preserva informacao por canal.

## Agregacao por canal e score geral

Foi discutido se a media por canal poderia atrapalhar o medico.

Conclusao:

- um score geral ainda e util como triagem;
- mas ele nao deve esconder canais especificos;
- a interface deve exibir trechos suspeitos, score, canais e contexto;
- canais normais nao devem "diluir" completamente um canal suspeito;
- por isso foi valorizado o caminho `per_channel` e o retorno de trechos/top trechos.

## Integracao do modelo sequencial

Foi implementado um fluxo sequencial CNN-LSTM de ponta a ponta:

- configuracao em `app/config.py`;
- inferencia sequencial em `app/ai_engine/sequence_inference.py`;
- branch no pipeline em `app/services/exame_pipeline.py`;
- schemas expandidos em `app/schemas/exame.py`;
- rota de diagnostico ajustada;
- frontend atualizado em `IaLaudoPanel.tsx`;
- repasse no `VisualizadorClinico.tsx`;
- tipos atualizados em `frontend/src/types/api.ts`.

O sistema passou a aceitar:

- `model_type`;
- `trecho_suspeito`;
- `top_trechos_suspeitos`;
- `janela_pico`;
- `n_sequences_analisadas`;
- `threshold`;
- `min_duration_seconds`;
- `score_agregacao`;
- `feature_mode`;
- canais processados/omitidos.

## Calibracao e localizacao temporal

O problema central mudou de "a IA funciona?" para:

- ela identifica crise?
- ela evita falso positivo?
- ela localiza o trecho certo?

Foram adicionadas/ajustadas metricas de localizacao:

- sobreposicao com crise real;
- `min_overlap_seconds`;
- `min_overlap_ratio`;
- falsos positivos sem sobreposicao;
- falsos negativos localizados;
- trechos suspeitos continuos;
- limite de cobertura suspeita.

## Estado final ate esta documentacao

O sistema esta funcional, mas o modelo ainda nao e clinicamente confiavel.

Melhor artefato conhecido ate agora para localizacao:

```text
sequence_cnn_lstm_tf_per_channel_localized_w1
```

Artefato mais recente com mais EDFs:

```text
sequence_cnn_lstm_tf_per_channel_expanded_20260702
```

O artefato mais recente nao foi promovido porque piorou a localizacao temporal em teste separado.

