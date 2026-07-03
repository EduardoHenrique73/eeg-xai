# Proximos passos

Atualizado em: 2026-07-03

## Objetivo tecnico

Melhorar a IA para detectar crise em paciente novo com menor falso positivo e melhor localizacao temporal.

O objetivo nao e apenas aumentar F1 geral. A prioridade deve ser:

1. reduzir falsos negativos;
2. reduzir falsos positivos;
3. aumentar F1 localizado;
4. manter separacao correta por paciente;
5. so promover modelo que melhore no teste separado.

## Passo 1: congelar baseline atual

Antes de novos experimentos, registrar como baseline:

```text
Modelo: sequence_cnn_lstm_tf_per_channel_localized_w1
F1 geral: 0.8235
F1 localizacao: 0.5185
FP: 3
FN: 3
FN localizacao: 10
```

Esse e o modelo a bater.

Nenhum modelo novo deve ser promovido se nao melhorar principalmente:

- F1 localizado;
- falsos positivos;
- falsos negativos localizados.

## Passo 2: baixar dados de forma controlada

Nao baixar arquivos aleatorios sem estrategia.

Baixar por paciente, garantindo:

- pacientes apenas de treino;
- pacientes apenas de calibracao;
- pacientes apenas de teste;
- pacientes com crises;
- pacientes com arquivos normais.

Prioridade:

1. aumentar pacientes de treino com crise;
2. aumentar pacientes de treino com normais completos;
3. manter pacientes de teste intocados;
4. nao usar pacientes de teste para calibrar threshold.

Meta inicial razoavel:

```text
Treino:      12 a 16 pacientes
Calibracao: 4 a 6 pacientes
Teste:      4 a 6 pacientes
```

Ideal:

```text
Usar quase todo CHB-MIT com leave-one-patient-out ou splits repetidos por paciente.
```

## Passo 3: melhorar amostragem de janelas

Implementar/fortalecer amostragem por contexto:

- janelas durante crise;
- janelas pre-crise;
- janelas pos-crise;
- janelas normais distantes de qualquer crise;
- janelas de arquivos totalmente normais.

Motivo:

O modelo precisa aprender nao so "crise", mas tambem o que e parecido com crise e nao deve ser marcado.

Sugestao de labels auxiliares para analise:

```text
ictal
pre_ictal
post_ictal
normal_near
normal_far
normal_file
```

Mesmo que a rede continue binaria, esses grupos devem ser medidos separadamente.

## Passo 4: penalizar localizacao errada

A avaliacao ja caminha nessa direcao, mas o treino ainda precisa refletir melhor isso.

Metricas obrigatorias por experimento:

- F1 por EDF;
- F1 localizado;
- FP por EDF normal;
- FN por EDF com crise;
- segmentos positivos sem overlap;
- overlap medio com crise real;
- cobertura suspeita do exame.

Regra de promocao:

```text
Nao promover modelo que melhora F1 geral mas piora F1 localizado.
```

## Passo 5: testar arquitetura melhor, mas sem fugir do TCC

Manter CNN-LSTM como linha principal.

Possiveis melhorias ainda alinhadas:

- CNN-LSTM com features `time_frequency_per_channel`;
- CNN-LSTM com atencao temporal simples;
- CNN-LSTM com pooling por canal mais cuidadoso;
- saida por janela/sequencia em vez de apenas arquivo;
- regularizacao mais forte;
- focal loss ou loss ponderada para crise;
- calibracao pos-treino.

Evitar neste momento:

- trocar para Random Forest como modelo principal;
- promover modelo so porque acerta arquivo;
- usar teste para escolher threshold.

## Passo 6: calibrar probabilidades

Problema atual:

O modelo pode ficar superconfiante em trechos normais.

Testar:

- Platt scaling;
- isotonic regression;
- temperature scaling;
- calibracao por paciente de validacao;
- curva precision-recall por threshold/duracao.

Mas sempre avaliar depois em teste separado por paciente.

## Passo 7: relatorio automatico por experimento

Cada experimento deve gerar um resumo padrao:

```text
Nome do modelo
Feature mode
Pacientes de treino
Pacientes de calibracao
Pacientes de teste
Quantidade de EDFs
Quantidade de crises
Threshold escolhido
Duracao minima escolhida
F1 geral
F1 localizado
FP
FN
FN localizacao
Predicoes sem overlap
Conclusao: promover ou nao promover
```

Isso evita confusao quando houver muitos JSONs em `modelos/`.

## Passo 8: melhorar frontend depois que o modelo estabilizar

O frontend ja mostra trechos suspeitos. Depois que o modelo melhorar, evoluir:

- mostrar linha do tempo do exame;
- marcar trecho suspeito principal;
- listar top trechos com score e duracao;
- indicar se houve sobreposicao quando estiver em modo avaliacao;
- permitir filtrar canais;
- mostrar aviso quando modelo estiver em modo experimental;
- mostrar threshold e duracao usados.

## Passo 9: texto academico/TCC

Documentar no TCC ou relatorio final:

- CNN-LSTM foi mantida como arquitetura principal;
- Random Forest foi usada apenas como baseline quando aplicavel;
- features de tempo-frequencia tiveram melhor desempenho que sinal bruto na base atual;
- validacao por paciente e mais adequada ao objetivo real;
- metricas gerais podem esconder erro de localizacao;
- sistema e apoio clinico, nao diagnostico automatico.

## Passo 10: criterio de "modelo bom"

Um modelo minimo aceitavel para demonstracao deve:

- ter F1 geral maior que baseline;
- ter F1 localizado maior que `0.5185`;
- reduzir falsos positivos;
- nao aumentar falsos negativos;
- mostrar estabilidade em pacientes nunca vistos;
- produzir trechos suspeitos clinicamente revisaveis.

Para uso real, isso ainda nao basta. Seria necessario validacao muito mais ampla e revisao clinica.

