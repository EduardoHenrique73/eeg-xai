# Proximos passos

Atualizado em: 2026-09-30

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
F1 localizacao reavaliado: 0.4000
FP: 3
FN: 3
FN localizacao: 12
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
- ter F1 localizado maior que `0.4000` no avaliador corrigido;
- reduzir falsos positivos;
- nao aumentar falsos negativos;
- mostrar estabilidade em pacientes nunca vistos;
- produzir trechos suspeitos clinicamente revisaveis.

Para uso real, isso ainda nao basta. Seria necessario validacao muito mais ampla e revisao clinica.

## Prioridade apos o experimento `center_v2`

1. Manter `localized_w1` como baseline ativo e manter `center_v2` rejeitado.
2. Criar um modo tempo-frequencia robusto a amplitude, comparando features
   relativas/log-ratios com as potencias absolutas atuais.
3. Medir distribuicao das features e dos scores por paciente antes de treinar.
4. Reutilizar `modelos/sequence_dataset_cache` para iteracoes com configuracao identica.
5. Treinar o novo candidato com alvo central e blocos contiguos, pois essas
   correcoes resolvem erros de rotulacao mesmo sem terem melhorado este candidato.
6. Calibrar apenas nos pacientes de calibracao e aplicar a regra congelada ao teste.
7. Promover somente se superar F1 localizado 0.4000, FP <= 3, FN <= 3 e FN de
   localizacao <= 12 no avaliador corrigido e em pacientes nao usados na escolha.

Nao priorizar agora:

- apenas aumentar epocas;
- escolher threshold no conjunto de teste;
- ativar `center_v2` por ter localizacao temporal mais precisa na implementacao;
- trocar o modelo ativo sem superar os criterios objetivos.

## Depois dos modelos `v3`, `v4` e `v5`

As features relativas e a normalizacao robusta foram implementadas e testadas,
mas ainda nao superaram a baseline. O proximo experimento nao deve reutilizar o
mesmo teste para escolher novas regras.

Prioridade atual:

1. baixar pacientes CHB-MIT adicionais e reservar um novo teste intocado;
2. ampliar a calibracao com crises curtas e longas para evitar duracao minima enviesada;
3. comparar perda focal ou modelo por janela com contexto, sem alterar o novo teste;
4. avaliar sensibilidade por evento, incluindo todas as crises de um EDF;
5. manter `localized_w1` ativo ate um candidato superar todos os criterios.

## Prioridade apos a auditoria de 2026-09-30

1. Reservar novos pacientes CHB-MIT como teste final intocado e registrar seus
   identificadores antes de qualquer novo treino.
2. Calibrar `v7` somente em pacientes de calibracao, processando EDFs completos,
   e medir sensibilidade por evento, falsos alarmes por hora e overlap temporal.
3. Aplicar uma unica vez a regra congelada ao novo teste. Nao escolher threshold
   nem duracao a partir desse resultado.
4. Comparar `v7` com a baseline recalculada (`F1 localizado 0.4000`), mantendo
   simultaneamente limites para FP, FN e alarmes por hora.
5. Executar validacao agrupada por paciente (GroupKFold ou leave-one-patient-out)
   e relatar media, desvio e resultados por paciente.
6. Somente promover um artefato depois da validacao completa. O arquivo de
   calibracao exploratoria amostrada nao deve ser configurado no `.env`.

Ja corrigido no produto:

- visualizacao multicanal com sobreposicao dos trechos suspeitos;
- Gradient SHAP real para a entrada sequencial e identificacao do metodo na UI;
- threshold do perfil medico aplicado ao modelo sequencial;
- isolamento de pacientes, exames e estatisticas por medico autenticado;
- validacao real de CPF;
- bloqueio de calibracao amostrada como resultado final sem flag explicita.
- normalizacao robusta feita no EDF completo antes da amostragem de treino;
- early stopping por PR-AUC;
- calibracao interpaciente com modo balanceado ou conservador explicito;
- extracao multicanal em memoria, permitindo avaliar EDFs completos.

## Prioridade apos a `v10`

1. Nao promover `v9` ou `v10`; manter a baseline ativa ate validacao final.
2. Reservar pacientes ainda nao consultados como teste final intocado.
3. Ampliar pacientes de calibracao e variar sua composicao; `chb20/chb24`
   sozinhos nao representam bem `chb15/chb18`.
4. Investigar especificamente o `chb18`, no qual nenhuma das duas crises foi
   localizada, comparando canais ausentes, distribuicao das features e scores.
5. Treinar com referencia robusta `v4`, modo balanceado e mais diversidade de
   pacientes, sem voltar a aumentar negativos indiscriminadamente.
6. Relatar os dois objetivos separadamente: classificacao do EDF e localizacao
   temporal. O melhor resultado atual ainda localiza apenas 4 de 9 crises.
7. Fazer uma unica avaliacao final no teste intocado e registrar intervalo de
   confianca, sensibilidade por evento e falsos alarmes por hora.

## Prioridade apos a `v11`

1. Manter o ranqueamento por evidencia acumulada e a calibracao com pelo menos
   quatro pacientes diversos.
2. Reservar novos pacientes que nunca participaram de treino, calibracao ou
   escolha de hiperparametros.
3. Atacar as cinco crises ainda nao localizadas, separando falha do classificador
   de falha da regra temporal; o `chb18_30` continua com score ictal baixo.
4. Avaliar suavizacao temporal e uniao de pequenos intervalos somente na coorte
   de calibracao, com limite explicito de falsos alarmes por hora.
5. Relatar sensibilidade por evento alem da classificacao por EDF; um EDF
   positivo nao prova que todas as crises foram encontradas.
6. Congelar threshold, duracao e pos-processamento antes de abrir o teste final.
7. Promover somente depois de manter o ganho em pacientes intocados. O resultado
   atual de F1 localizado `0.6111` ainda e de desenvolvimento.

## Prioridade apos a segmentacao `v14`

1. Preservar a `v14` como candidata de alta sensibilidade: ela obteve recall por
   EDF `1.0` e zero falsos negativos.
2. Implementar fechamento de lacunas curtas ou histerese temporal na saida por
   janela, calibrando simultaneamente threshold, duracao e lacuna maxima.
3. Impor limite de falsos alarmes por hora durante a calibracao; o `chb18` ainda
   apresenta alarmes excessivos.
4. Fazer hard-positive mining de `chb15_10`, `chb15_17` e crises curtas de
   pacientes de treino, sem usar os pacientes de teste para atualizar pesos.
5. Comparar o ensemble entre a `v11` e a segmentacao somente na coorte de
   calibracao: concordancia pode reduzir artefatos, enquanto a `v14` preserva
   sensibilidade.
6. Repetir a avaliacao uma unica vez em pacientes intocados antes de promover.

## Prioridade apos a `v16` a `v19`

1. Manter `v16/v18` como melhor referencia de desenvolvimento: F1 por EDF
   `0.8963`, F1 localizado `0.5794` e sensibilidade por evento `6/9`.
2. Nao adotar o balanceamento por evento da `v17` nem os rotulos de borda da
   `v19`; ambos pioraram a avaliacao interpaciente.
3. Ampliar o numero de crises independentes de treino, principalmente eventos
   curtos e morfologias semelhantes aos falsos negativos, sem copiar pacientes
   do teste para o treino.
4. Reservar novos pacientes para teste final antes da proxima escolha de
   arquitetura ou regra temporal.
5. Avaliar um detector em dois estagios na calibracao: candidato sensivel por
   janela e confirmador de trecho para rejeitar artefatos. Medir sensibilidade
   por evento e falsos alarmes por hora simultaneamente.
6. Melhorar o ranqueamento dos candidatos secundarios sem confundir isso com
   deteccao: na `v16`, 6/9 eventos aparecem em algum candidato, mas apenas 4/9
   ocupam o trecho principal.
7. Congelar todos os hiperparametros e executar uma unica avaliacao final. Nao
   promover o modelo enquanto a localizacao permanecer abaixo do criterio
   acordado para o TCC.

## Prioridade apos o prototipo de dois estagios `v20-v22`

1. Preservar a infraestrutura do confirmador, hard negatives e FROC, mas nao
   promover seus artefatos atuais.
2. Ampliar o treino do gerador com novos pacientes e crises independentes,
   mantendo `chb09`, `chb15` e `chb18` fora de qualquer atualizacao de pesos.
3. Gerar candidatos a partir de um modelo treinado para alta sensibilidade, em
   vez de apenas baixar o threshold da `v16/v18`.
4. Recriar os hard negatives com predicoes fora do paciente. Nao treinar o
   confirmador em predicoes do mesmo paciente usadas para ajustar o gerador.
5. Selecionar o ponto de operacao pela FROC com um limite de FA/h declarado e
   medir atraso de deteccao e cobertura temporal alem do F1.
6. Abrir uma coorte final intocada somente depois de congelar gerador,
   confirmador, features e thresholds.

## Prioridade apos a expansao `v23b` e os testes `v24-v25`

1. Usar a `v23b` como referencia controlada de desenvolvimento: F1 por EDF
   `0.8667`, F1 localizado `0.6667`, 5/9 crises principais e 6/9 eventos.
2. Nao promover o ensemble `v24` nem o confirmador `v25`.
3. Testar maior contexto temporal da CNN-BiLSTM, mantendo manifesto, pacientes
   reservados, calibracao, loss e regra de decisao fixos.
4. Priorizar `chb15_10` como falha de representacao; pos-processamento sozinho
   nao recupera uma crise cujo score ictal maximo permanece baixo.
5. Tratar `chb15_17` e `chb18_30` como falhas de continuidade, medindo cobertura
   temporal e atraso, alem do simples acerto por EDF.
6. Baixar novos dados de forma retomavel, validar integridade e incluir somente
   EDFs completos. Arquivos `.part` nunca entram no manifesto.
7. Congelar a proxima configuracao usando apenas treino/calibracao e abrir uma
   nova coorte intocada uma unica vez. `chb09`, `chb15` e `chb18` sao agora
   pacientes de desenvolvimento, nao teste final independente.

## Prioridade apos `v26-v28`

1. Manter a `v23b` como melhor decisao unica controlada e nao promover
   `v26-v28`.
2. Preservar a amostragem por sequencia como opcao metodologicamente correta,
   mas recalibrar seus limites por arquivo apenas em pacientes de calibracao.
3. Usar a `v27` como referencia de gerador sensivel: ela encontrou 7/9 eventos,
   embora com `1.6865` alarmes por hora.
4. Reformular o confirmador para priorizar recall de evento sob teto de FA/h;
   a `v28` filtrou alarmes, mas eliminou o evento adicional.
5. Evitar contexto longo unico. Uma futura abordagem multiescala deve manter o
   ramo curto para crises breves e adicionar contexto sem descartar amostras.
6. Selecionar novos limites e arquitetura sem consultar novamente os tres
   pacientes de desenvolvimento; a confirmacao deve ocorrer em coorte intocada.

## Prioridade apos `v29`

1. Nao promover a `v29`: ela atingiu 7/9 eventos, mas ficou em F1 localizado
   `0.6111` e `1.2973` FA/h.
2. Preservar 48 sequencias normais/EDF como valor escolhido pela calibracao para
   o proximo experimento controlado.
3. Atacar separadamente as falhas de representacao `chb15_10` e `chb18_30`; os
   scores ictais maximos continuam baixos e nao podem ser corrigidos por regra
   temporal.
4. Melhorar o ranqueamento de candidatos para `chb09_06` e `chb15_20`, onde o
   evento foi detectado, mas um falso trecho virou o principal.
5. No proximo segundo estagio, selecionar threshold por recall de evento sujeito
   a FA/h menor ou igual a 1.0 exclusivamente na calibracao.
6. Repetir seeds somente na calibracao para estimar estabilidade; congelar uma
   unica seed/modelo antes de abrir uma nova coorte final.
