# Estado consolidado do EEG-XAI

Atualizado em: 2026-10-07

Este e o documento canonico sobre o estado tecnico e cientifico do projeto. Os
arquivos `01` a `17` preservam o historico detalhado, comandos, tabelas e
decisoes de cada etapa.

## 1. Objetivo do projeto

O sistema recebe exames EEG em EDF, processa sinais multicanais, executa uma
rede neural para estimar probabilidade ictal ao longo do tempo, agrega janelas
em trechos suspeitos e apresenta o resultado como apoio a revisao medica. A IA
nao deve emitir diagnostico autonomo.

O objetivo cientifico principal e generalizacao interpaciente: analisar um
paciente que nao participou do treino. O objetivo de engenharia e integrar
essa inferencia ao backend, API, visualizador EEG, SHAP e fluxo de laudo.

## 2. O que ja esta implementado

### Backend e inferencia

- leitura e validacao de EDF com MNE;
- extracao de features por canal e janela;
- representacao `time_frequency_per_channel`;
- normalizacao robusta por EDF;
- inferencia CNN-BiLSTM com probabilidade por passo temporal;
- agregacao em trecho principal e top trechos suspeitos;
- threshold, duracao, gap e histerese calibraveis;
- suporte a modelo sequencial ou fluxo legado por `.env`;
- fallback automatico quando artefatos sequenciais nao existem;
- isolamento de treino, calibracao e avaliacao por paciente;
- metricas por EDF, evento e localizacao;
- FROC e falsos alarmes por hora;
- reconstrucao de derivacoes bipolares a partir de canais com referencia comum,
  como `F7-T7 = (F7-CS2) - (T7-CS2)`;
- sampler por evento que preserva crises distintas dentro do mesmo EDF;
- cache em disco versionado para datasets janelados;
- auditoria por crise e por seed.

### Explicabilidade e API

- SHAP para a sequencia de pico;
- contribuicoes associadas a janelas temporais e canais;
- overlay temporal do SHAP sobre o tracado EEG;
- API com tipo de modelo, threshold, duracao, score, trecho principal, top
  trechos, janela de pico, canais processados e canais omitidos;
- linguagem de apoio clinico, sem afirmar diagnostico definitivo.

Limite atual: o SHAP explica features da sequencia escolhida, nao cada amostra
bruta nem qualquer intervalo arbitrariamente selecionado pelo medico. Exames
antigos precisam ser reanalisados para receber os novos metadados de overlay.

### Frontend

- fluxo de upload, processamento e polling do diagnostico;
- visualizador multicanal;
- selecao de canais;
- marcacao de trechos suspeitos e SHAP no eixo temporal;
- painel de IA com informacoes do modelo sequencial;
- configuracoes de threshold, montagem, SHAP e perfil medico;
- telas clinicas revisadas em azul e branco;
- estados de carregamento e textos clinicos refinados.

### Qualidade

- 168 testes automatizados aprovados em 2026-10-07;
- build de producao do frontend aprovado;
- `git diff --check` aprovado;
- EDFs, caches, modelos `.keras`, scalers e `.env` permanecem fora do Git.

## 3. Alinhamento com o TCC

### Aderente

- processamento de EEG multicanal em EDF;
- uso de CNN e recorrencia LSTM/BiLSTM;
- divisao do sinal em janelas;
- extracao de informacao temporal e espectral;
- dinamica simbolica preservada no extrator historico;
- explicabilidade com SHAP;
- configuracoes de confianca, montagem e perfil medico;
- interface de apoio a analise e emissao de parecer;
- avaliacao quantitativa com precisao, sensibilidade e F1.

### Extensoes justificadas

- segmentacao temporal em vez de somente classificacao global;
- FROC, FA/h, IoU, erros de inicio/fim e metricas por evento;
- sampler por crise;
- reconstrucao de montagem referencial;
- experimentos com TCN, multiescala, mascara de canais e confirmador.

Essas extensoes nao contradizem o TCC. Elas tornam a validacao temporal e
interpaciente mais rigorosa.

### Ainda parcial ou pendente

- validacao clinica por especialista;
- validacao externa realmente independente;
- SHAP para qualquer trecho escolhido pelo medico;
- prova de generalizacao clinica;
- modelo final congelado e promovido com artefatos reproduziveis;
- relatorio final do TCC atualizado com o protocolo real de segmentacao.

## 4. Dados e protocolo atual

O manifesto v38 possui 102 EDFs:

- treino: 77 EDFs de 17 grupos de paciente;
- calibracao fixa: 13 EDFs de `chb16`, `chb19`, `chb20` e `chb24`;
- desenvolvimento reservado historico: 12 EDFs de `chb09`, `chb15` e `chb18`.

Na configuracao v39:

- 4.648 sequencias amostradas no treino;
- 952 sequencias positivas pelo rotulo de sequencia;
- 11.587 sequencias completas na calibracao;
- 153 sequencias positivas na calibracao;
- aproximadamente 72 eventos de treino, 20 de calibracao e nove no conjunto
  reservado historico.

O manifesto v38 adicionou nove EDFs e 14 eventos ao v23. Os arquivos EDF nao
sao versionados; somente os manifestos publicos sao enviados ao Git.

`chb09`, `chb15` e `chb18` foram consultados em varias versoes. Portanto, sao
um conjunto de desenvolvimento comparavel, mas nao podem mais ser descritos
como teste final intocado.

## 5. Evolucao experimental

### Fase inicial ate v22

O projeto evoluiu de classificacao por janela e medias por canal para uma
CNN-BiLSTM de segmentacao temporal. Foram introduzidos normalizacao robusta,
weighted BCE, agrupamento em eventos, gap closing, histerese e confirmador
Random Forest.

Principais aprendizados:

- accuracy por janela escondia falsos alarmes e erros temporais;
- split aleatorio de janelas causava vazamento entre pacientes;
- o confirmador reduz falsos positivos, mas nao recupera uma crise sem candidato;
- focal loss e pesos agressivos aumentaram recall em alguns casos, mas pioraram
  precisao ou estabilidade;
- pos-processamento sozinho chegou a um plato.

Detalhes e numeros completos estao em `03_experimentos_e_metricas.md`.

### Baselines de desenvolvimento v23b-v30

| Versao | F1 EDF | F1 localizado | Eventos | FA/h | Decisao |
|---|---:|---:|---:|---:|---|
| v23b | 86,7% | 66,7% | 6/9 | 1,36 | baseline de localizacao |
| v27 | 76,7% | 55,6% | 7/9 | 1,69 | gerador sensivel, rejeitado |
| v28 | - | 61,1% | 6/9 | 0,58 | confirmador perde evento |
| v29 | 86,7% | 61,1% | 7/9 | 1,30 | rejeitado |
| v30 | 93,3% | 61,1% | 5/9 | 0,78 | rejeitado por recall |

v23b preservou a melhor localizacao; v29 preservou mais eventos. Nenhuma versao
atingiu simultaneamente localizacao alta e menos de um falso alarme por hora.

### Investigacao arquitetural v31

Foram comparados CNN-BiLSTM, TCN residual, multiescala, channel attention,
boundary loss, Tversky, continuidade e sampler por evento. A variante
multiescala escolhida na calibracao caiu para 5/9 eventos e 1,75 FA/h no
desenvolvimento reservado. Decisao: rejeitar.

Aprendizado: arquitetura menor e mais moderna nao garante generalizacao.
Channel attention simples reduziu eventos e nao deve ser repetida sem nova
hipotese.

### Representacao v32-v36

| Versao | Mudanca | Eventos calibracao | F1 localizado | F1 EDF | FA/h | Decisao |
|---|---|---:|---:|---:|---:|---|
| v32 | mascara de canais | 12,33/20 | 71,3% | 90,9% | 0,81 | nao promover |
| v33 | deltas temporais | 13,33/20 | 77,9% | 94,1% | 0,81 | rejeitada no reservado |
| v34 | morfologia precoce | 9,00/20 | 69,0% | 90,9% | 0,73 | rejeitar |
| v35 | morfologia residual | 10,00/20 | 71,3% | 89,2% | 0,76 | rejeitar |
| v36 | branch de sinal bruto | 8,00/20 | 63,5% | 87,3% | 0,84 | rejeitar |

A v33 parecia superior na calibracao, mas caiu para 6/9 eventos, F1 localizado
55,6% e 1,69 FA/h no desenvolvimento reservado. Isso confirmou overfitting ao
conjunto de calibracao.

### Expansao de dados v37-v39

| Versao | Mudanca | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h |
|---|---|---:|---:|---:|---:|---:|
| v37 | dois eventos novos | 12,67/20 | 59,7% | 72,5% | 89,6% | 0,76 |
| v38 | 14 eventos novos | 11,33/20 | 53,4% | 67,5% | 87,9% | 0,86 |
| v39 | sampler por evento | 12,33/20 | 64,7% | 75,8% | 90,9% | 0,47 |

A v38 piorou porque o sampler antigo podia eliminar crises inteiras de EDFs
com varios eventos. Em `chb12_27`, duas das seis crises nao contribuiam com
nenhuma sequencia. A v39 distribui a cota positiva entre eventos e passou a
cobrir todas as seis crises com 5-6 sequencias cada.

Aprendizado: aumentar dados sem auditar amostragem pode piorar o modelo. O ganho
da v39 veio da cobertura correta dos eventos, nao de nova arquitetura.

### Checkpoint e crises curtas v40-v41

| Versao/ponto | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h | Decisao |
|---|---:|---:|---:|---:|---:|---|
| v40, teto 0,50 | 12,33/20 | 68,0% | 77,9% | 89,2% | 0,31 | promissor |
| v40b, teto 0,75 | 13,33/20 | 65,6% | 75,0% | 92,9% | 0,58 | promissor |
| v40, FROC 1,00 | 13,67/20 | 64,1% | 76,6% | 92,6% | 0,71 | trade-off |
| v41, peso curto 1,5x | 12,00/20 | 61,7% | 73,7% | 90,9% | 0,55 | rejeitar |

A v40 seleciona checkpoint por recall de evento sob limite de FA/h. Ela melhora
o compromisso no teto 0,75, mas nao aumenta eventos no teto estrito 0,50. A
v41 mostra que aumentar moderadamente o peso das crises de ate 30 segundos nao
resolve o problema.

Nenhuma dessas versoes foi promovida para producao.

### Detector temporal duplo v42

A v42 manteve o caminho sustentado da v40b e adicionou um caminho transiente
de alta confianca com duracao de 6-10 segundos. A media passou de 13,33 para
13,67 eventos em 20, mas o F1 localizado caiu de 75,0% para 71,0% e o FA/h
subiu de 0,58 para 0,86. Somente a seed 43 recuperou um evento novo;
`chb24_04.edf`, evento 3. As outras duas seeds nao recuperaram crises e
adicionaram falsos alarmes.

Decisao: rejeitar. Uma regra curta baseada apenas em score e duracao nao separa
transientes normais de crises curtas com estabilidade suficiente.

### Fusao espacial tardia v43

A v43 preserva canais durante duas convolucoes temporais e realiza fusao
espacial antes do mesmo BiLSTM da baseline. Em cinco seeds no teto 0,75,
atingiu media de 13,8/20 eventos, F1 localizado de 81,9%, F1 EDF de 96,4% e
0,64 FA/h. Nas seeds 42-44, superou a baseline em eventos, localizacao e F1
EDF, com pequeno aumento de FA/h.

Decisao: promissor, mas nao superior. A pior seed caiu para 12/20 e os eventos
2-4 de `chb16_17` continuaram invisiveis. A arquitetura nao foi promovida.

### Estabilidade de otimizacao v44

A v44 reduziu o learning rate, adicionou reducao por plateau e antecipou F1 EDF
no desempate de checkpoint. Em cinco seeds, caiu para 13,4/20 eventos, F1
localizado de 76,7%, F1 EDF de 94,8% e 0,63 FA/h. A pior seed permaneceu em
12/20. Decisao: rejeitar; a v43 continua como candidata experimental.

### Crises curtas e auditoria de similaridade v45

A auditoria mostrou apenas uma crise de ate 10 segundos entre 72 eventos do
treino e nenhuma assinatura com similaridade maior ou igual a 0,8 para os
eventos de `chb16_17`. Foram adicionadas tres crises de 12-13 segundos do
`chb06`. Em seeds 42-44, a v45 manteve 14/20 eventos, mas o F1 localizado caiu
de 83,4% para 75,4% e o F1 EDF de 97,1% para 92,5%. Decisao: rejeitar.

### Resolucao temporal v46

A comparacao dirigida entre 4 s/2 s e 2 s/1 s mostrou que a resolucao curta
melhora separacao e similaridade dos eventos 3 e 4 de `chb16_17`, mas piora os
eventos 1 e 2. Nao foi executado treino completo. A conclusao e preservar a
v43 e investigar uma branch multirresolucao pequena, sem substituicao global.

## 6. Auditoria por crise da v40

Comparando seeds 42 e 44 no teto 0,50:

- 11/20 eventos sao detectados pelas duas;
- seis sao perdidos pelas duas;
- tres aparecem somente na seed 44.

Casos persistentes:

- `chb16_11`: crise de 12 s, score maximo 0,778/0,965, bloqueada por duracao;
- `chb24_04` evento 3: 22 s, score aproximadamente 1,0, bloqueada pela regra
  de 26 s;
- `chb16_17` eventos 2 e 3: scores muito baixos nas duas seeds;
- `chb16_17` evento 4: score 0,169/0,780, forte instabilidade;
- `chb20_13` e `chb24_01`: picos altos, mas evidencia fragmentada;
- `chb20_14`: score 0,434/0,988 conforme a seed.

Conclusao: ha quatro gargalos diferentes: duracao minima, fragmentacao,
representacao fraca e instabilidade de otimizacao. Uma unica troca de
arquitetura nao corrige automaticamente os quatro.

## 7. O que nao funcionou

- aumentar apenas epochs;
- contexto fixo maior para todas as crises;
- ensemble simples de modelos;
- focal loss sem controle de falsos alarmes;
- pesos agressivos por evento;
- features morfologicas agregadas;
- branch curta de sinal bruto residual;
- channel attention da v31;
- Random Forest como substituto do primeiro estagio;
- calibrar somente threshold, duracao ou smoothing;
- adicionar EDFs sem garantir cobertura de todos os eventos.

Esses resultados devem permanecer documentados para evitar repetir
experimentos sem nova hipotese.

## 8. Problemas que impedem finalizar o TCC

### Cientificos

1. Algumas crises continuam com score ictal baixo.
2. Crises curtas com score alto sao eliminadas pela duracao minima global.
3. A variacao entre seeds ainda e relevante.
4. A calibracao tem apenas 20 eventos.
5. O conjunto historicamente reservado ja foi consultado repetidamente.
6. Nao existe validacao externa independente.
7. As metricas ainda nao sustentam uso clinico autonomo.

### Engenharia e produto

1. Nao ha artefato v40 final salvo em `.keras` para promocao.
2. Threshold e regra temporal finais ainda nao estao congelados.
3. O overlay SHAP precisa de validacao visual sistematica com EDF real.
4. Inferencia e SHAP ainda podem ser pesados em exames longos.
5. O bundle frontend possui aviso de chunk acima de 500 kB.
6. O fluxo precisa comunicar claramente resultado inconclusivo e limitacoes.

### Redacao do TCC

1. Atualizar metodologia para split por paciente e segmentacao temporal.
2. Explicar treino, calibracao e desenvolvimento reservado separadamente.
3. Reportar FA/h e metricas por evento, nao apenas accuracy.
4. Descrever extensoes ao plano original sem afirmar que estavam no texto
   inicial.
5. Registrar que o sistema e apoio a decisao e nao dispositivo clinico validado.

## 9. Proximos passos priorizados

### P0 - Detector temporal de dois caminhos - concluido

A v42 executou esta hipotese e foi rejeitada. Ela nao deve ser ativada: elevou
FA/h, reduziu localizacao e nao recuperou eventos de forma consistente.

### P1 - Prototipo multirresolucao dirigido

- manter a v43 congelada como baseline;
- preservar a branch de 4 s/2 s;
- adicionar branch curta de 2 s/1 s alinhada ao mesmo eixo temporal;
- testar primeiro no subconjunto auditado;
- exigir ganho nos eventos 3-4 sem perder 1-2;
- gerar cache completo apenas se o prototipo passar.

### P2 - Reprodutibilidade do modelo selecionado

- salvar pesos do checkpoint escolhido;
- salvar scaler, metadata e calibracao juntos;
- persistir traces temporais comprimidos da calibracao;
- registrar versao do codigo, manifesto, seed e hash dos artefatos;
- permitir reavaliar pos-processamento sem retreinar.

### P3 - Representacao para eventos realmente invisiveis

Somente depois do P0, investigar os eventos de `chb16_17` com score baixo:

- frontend temporal por canal com compartilhamento de pesos;
- fusao espacial tardia e regularizada;
- sem pooling temporal global;
- ablacao pareada contra v39/v40;
- nao repetir a channel attention da v31 sem mudanca fundamentada.

### P4 - Estabilidade

- aumentar para pelo menos cinco seeds no candidato final;
- reportar media, desvio e pior seed;
- estudar snapshot averaging dentro da mesma trajetoria apenas se nao mascarar
  o protocolo;
- nao usar ensemble heterogeneo simples, que ja piorou.

### P5 - Avaliacao final honesta

- congelar arquitetura e pos-processamento antes da avaliacao;
- tratar `chb09/chb15/chb18` como desenvolvimento, nao teste intocado;
- usar validacao cruzada agrupada/nested por paciente ou dataset externo;
- reportar intervalo de confianca e resultados por paciente/evento;
- manter todos os pacientes de um mesmo individuo no mesmo fold.

### P6 - Produto e TCC

- validar visualmente EEG + trecho + SHAP em desktop e mobile;
- medir tempo e memoria em EDF longo;
- revisar textos de limitacao clinica;
- atualizar capitulos de metodologia, resultados e discussao;
- anexar tabela de ablacao e FROC ao trabalho.

## 10. Regras para os proximos experimentos

1. Uma mudanca principal por versao.
2. Nenhuma escolha olhando o conjunto de avaliacao.
3. Split sempre por paciente.
4. Seeds e manifestos sempre registrados.
5. Comparar no mesmo conjunto e mesma grade.
6. Reportar EDF, evento, temporal, FA/h e FROC.
7. Nao promover por uma unica seed.
8. Nao chamar score de probabilidade clinica calibrada sem validacao.
9. Nao apagar resultados negativos.
10. Nao substituir o modelo em uso ate superar criterios predefinidos.

## 11. Estado de decisao

- melhor baseline temporal historica no reservado: v23b para localizacao;
- gerador historico mais sensivel no reservado: v29, com 7/9 eventos;
- melhor compromisso recente de calibracao: v40/v40b;
- melhor reducao recente de FA/h: v40 no teto 0,50;
- modelo novo promovido: nenhum;
- experimento v42: rejeitado;
- experimento v43: promissor, ainda nao promovido;
- experimento v44: rejeitado;
- experimento v45: rejeitado;
- auditoria v46: suporta multirresolucao, nao substituicao global;
- experimento v47: promissor, mas ainda sem avaliacao FROC completa;
- experimento v48 residual: rejeitado por instabilidade entre seeds;
- proximo experimento aprovado: v47 no treino/calibracao completos, com
  pos-processamento fixo e sem acesso aos pacientes reservados;
- gargalo principal: eventos invisiveis ao primeiro estagio e transientes
  normais indistinguiveis por threshold/duracao.

## 12. Mapa da documentacao

- `01_historico_do_trabalho.md`: historico geral inicial;
- `02_estado_atual_da_ia.md`: arquitetura integrada;
- `03_experimentos_e_metricas.md`: resultados ate v29;
- `04_problemas_conhecidos.md`: riscos e limitacoes;
- `05_proximos_passos.md`: plano historico;
- `06_auditoria_v29.md`: auditoria temporal v29;
- `07_experimento_v30_confirmador_froc.md`: confirmador v30;
- `08_investigacao_arquitetural_v31.md`: alternativas arquiteturais;
- `09_resultado_v31_multiescala.md`: ablacao v31;
- `10_auditoria_representacao_v32.md`: canais e saturacao;
- `11_resultado_v33_deltas_temporais.md`: deltas;
- `12_resultados_v34_v35_morfologia.md`: morfologia;
- `13_resultado_v36_branch_sinal_bruto.md`: sinal bruto;
- `14_resultado_v37_dados_expandidos.md`: primeiro lote novo;
- `15_resultados_v38_v39_expansao_eventos.md`: expansao e sampler;
- `16_resultados_v40_v41_checkpoint_eventos.md`: checkpoint e crises curtas;
- `17_auditoria_estabilidade_v40.md`: comparacao por crise;
- `18_estado_consolidado_e_plano.md`: estado canonico atual.
- `19_resultado_v42_detector_duplo.md`: detector temporal de dois caminhos.
- `20_resultado_v43_fusao_tardia_canais.md`: frontend por canal e fusao tardia.
- `21_resultado_v44_estabilidade_otimizacao.md`: learning rate e estabilidade.
- `22_auditoria_v45_crises_curtas.md`: similaridade e expansao com crises curtas.
- `23_auditoria_v46_resolucao_temporal.md`: comparacao 4 s/2 s contra 2 s/1 s.
- `24_resultados_v47_v48_multirresolucao.md`: prototipo de duas resolucoes.
