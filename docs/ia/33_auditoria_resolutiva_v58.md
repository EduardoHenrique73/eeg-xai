# Auditoria resolutiva e protocolo v58

## Evidencia que motivou a rodada

O objetivo e localizar mais eventos, mantendo precisao temporal e poucos alarmes.
Nao existe garantia de que uma ultima rodada resolva a generalizacao interpaciente.

| Tentativa | Evidencia registrada | Consequencia pratica |
|---|---|---|
| Threshold, gap e duracao | Alteram candidatos, mas nao criam evidencia ictal ausente | Nao selecionar regras nos pacientes historicos |
| v31 multiescala | Nao estabeleceu superioridade | Arquitetura maior isolada nao resolveu |
| v30 confirmador | FA/h caiu, mas eventos passaram de 7/9 para 5/9 no protocolo antigo | Confirmador tambem pode eliminar candidatos verdadeiros |
| v32 mascara | Ausencia de canais nao explica crises dificeis com todos os canais | Preservar mascara, nao tratar como causa unica |
| v33 deltas | Ganho na calibracao, pior localizacao historica | Transientes tambem aumentam falsos alarmes |
| v34/v35 morfologia | Menor sensibilidade | Mais features nao implica melhor representacao |
| v36 sinal bruto | Nao resultou em substituicao promovida | Nao repetir branch sem nova evidencia |
| v38/v39/v40 | Amostragem por evento e checkpoint ja implementados | Nao apresentar esses recursos como novidade |
| v43 fusao tardia | Baseline atual; 15/20 calibracao e 4/9 historico | Generalizacao continua sendo gargalo |
| v44 otimizacao | Nao superou baseline | Mais ajuste de learning rate nao tem ganho comprovado |
| v45 crises curtas | Adicionar poucos exemplos nao recuperou eventos invisiveis | Diversidade conta mais que quantidade bruta |
| v46-v53 multirresolucao | Evidencia complementar, sem superioridade final | Nao substituir globalmente janelas de 4 s |
| v55/v56 hard negatives | Conservadorismo ou localizacao pior | Nao aumentar penalidade negativa novamente |
| v57 balanceamento de paciente | Mesmo 4/9; FA/h 2,08 para 1,04 | Reduzir alarmes nao equivale a recuperar crises |

As tabelas antigas usam coortes, seeds e agregacoes diferentes. Nao constituem
um ranking diretamente comparavel de todas as versoes. Consultas repetidas ao
historico impedem apresenta-lo como teste independente intocado.

## Gaps prioritarios

Auditoria de anotacoes oficiais e cabecalhos (sem carregar sinais historicos):

| Conjunto | EDFs | Pessoas | Crises | Horas EEG | Minutos ictais | EDFs sem crise |
|---|---:|---:|---:|---:|---:|---:|
| Treino | 77 | 16 | 72 | 109,676 | 71,533 | 22 |
| Calibracao | 13 | 4 | 20 | 12,931 | 10,050 | 1 |

CHB01/CHB21 foram contados como uma pessoa. Os numeros de tempo acima usam
cabecalho e anotacao original; diferem ligeiramente dos denominadores derivados
de janelas. Treino efetivo: 4.648 sequencias, 952 positivas pelo rotulo central.
Sequencias sobrepostas nao sao 4.648 eventos independentes.

O treino tem 12 crises de CHB12 e 12 de CHB13, mas somente uma de CHB23.
CHB08 responde por 13,08 dos 71,53 minutos ictais, enquanto CHB06 tem 0,73.
Isso motiva estudar diversidade e dependencias; a tentativa v57 ja mostrou
que corrigir apenas o peso relativo dessas pessoas nao bastou.

A calibracao contem apenas um EDF inteiramente sem crise. Embora os registros
positivos tambem contenham horas normais para FA/h, a precisao por EDF tem
denominador negativo muito pequeno. F1 EDF alto nesse conjunto exige cautela.

1. Evidencia ictal baixa em eventos inteiros: problema de aprendizado/representacao,
   que nao pode ser atribuido apenas a pos-processamento.
2. Dependencia de amplitude e topografia e uma hipotese: a auditoria v32 encontrou
   potencia saturada em muitas janelas ictais, mas pouco nas crises dificeis.
3. Flatten espacial seguido de fusao aprendida pode favorecer canais dominantes.
   Preservar canais ate esse ponto nao garante robustez a outra topografia.
4. Somente quatro grupos de calibracao, repetidamente consultados: alto risco de
   especializacao das decisoes. Confirmacao futura requer coorte externa ou nova
   validacao agrupada; as tres seeds nao substituem pacientes independentes.
5. F1 por EDF pode melhorar por alarmes no lugar errado. Priorizar eventos reais,
   FA/h e sobreposicao; usar F1 EDF como medida complementar.
6. Ranking do trecho principal e perda de evento sao problemas diferentes.
7. Regra minima pode rejeitar crises curtas mesmo com score alto. Nao soluciona
   eventos que permanecem com score baixo; requer experimento separado.

## Gaps metodologicos adicionais observados no codigo

- `avaliar_combo` considera um evento detectado quando algum candidato o cobre,
  sem pareamento exclusivo 1:1 entre candidatos e crises. Um candidato extenso
  pode cobrir varios eventos; fragmentos de uma mesma crise nao sao penalizados
  como alarmes duplicados se cada um cumprir a cobertura minima. Trata-se da
  convencao atual; metricas 1:1 devem ser acrescentadas com outro nome e ambas
  as versoes reavaliadas, sem substituir silenciosamente numeros publicados.
- O `localized_f1` atual e calculado por EDF, condicionando o positivo a
  sobreposicao do candidato principal com crise. Nao e Dice/F1 de cada segundo
  ictal e nao garante bordas precisas. O nome completo ao apresentar e "F1 por
  EDF com localizacao do trecho principal"; IoU, cobertura e erro de borda
  respondem a perguntas temporais diferentes.
- Os intervalos de avaliacao sao reconstruidos de alvos janelados; nao devem
  ser tratados como precisao de inicio/fim na resolucao de amostra. Para medir
  erro exato de bordas, comparar tambem com anotacoes EDF originais.
- O peso chamado `boundary_weight` incide nas janelas parcialmente sobrepostas
  cujo rotulo ainda e negativo; nao equivale a uma loss explicita para inicio
  e fim de eventos positivos. Nao afirmar que ja existe boundary loss completa.
- O treino usa ate 48 sequencias normais por EDF, enquanto calibracao percorre
  registros completos: a exposicao a padroes normais e muito limitada. v55/v56
  mostraram que adicionar somente negativos de maior score nao corrigiu isso;
  cobertura diversificada por epoca e uma alternativa futura distinta.
- A grade FROC usada no treino atual testa thresholds de 0,10 a 0,60, duracoes
  6/10/18/26/30/40/60 s e gaps 0/4/6/8/10 s. Nao cobre 0,90. Portanto, dizer
  que todo espaco de pos-processamento foi esgotado seria excessivo. Uma grade
  ampliada deve comparar baseline e candidata igualmente, somente na calibracao;
  aumentar limiar sozinho nao recupera os eventos de score baixo.
- Normalizacao robusta usa o EDF inteiro. O protocolo e retrospectivo e nao
  demonstra um detector online causal. Isso e compativel com upload do exame,
  mas deve constar ao interpretar latencia.
- O extrator calcula cinco bandas de 0,5 a 45 Hz, mas usa todo o espectro
  positivo ate Nyquist para potencia total, frequencia dominante, centroide e
  entropia. Portanto, energia fora dessas bandas pode influenciar 15 features.
  Nao ha filtro notch/bandpass nesse caminho. E um gap de consistencia a auditar
  com PSD real; nao prova que interferencia de rede causou os erros. Alterar
  essas definicoes exige cache novo, retreino e metadados novos.
- Os 23 slots da montagem incluem derivacoes repetidas/invertidas; nao sao
  23 fontes fisiologicas independentes. Remover slots na inferencia de um modelo
  ja treinado seria mudar seu contrato de entrada. Auditar redundancia antes
  de redesenhar montagem.
- A prevalencia ictal no tempo de treino e cerca de 1,09%, enquanto os pesos
  amostrados atribuem 21.094,80 de massa positiva contra 18.125,88 negativa
  (53,79% da massa para ictal). Isso ajuda a expor crises ao aprendizado, mas
  o sigmoid treinado sob outra prevalencia nao esta calibrado como certeza
  diagnostica. Escolher threshold/duracao nao e calibrar probabilidades.
  Uma calibracao posterior pode melhorar interpretacao do score, mas nao
  recupera automaticamente padroes ictais que a representacao nao separa.

Nenhum desses pontos comprova ser a causa unica dos cinco eventos perdidos.
Alterar avaliador, amostragem e arquitetura simultaneamente impediria atribuir
o resultado; nesta rodada eles sao registrados e mantidos como controles.

### Verificacao espectral descritiva, somente no treino

Foram amostrados quatro contextos normais por EDF (descartando sobreposicao
com crises) e um centro por crise, usando janelas de 4 s e a mesma FFT/Hann.
A auditoria usa derivacoes bipolares presentes diretamente no EDF; registros
apenas referenciais sao listados como excluidos no JSON, sem inventar dados.

| Contexto | Janelas | Canal/janelas | Mediana energia fora 0,5-45 Hz | P95 fora da banda | Dominante >=45 Hz |
|---|---:|---:|---:|---:|---:|
| Normal | 303 | 6.921 | 5,22% | 31,87% | 2,51% |
| Centro ictal | 66 | 1.506 | 2,85% | 20,43% | 0,20% |

Ha influencia mensuravel de energia externa as bandas. Os valores nao provam
que remover frequencias externas melhore discriminacao: atividade cerebral e
artefatos podem coexistir nessas faixas e as amostras nao sao independentes.
Nao foi feita selecao de filtros usando pacientes historicos ou calibracao.

## Uma unica alteracao v58, registrada antes do treino

Ocultar aleatoriamente 10% dos mapas aprendidos por canal antes da fusao espacial.
A mascara e constante no tempo e nas features dentro de uma sequencia, evitando
introduzir transientes artificiais. Dropout padrao usa reescala durante treino.
Na inferencia a camada e identidade e nao acrescenta parametros treinaveis.

Hipotese: reduzir dependencia de canais dominantes e aprender evidencia distribuida.
Risco: crises focais podem perder seus poucos canais informativos no treino.
O resultado deve decidir se o risco compensou; nao se assume beneficio.

Controles: mesmos 77 EDFs, quatro grupos de calibracao, scaler, amostragem por
evento, pesos de borda/BCE, contexto 8 passos, 20 epocas e checkpoint por evento.
Sem balanceamento v57 e sem novos hard negatives. FA/h <= 0,75. Seed exploratoria
42, referencia v43 reproduzida na v57 com pesos identicos ao modelo oficial.

Continuar para seeds 43/44 apenas se a seed 42 recuperar pelo menos um evento
na calibracao (mais de 15/20), com F1 localizado >= 90,9%, F1 EDF >= 95,65%
e FA/h <= 0,75. Taxa 0,10 fixa: nao procurar outras taxas se falhar.

Para avaliacao historica unica, exigir medias pareadas superiores em eventos,
sem queda media de F1 localizado/F1 EDF, teto respeitado nas tres seeds e ganho
de eventos em pelo menos duas. Congelar seed/artefatos antes de acessar historico.
Sem esse criterio, rejeitar e encerrar treino desta rodada.

## Entregas

- Implementacao opt-in com default zero, preservando baseline.
- Teste de coerencia da mascara, identidade de inferencia e serializacao Keras.
- Resultado numerico e decisao registrados neste arquivo apos execucao.
- Modelo ativo e resultados oficiais do TCC permanecem congelados ate evidencia
  que justifique promover um novo modelo.

## Resultado da v58

20 epocas completas, seed 42, melhor checkpoint na epoca 16:

| Calibracao no teto 0,75 FA/h | V43 reproduzida | V58 |
|---|---:|---:|
| Eventos | 15/20 | 13/20 |
| Sensibilidade por evento | 75,0% | 65,0% |
| Precisao por evento | 68,2% | 59,1% |
| F1 por evento | 71,4% | 61,9% |
| F1 com localizacao principal | 90,9% | 80,0% |
| F1 EDF | 95,7% | 95,7% |
| Falsos alarmes | 7 | 9 |
| FA/h | 0,549 | 0,706 |
| Threshold / duracao / gap | 0,60 / 18 / 8 | 0,60 / 18 / 6 |

**REJEITAR no piloto.** A v58 perdeu o primeiro evento de `chb16_17` e
`chb20_13`, sem recuperar nenhum evento da baseline. Os picos nesses eventos
continuaram altos (aproximadamente 0,97 e 0,99), mas nao produziram candidatos
validos. Os demais tres eventos de `chb16_17` continuaram perdidos.

Nao foram treinadas seeds extras nem consultados EDFs historicos, pois o
criterio de continuidade definido antes do treino nao foi cumprido. Este
resultado rejeita esta configuracao; nao prova que toda regularizacao espacial
seja inutil. A v43 permanece como modelo ativo.

## Bug concreto corrigido: montagem na inferencia

A auditoria espectral encontrou `chb12_27.edf` sem nenhum slot bipolar direto
da montagem fixa. O treino ja reconstruia derivacoes por referencia comum
e invertia derivacoes quando necessario. A extracao da aplicacao nao repetia
essa regra: um registro referencial podia virar 345 features zeradas apesar
de possuir eletrodos suficientes para derivar a maior parte da montagem.

Correcao: `resolver_sinais_referencia` em `feature_extractor.py` agora e usado
no treino e na extracao global/janelada de inferencia. Prioriza derivacao
direta, depois inversao, depois subtracao com referencia comum. Respeita os
canais fontes selecionados pelo usuario; nao recria fontes desmarcadas.

Verificacao real, quatro segundos de `chb12_27.edf` (pertence ao treino):

- slots bipolares presentes diretamente: 0;
- slots recuperados pela regra compartilhada: 20 de 23;
- 300 de 345 features deixaram de ser zeradas;
- tres slots FT9/FT10 continuam corretamente omitidos;
- diferenca absoluta maxima entre features de treino e inferencia: **0,0**.

Foram aprovados 54 testes focados da montagem, treino e inferencia sequencial,
alem dos 16 testes da regularizacao/checkpoint/comparador. A suite completa
posterior terminou em **209 testes aprovados**, com seis avisos de nomes de
canais duplicados e estado de otimizador em modelos de teste. A primeira
execucao foi interrompida por bloqueio do socket interno do asyncio no sandbox;
a repeticao autorizada fora dessa restricao concluiu em 53,53 s.
A mudanca padroniza
a entrada em casos referenciais; nao e evidencia de ganho de sensibilidade em
crises e nao altera os pesos da v43. Os cinco eventos historicos perdidos
possuem os canais principais, portanto nao se atribui sua perda a este bug.

## Artefatos e reproducao

- `modelos/v58_channel_dropout_seed42_training.json`: treino e trajetoria.
- `modelos/v58_channel_dropout_comparison.json`: comparacao e todas as crises.
- `modelos/v58_channel_dropout_comparison.md`: tabela legivel por evento.
- `modelos/v58_channel_dropout_comparison.png`: FROC discreta.
- `modelos/v58_cohort_audit.json`: coorte por cabecalhos/anotacoes originais.
- `modelos/v58_spectral_scope_audit.json`: amostra espectral de treino.
- `modelos/v58_reference_montage_parity.json`: igualdade no EDF referencial real.

```powershell
.venv\Scripts\python.exe scripts\audit_v58_cohort.py
.venv\Scripts\python.exe scripts\audit_spectral_scope.py
.venv\Scripts\python.exe scripts\verify_reference_montage_parity.py
.venv\Scripts\python.exe scripts\summarize_channel_dropout.py --baseline modelos\v57_baseline_seed42_training.json --candidate modelos\v58_channel_dropout_seed42_training.json --output modelos\v58_channel_dropout_comparison.json
```

## Caminho pratico para os gaps que uma rodada nao resolve

Forca das conclusoes:

- Observado: crises com score baixo, troca de eventos entre seeds, dominio
  desigual de amostras, poucas pessoas e negativos completos na calibracao,
  diferencas de protocolo entre versoes e limites do avaliador atual.
- Hipotese a testar: dependencia excessiva de amplitude/topografia, impacto
  da energia fora da banda e beneficio de regularizacao por canal.
- Nao demonstrado: anotacoes CHB-MIT incorretas, bug unico que explique toda
  a queda, superioridade clinica ou recuperacao dos cinco eventos apenas
  alterando threshold. Nenhuma dessas alegacoes deve entrar como conclusao.

| Gap | Correcao verificavel | Como demonstrar |
|---|---|---|
| Especializacao nos quatro pacientes de calibracao | Separar selecao de epoca, escolha operacional e avaliacao; usar folds por pessoa no desenvolvimento | Mesmos folds/arquivos nas duas variantes, medias e dispersao por pessoa |
| Poucos negativos completos na calibracao | Incluir EDFs normais dos mesmos pacientes antes de congelar nova coorte | FA/h em EEG normal completo, alem de trechos normais de EDF positivo |
| Poucas morfologias distintas | Ampliar crises diferentes dos pacientes de treino e revisar cobertura de bandas/topografias | Contagem de eventos por pessoa e desempenho em pacientes fora da selecao |
| Negativos de treino fixos | Rotacionar amostras normais diversas por epoca com orcamento fixo | Comparacao com mesmas atualizacoes; nao confundir com v55 de maior score |
| Espectro fora das bandas | Definir extrator versionado com dominio espectral consistente | Testar PSD/features, refazer cache e retreinar A/B; nunca trocar extrator sob pesos antigos |
| Erro de pareamento/fragmentacao nas metricas | Acrescentar avaliador 1:1 e cobertura temporal por anotacao original | Casos sinteticos: dois alarmes/uma crise, um alarme/duas crises, fragmentos e ausencia de alarme |
| Score saturado | Medir confiabilidade e calibrar probabilidades em dados exclusivos para isso | Brier/reliability curve; distinguir de otimizacao do threshold |
| Candidato principal errado | Avaliar recall de todos os candidatos separado do ranking | Relatar principal correto e qualquer candidato correto; confirmador so apos geracao adequada |

Essas mudancas nao devem ser acumuladas sob uma unica versao sem ablacao.
Com todos os grupos locais ja consultados no desenvolvimento, nenhuma nova
consulta aos mesmos registros pode receber o nome de validacao externa intocada.
Mais seeds reduzem incerteza de otimizacao, nao criam novas pessoas na amostra.
