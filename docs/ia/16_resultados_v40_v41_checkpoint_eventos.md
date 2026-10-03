# v40-v41: checkpoint por evento e peso para crises curtas

## Protocolo

Os experimentos mantiveram o manifesto v38, sampler por evento da v39,
CNN-BiLSTM, weighted BCE, normalizacao, tres seeds e os mesmos quatro pacientes
de calibracao. `chb09`, `chb15` e `chb18` nao foram carregados. O Random Forest
permaneceu desligado.

- v40: restaura a epoca com maior recall de evento sob `FA/h <= 0,50`.
- v40b: repete a selecao sob `FA/h <= 0,75`.
- v41: volta ao checkpoint por `val_pr_auc` e aplica peso `1,5x` somente aos
  passos ictais de eventos com duracao real de ate 30 segundos.

O callback v40 registra todas as epocas, ponto operacional, restricao de FA/h e
pesos restaurados. A duracao usada na v41 vem da anotacao real do evento; eventos
longos nao sao reduzidos e nenhuma amostra e duplicada.

## Resultado comparavel

| Variante | Restricao | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h |
|---|---:|---:|---:|---:|---:|---:|
| v39 | FROC `<= 1,00` | 12,33/20 | 64,7% | 75,8% | 90,9% | 0,47 |
| v40 | checkpoint e operacao `<= 0,50` | 12,33/20 | 68,0% | 77,9% | 89,2% | 0,31 |
| v40/v40b | checkpoint/operacao `<= 0,75` | 13,33/20 | 65,6% | 75,0% | 92,9% | 0,58 |
| v40 | modelo selecionado, FROC `<= 1,00` | 13,67/20 | 64,1% | 76,6% | 92,6% | 0,71 |
| v41 | FROC `<= 1,00` | 12,00/20 | 61,7% | 73,7% | 90,9% | 0,55 |

O valor v40 em `<= 0,50` usa `checkpoint_best_metrics`; o valor em `<= 1,00`
usa o mesmo modelo restaurado e a grade FROC padrao. Eles sao pontos
operacionais diferentes do mesmo conjunto de pesos, nao dois modelos.

## Variacao por seed

No limite estrito de `0,50`, a v40 encontrou 11, 12 e 14 eventos. A seed 44
atingiu 14/20 com `0,16 FA/h`, enquanto a seed 42 ficou em 11/20 com `0,47
FA/h`. No limite `0,75`, os resultados foram 13, 12 e 15 eventos. A variacao
mostra que a inicializacao ainda afeta fortemente o compromisso entre recall e
localizacao.

A v40b selecionou as mesmas epocas da v40 nas tres seeds (3, 1 e 6). Assim,
trocar apenas o teto de 0,50 para 0,75 nao criou um checkpoint novo; apenas
escolheu outro ponto operacional na mesma FROC.

## Decisao

### v40: PROMISSOR MAS NAO SUPERIOR

Sob `FA/h <= 0,50`, nao aumentou a media de eventos em relacao a v39, embora
tenha melhorado F1 de evento, localizacao e alarmes. Sob `FA/h <= 0,75`, ganhou
1,33 evento em media e manteve localizacao proxima de 75%, ao custo de elevar
FA/h. E um candidato de compromisso, mas ainda nao justifica promocao nem acesso
ao conjunto reservado.

### v41: REJEITAR

O peso moderado de crises curtas nao recuperou eventos: todas as seeds ficaram
em 12/20 e a localizacao caiu. Como v41 nao ajudou, ela nao deve ser combinada
com v40. Isso evita atribuir eventual mudanca a duas variaveis simultaneas.

## Proximo gargalo

O ganho isolado da seed 44 e a ausencia de ganho medio no teto estrito indicam
variancia de otimizacao e representacao insuficiente de alguns eventos, nao
apenas falta de peso. Channel attention simples ja piorou na v31 e nao deve ser
repetida sem uma hipotese nova.

O proximo experimento deve auditar quais eventos de calibracao mudam entre as
seeds 42 e 44 e comparar seus scores ictais antes de alterar novamente a rede.
Se os mesmos eventos permanecerem invisiveis em todas as seeds, o proximo teste
justificavel e uma representacao espacial por canal mais conservadora que a
attention da v31, mantendo o sampler v39 e o checkpoint v40. Se os eventos
alternarem por seed, a prioridade passa a ser estabilidade de treino e
ensemble de checkpoints somente dentro da mesma seed, sem ensemble de modelos
heterogeneos.

Essa auditoria foi concluida em `docs/ia/17_auditoria_estabilidade_v40.md`.
Ela separou perdas por duracao minima, representacao fraca, fragmentacao e
variacao entre seeds.

Artefatos:

- `modelos/v40_event_checkpoint_calibration.json`
- `modelos/v40b_event_checkpoint_fa075_calibration.json`
- `modelos/v41_short_event_weight_calibration.json`
