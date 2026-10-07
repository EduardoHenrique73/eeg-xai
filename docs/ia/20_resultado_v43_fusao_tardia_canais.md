# Resultado v43: frontend por canal com fusao espacial tardia

Atualizado em: 2026-10-06

## Hipotese e controle

A baseline achatava `canais x features` antes da primeira `Conv1D`. A v43
preserva `[tempo, canal, feature]` durante duas convolucoes temporais
compartilhadas, mascara canais ausentes e realiza a fusao espacial somente
depois. O nucleo CNN-BiLSTM, a weighted BCE, os dados, o sampler por evento, a
sequencia de oito passos e a calibracao permaneceram fixos.

- arquitetura: 84.081 parametros;
- treino: manifesto v38;
- calibracao: `chb16`, `chb19`, `chb20` e `chb24`, 20 eventos;
- ponto operacional: FROC sob `FA/h <= 0,75`;
- seeds principais: 42, 43, 44, 45 e 46;
- pacientes historicamente reservados: nao acessados.

## Resultado das cinco seeds

| Seed | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h | Melhor epoca |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 15/20 | 71,4% | 90,9% | 95,7% | 0,55 | 14 |
| 43 | 14/20 | 66,7% | 85,7% | 100,0% | 0,63 | 14 |
| 44 | 13/20 | 61,9% | 73,7% | 95,7% | 0,71 | 1 |
| 45 | 12/20 | 58,5% | 73,7% | 90,9% | 0,71 | 1 |
| 46 | 15/20 | 69,8% | 85,7% | 100,0% | 0,63 | 8 |
| **Media** | **13,8/20** | **65,7%** | **81,9%** | **96,4%** | **0,64** | - |

Desvios entre seeds: 1,30 evento, 7,8 pontos de F1 localizado, 3,8 pontos de
F1 EDF e 0,066 FA/h.

## Comparacao controlada com v40b

Nas seeds 42-44 e no mesmo ponto `FA/h <= 0,75`:

| Modelo | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h |
|---|---:|---:|---:|---:|---:|
| baseline CNN-BiLSTM | 13,33/20 | 65,6% | 75,0% | 92,9% | 0,58 |
| v43 fusao tardia | 14,00/20 | 66,7% | 83,4% | 97,1% | 0,63 |

A v43 melhora eventos, localizacao e classificacao de EDF, com aumento de
0,05 FA/h. O ganho ocorreu nas seeds 42 e 43; a seed 44 trocou eventos em vez
de aumentar o total.

## Auditoria por evento

Ganhos consistentes nas cinco seeds:

- `chb16_17`, evento 1: passou de instavel para detectado em 5/5; score medio
  subiu aproximadamente de 0,61 para 0,94;
- `chb20_12`, evento 1: detectado em 4/5; score medio de 0,87;
- `chb20_13`: detectado em 4/5; score medio de 0,99.

Gargalos persistentes:

- `chb16_17`, eventos 2 e 3: 0/5, scores medios 0,21 e 0,18;
- `chb16_17`, evento 4: 0/5, score medio 0,47 e alta variacao;
- `chb16_11`: 0/5 apesar de score medio 0,79, bloqueado principalmente pela
  curta duracao/fragmentacao;
- `chb24_04`, evento 3: score proximo de 1,0, mas apenas 1/5 deteccoes, indicando
  evidencia temporal insuficientemente continua.

## Auditoria da seed 45

Uma execucao separada aumentou paciencia para 12 e limite para 30 epocas. Ela
recuperou 14/20 na epoca 17, mas o F1 EDF caiu para 87,0% e apareceu um EDF
falso positivo. Logo, early stopping curto explica parte do recall, mas treinar
mais apenas desloca o erro e nao resolve a estabilidade.

## Decisao

**PROMISSOR, MAS NAO SUPERIOR.** A representacao espacial tardia tem evidencia
real de utilidade e deve substituir a linha arquitetural da v31 como principal
candidata de pesquisa. Ainda nao deve ser promovida porque:

- a media ficou em 13,8, abaixo da meta de 14/20;
- a pior seed caiu para 12/20;
- FA/h medio de 0,64 ficou acima da meta desejada de 0,60;
- tres eventos de `chb16_17` continuam invisiveis;
- os pacientes reservados nao foram usados para confirmacao final.

## Proximo experimento recomendado

Manter a v43 e alterar apenas a estabilidade da otimizacao:

1. usar reducao de learning rate por plateau ou learning rate menor;
2. selecionar checkpoint por criterio multiobjetivo que nao sacrifique F1 EDF;
3. comparar com as mesmas cinco seeds;
4. nao aumentar capacidade, contexto, loss ou pos-processamento nessa rodada;
5. somente depois estudar uma branch local para os eventos 2-4 de `chb16_17`.

Artefatos:

- `modelos/v43_late_channel_fusion_calibration.json`;
- `modelos/v43_late_channel_fusion_confirmation.json`;
- `modelos/v43b_late_fusion_seed45_patience.json`.
