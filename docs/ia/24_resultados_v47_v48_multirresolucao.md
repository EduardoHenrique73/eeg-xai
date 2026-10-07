# Resultados v47 e v48: multirresolucao

## Objetivo

Testar a hipotese da auditoria v46 sem substituir globalmente janelas de 4 s.
O caminho principal continua usando 4 s/2 s e uma segunda branch recebe 2 s/1 s.
O alvo permanece uma probabilidade por passo da grade de 4 s.

Este experimento e exploratorio. `chb16_17.edf` pertence a calibracao, foi
inspecionado por evento e nao autoriza promocao. Os pacientes reservados
`chb09`, `chb15` e `chb18` nao foram carregados.

## Correcao metodologica

Na primeira execucao, cada resolucao foi amostrada independentemente e apenas
374 sequencias coincidiram. O protocolo final amostra somente a branch de 4 s
e busca seus contextos na base completa de 2 s. Assim, 687/687 sequencias de
treino foram pareadas. Os dois modelos comparados receberam exatamente os
mesmos alvos temporais e pesos.

## v47: fusao por concatenacao

Medias de tres seeds no `chb16_17`:

| Medida | v43 reduzida | v47 |
|---|---:|---:|
| percentil 95 normal | 0,108 | 0,037 |
| percentil 99 normal | 0,948 | 0,991 |
| evento 1, maximo | 0,997 | 1,000 |
| evento 2, maximo | 0,317 | 0,666 |
| evento 3, maximo | 0,543 | 0,719 |
| evento 4, maximo | 0,837 | 0,988 |

A evidencia media dentro dos eventos 2 e 3 tambem aumentou. A branch curta
portanto recuperou informacao ictal que a resolucao unica diluia. Entretanto,
o percentil 99 normal piorou: poucos transientes normais continuam recebendo
scores extremos. O experimento e classificado como **PROMISSOR MAS NAO
SUPERIOR** ate existir avaliacao FROC em toda a calibracao.

## v48: fusao residual inicialmente nula

A v48 preserva o caminho grosso e adiciona uma projecao fina inicializada em
zero. A intencao era impedir que a branch nova destruisse a representacao da
v43. O resultado foi instavel entre seeds: o percentil normal caiu muito nas
seeds 42 e 44, mas diferentes eventos desapareceram nas seeds 42 e 43. A seed
43 terminou com loss 0,0214, muito acima das demais.

Decisao: **REJEITAR v48**. Inicializacao residual, sozinha, nao estabilizou o
aprendizado multirresolucao.

## Interpretacao

1. A resolucao de 2 s contem sinal complementar real para crises curtas.
2. O ganho nao e uniforme por seed nem por evento.
3. O problema imediato deixou de ser apenas recuperar score ictal: e separar
   esses eventos dos raros transientes normais com score proximo de 1.
4. Threshold escolhido no EDF alvo mascararia o problema e nao deve ser usado.

## Proximo experimento controlado

Executar a v47 no treino completo e em todos os pacientes de calibracao, com o
mesmo pos-processamento da v43. Comparar FROC, eventos, F1 localizado e FA/h.
O modelo so avanca se o ganho de eventos ocorrer em pelo menos duas seeds sem
ultrapassar o teto de 0,75 FA/h. Hard negatives devem ser extraidos apenas da
calibracao/treino e categorizados antes de qualquer confirmador.

Artefatos:

- `modelos/v47_multiresolution_prototype.json`;
- `modelos/v48_residual_multiresolution_prototype.json`;
- `scripts/experiment_multiresolution_v47.py`.
