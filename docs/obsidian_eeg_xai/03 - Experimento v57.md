# Experimento v57 | Balanceamento por paciente

## Pergunta

Reduzir a dominancia de pacientes com muitas janelas positivas melhora a generalizacao sem retirar sensibilidade?

## Alteracao controlada

Foi aplicado um fator moderado por paciente e classe, com expoente 0,5. A soma de pesos de cada classe foi preservada. O sampler, a arquitetura, a loss, o conjunto de dados e o ponto de comparacao foram mantidos.

## Calibracao

| Medida | v43 | v57 |
| --- | ---: | ---: |
| Eventos medios | 14,0 | 14,0 |
| F1 localizado | 83,44% | 83,64% |
| F1 por EDF | 97,10% | 97,10% |
| FA/h | 0,63 | 0,55 |
| Decisao | baseline | promissora, nao superior |

## Avaliacao historica

| Medida | v43 | v57 |
| --- | ---: | ---: |
| Eventos | 4/9 | 4/9 |
| F1 localizado | 46,2% | 53,3% |
| F1 por EDF | 82,4% | 62,5% |
| FA/h | 2,08 | 1,04 |
| Crises recuperadas | - | 0 |

## Decisao

**Nao promover.** A v57 reduziu os falsos alarmes pela metade e melhorou a medida de localizacao, mas nao recuperou nenhum dos eventos perdidos e piorou o F1 por EDF. Ela deve permanecer como evidencia de investigacao, nao como modelo final.

## Relatorio completo

`docs/ia/32_experimento_v57_balanceamento_pacientes.md`
