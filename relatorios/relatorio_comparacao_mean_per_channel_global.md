# Comparacao global CNN-LSTM: mean vs per_channel
## Objetivo
Comparar duas formas de entrada para o CNN-LSTM sequencial no objetivo principal do projeto: treinar em pacientes conhecidos e testar em paciente nunca visto.
## Correcoes metodologicas aplicadas
- A validacao agora separa pacientes de treino, pacientes de calibracao e paciente de teste.
- O threshold e a duracao minima sao escolhidos em pacientes de calibracao, nao no paciente de teste.
- Foi adicionado cache de extracao de features por EDF para reduzir retrabalho entre folds.
- O modo `per_channel` usa uma montagem fixa de canais de referencia.
## Resultado: feature_mode=mean
| Paciente teste | EDFs crise | EDFs normais | Crises localizadas | FP | FN | EDF F1 | Window F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| chb01 | 3 | 1 | 2 | 0 | 0 | 1.000 | 0.149 |
| chb03 | 2 | 1 | 1 | 1 | 0 | 0.800 | 0.168 |
| chb13 | 7 | 2 | 2 | 2 | 0 | 0.875 | 0.038 |
| chb14 | 6 | 1 | 0 | 1 | 0 | 0.923 | 0.005 |

Totais `mean`:
- Crises localizadas: 5 de 18 EDFs com crise
- Falsos positivos: 4 em 5 EDFs normais
- F1 medio por EDF: 0.900
- F1 medio por janela: 0.090
- Pacientes solicitados mas ausentes no manifesto usado: chb11, chb15

## Resultado: feature_mode=per_channel
A validacao completa `per_channel` ainda ficou pesada para rodar em todos os folds neste ambiente. Foi executado um fold comparavel em `chb03`, com volume reduzido de janelas.
- chb03: crises localizadas 2/2, FP=1, FN=0, EDF F1=0.800, Window F1=0.200.

## Interpretacao
O modo `mean` ficou mais rapido e teve boa classificacao por EDF em alguns pacientes, mas ainda localizou poucas crises reais: 5 de 18 EDFs com crise. Isso mostra que acertar o arquivo como positivo nao basta; a localizacao temporal continua sendo o ponto fraco.
O modo `per_channel` preserva mais informacao espacial, mas ficou muito mais caro. No fold `chb03`, localizou as duas crises reais, mas ainda gerou falso positivo no EDF normal. Portanto, ele e promissor, mas precisa de uma estrategia de treino/validacao mais eficiente antes de substituir o `mean`.
## Proximo passo recomendado
O proximo ajuste deve ser reduzir o custo do `per_channel` e melhorar a localizacao temporal: salvar/cachear features por EDF em disco, treinar com menos folds inicialmente e adicionar uma metrica que penalize positivo sem sobreposicao com a crise real. Depois disso, rodar `per_channel` em todos os pacientes do manifesto curado.
