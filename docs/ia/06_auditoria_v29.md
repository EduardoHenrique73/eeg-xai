# Auditoria temporal da v29

## Configuracao

- hipotese: 96 sequencias normais por EDF prejudicavam recall e localizacao;
- unica mudanca principal: limite normal reduzido para 48;
- escolha: feita somente em `chb16`, `chb19`, `chb20` e `chb24`;
- treino: 17 pacientes, sem `chb09`, `chb15` ou `chb18`;
- teste de desenvolvimento reservado: `chb09`, `chb15` e `chb18`;
- seed base: 42; seeds efetivas das dobras: 43, 44 e 45;
- modelo ativo: nao alterado.

## Resultado agregado

| Metrica | v23b | v27 | v28 confirmador | v29 |
|---|---:|---:|---:|---:|
| F1 EDF | 0.867 | 0.767 | 0.822 | 0.867 |
| F1 localizado | 0.667 | 0.556 | 0.611 | 0.611 |
| Trechos principais | 5/9 | 4/9 | - | 5/9 |
| Eventos detectados | 6/9 | 7/9 | 6/9 | 7/9 |
| FA/h | 1.362 | 1.687 | 0.584 | 1.297 |

Precisao EDF da v29: `0.8222`; recall EDF: `0.9333`; FP: 2; FN: 1.
Precisao de evento macro: `0.2556`; recall de evento macro: `0.7667`; F1 de
evento macro: `0.3683`; falsos alarmes: 20.

## Localizacao por EDF positivo

| Arquivo | Real (s) | Principal (s) | IoU | Inicio (s) | Fim (s) | Media ictal | Maximo ictal |
|---|---:|---:|---:|---:|---:|---:|---:|
| chb09_06 | 12230-12296 | 1238-1438 | 0.00 | -10992 | -10858 | 0.955 | 1.000 |
| chb09_19 | 5298-5362 | 5296-5358 | 0.91 | -2 | -4 | 0.923 | 1.000 |
| chb15_06 | 270-398 | 276-318 | 0.33 | 6 | -80 | 0.382 | 0.969 |
| chb15_10 | 1080-1114 | 2468-2476 | 0.00 | 1388 | 1362 | 0.105 | 0.317 |
| chb15_17 | 1924-1962 | 1928-1954 | 0.68 | 4 | -8 | 0.555 | 0.936 |
| chb15_20 | 606-664 | 82-142 | 0.00 | -524 | -522 | 0.633 | 0.969 |
| chb15_31 | 1750-1872 | 1756-1862 | 0.87 | 6 | -10 | 0.861 | 0.996 |
| chb18_29 | 3476-3528 | 3478-3598 | 0.41 | 2 | 70 | 0.956 | 0.996 |
| chb18_30 | 540-572 | 2968-3032 | 0.00 | 2428 | 2460 | 0.093 | 0.243 |

`Inicio` e `Fim` sao diferencas `previsto - real`. Valores negativos indicam
antecipacao e positivos indicam atraso.

## Falhas principais

### chb09_06: categoria B, candidato principal deslocado

A crise real teve score alto e foi detectada no segundo candidato, com overlap
de 66 segundos e razao 1.0. Um falso trecho de 200 segundos foi ranqueado como
principal. O problema e selecao do candidato, nao representacao.

### chb15_20: categoria B, candidato principal deslocado

A crise foi detectada em candidatos secundarios, incluindo um trecho com 26
segundos de overlap. Trechos falsos no inicio do EDF receberam evidencia maior e
ocuparam as primeiras posicoes.

### chb15_10: categoria C, score baixo durante toda a crise

Media ictal `0.105`, maximo `0.317` e nenhuma janela acima do threshold
calibrado. Pos-processamento ou confirmador nao recuperam essa crise.

### chb18_30: categoria C, score baixo durante toda a crise

Media ictal `0.093`, maximo `0.243` e nenhuma janela acima do threshold
calibrado. A falha tambem esta na representacao do primeiro estagio.

## Limitacao do artefato

A v29 registra o numero de janelas ictais acima do threshold selecionado, mas
nao conserva os scores por janela para recalcular essa contagem em todos os
thresholds depois do treino. Experimentos futuros devem salvar a auditoria por
evento durante a avaliacao, incluindo maior bloco continuo em cada threshold.

## Decisao

Rejeitar para promocao. A v29 preservou 7/9 eventos e recuperou o F1 por EDF da
v23b, mas nao superou F1 localizado `0.667` e permaneceu acima de 1 FA/h.
