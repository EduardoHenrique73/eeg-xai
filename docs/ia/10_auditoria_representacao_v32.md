# v32: auditoria da representacao e mascara de canais

## Objetivo e isolamento

Este experimento compara a CNN-BiLSTM atual com a mesma rede recebendo 23
indicadores explicitos de presenca dos canais de referencia. Scaler, treino,
weighted BCE, early stopping, grid de calibracao e pos-processamento sao
identicos. As seeds pareadas sao 42, 43 e 44. O Random Forest permanece
desligado. O script e `scripts/experiment_channel_mask_v32.py`, e o resultado
completo e `modelos/v32_channel_mask_calibration.json`.

Foram usados 68 EDFs de treino (4.053 sequencias, 789 positivas) e 13 EDFs de
calibracao (11.587 sequencias, 153 positivas pelo rotulo central). `chb09`,
`chb15` e `chb18` nao foram carregados no A/B. `chb01` e `chb21` permanecem
juntos no treino porque sao registros da mesma pessoa, conforme a descricao
oficial do CHB-MIT: https://physionet.org/content/chbmit/1.0.0/.

## Auditoria de canais

Somente dois EDFs de treino, `chb13_40` e `chb13_47`, nao tem seis dos 23
canais de referencia. Nenhum EDF de calibracao tem canal de referencia
ausente. Assim, a calibracao nao mede diretamente robustez a canais ausentes;
ela mede apenas o efeito indireto dos indicadores na rede. Os canais ausentes
sao preenchidos com zero antes do `StandardScaler`, mas esse zero passa a um
valor escalado nao nulo (aproximadamente 0,46 a 0,50 em magnitude para a
primeira feature desses seis canais). A v32 anexa a mascara obtida dos
cabecalhos EDF apos aplicar o mesmo scaler nos dois ramos. A producao nao foi
alterada.

Checagem descritiva, sem selecao de modelo, dos EDFs dificeis: `chb15_10`,
`chb15_17`, `chb15_20` e `chb18_30` contem todos os canais de referencia. Os
intervalos anotados cabem integralmente nos EDFs de 3.600 s. `chb15` possui
canais extras CP/FC e `PZ-OZ` ignorados pela representacao fixa atual; isso
nao prova que sejam necessarios para detectar essas crises.

## Auditoria de saturacao

Fracao de features de potencia absoluta no passo central que atingiu
`abs(valor) >= 9,999`, apos normalizacao robusta por EDF com clipping em 10:

| Grupo | Sequencias | Potencia absoluta saturada | Todas as features saturadas |
|---|---:|---:|---:|
| Treino normal | 3.264 | 2,32% | 1,14% |
| Treino ictal | 789 | 25,45% | 9,65% |
| Calibracao normal | 11.434 | 1,35% | 0,65% |
| Calibracao ictal | 153 | 42,14% | 15,49% |

As sequencias se sobrepoem e nao equivalem a eventos independentes. Nos EDFs
reservados ja conhecidos, a mesma auditoria descritiva encontrou apenas
1,30% em `chb15_10` e 1,85% em `chb18_30`; `chb15_17` teve 1,93% e
`chb15_20` 10,06%. Isso e compativel com a hipotese de que a rede se apoia
demais em evidencia de potencia alta. Nao demonstra que o clipping cause os
falsos negativos: as crises perdidas quase nao chegam ao limite. Portanto,
simplesmente aumentar o limite de clipping nao e uma correcao justificada.

## A/B somente na calibracao

Cada linha usa seu melhor ponto na **mesma** grade de calibracao, com
FA/h <= 1 como restricao. Nao ha avaliacao reservada nem promocao.

| Seed | Entrada | Eventos | Alarmes falsos | FA/h | F1 localizado | F1 EDF |
|---:|---|---:|---:|---:|---:|---:|
| 42 | Atual | 13/20 | 12 | 0,94 | 73,7% | 90,9% |
| 42 | Mascara | 12/20 | 12 | 0,94 | 73,7% | 90,9% |
| 43 | Atual | 12/20 | 11 | 0,86 | 63,2% | 81,8% |
| 43 | Mascara | 12/20 | 9 | 0,71 | 66,7% | 90,9% |
| 44 | Atual | 13/20 | 11 | 0,86 | 73,7% | 90,9% |
| 44 | Mascara | 13/20 | 10 | 0,78 | 73,7% | 90,9% |
| Media | Atual | 12,67/20 | 11,33 | 0,89 | 70,2% | 87,9% |
| Media | Mascara | 12,33/20 | 10,33 | 0,81 | 71,3% | 90,9% |

A seed 42 da baseline reproduziu exatamente a ablação A da v31. A mascara
reduziu discretamente os alarmes em duas seeds, mas perdeu um evento na
terceira. A variacao entre seeds e o uso dos mesmos pacientes de calibracao
para early stopping e escolha do ponto operacional impedem concluir ganho
robusto. A mascara nao deve ser ativada no modelo em uso apenas com esses
resultados.

## Proximo experimento justificavel

O alvo principal nao e mais a ausencia de canais nem aumentar o teto de
clipping. Antes de outra arquitetura, comparar **na calibracao** uma entrada
menos dependente de potencia absoluta: potencia relativa, mudanca temporal
por canal ou um ramo curto de sinal bruto, preservando o mesmo split e o
mesmo pos-processamento. Selecionar uma alteracao por vez e avaliar crises
de baixa evidencia presentes no treino/calibracao. Falsos alarmes de treino
precisam de revisao e categorias anotadas antes de hard-negative mining
clinicamente interpretavel. O conjunto `chb09/chb15/chb18` ja foi consultado
repetidamente e nao serve como teste final independente.
