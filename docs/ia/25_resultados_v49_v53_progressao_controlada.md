# Resultados v49-v53: progressao controlada da multirresolucao

## Protocolo

- treino e calibracao separados por paciente;
- calibracao fixa: `chb16`, `chb19`, `chb20` e `chb24`;
- 13 EDFs completos, 20 eventos e 11.587 sequencias na calibracao;
- `chb09`, `chb15` e `chb18` nao foram carregados;
- comparacao principal no ponto FROC com `FA/h <= 0,75`;
- tres seeds: 42, 43 e 44.

## v49: validacao completa do prototipo

A v49 comparou a v43 reduzida e a v47 multirresolucao treinadas nos mesmos 11
EDFs. A v47 melhorou modestamente evento e falsos alarmes, mas nao localizacao
ou F1 por EDF.

| Metrica media | v43 reduzida | v49 multirresolucao |
|---|---:|---:|
| eventos | 10,33/20 | 10,67/20 |
| F1 evento | 53,3% | 56,0% |
| F1 localizado | 73,7% | 73,5% |
| FA/h | 0,654 | 0,576 |
| F1 EDF | 87,1% | 85,5% |

Decisao: a informacao complementar existe, mas o treino era pequeno demais.

## v50: diversidade dirigida

O treino foi ampliado de 11 para 23 EDFs, incluindo seis novos pacientes e
eventos curtos/medios. Baseline e multirresolucao foram retreinadas nas mesmas
1.372 sequencias.

| Metrica media | resolucao unica | multirresolucao |
|---|---:|---:|
| eventos | 10,67/20 | 12,00/20 |
| F1 evento | 55,8% | 60,5% |
| F1 localizado | 68,7% | 75,4% |
| FA/h | 0,576 | 0,602 |
| F1 EDF | 81,7% | 85,5% |

Conclusao: diversidade de pacientes tornou o ganho da segunda resolucao mais
claro. A arquitetura passou a recuperar mais eventos sem ultrapassar o teto de
falsos alarmes.

## v51: checkpoint orientado a evento

O modelo deixou de usar obrigatoriamente a ultima epoca. As epocas 4, 8, 12 e
16 foram comparadas por evento/localizacao sob `FA/h <= 0,75`.

| Metrica | v50 | v51 |
|---|---:|---:|
| eventos | 12,00/20 | 12,00/20 |
| precisao evento | 61,1% | 65,6% |
| F1 evento | 60,5% | 62,6% |
| F1 localizado | 75,4% | 75,4% |
| FA/h | 0,602 | 0,497 |
| F1 EDF | 85,5% | 85,5% |

O checkpoint melhorou principalmente falsos alarmes. A seed 42 caiu de 9 para
5 falsos alarmes sem perder eventos.

## v52: auditoria dos erros persistentes

Sete eventos falharam nas tres seeds, concentrados em `chb16` e `chb20`. A
comparacao de assinaturas apontou como vizinhos de treino mais relevantes:

- `chb03_34.edf`;
- `chb04_28.edf`;
- `chb23_06.edf`;
- `chb13_55.edf`, que ja estava no treino.

Isso indicou falta de diversidade morfologica, nao apenas threshold inadequado.

## v53: expansao por similaridade morfologica

Os tres EDFs ausentes encontrados na v52 foram adicionados. O treino passou a
26 EDFs e 1.589 sequencias pareadas.

| Seed | eventos | F1 localizado | FA/h | F1 EDF |
|---|---:|---:|---:|---:|
| 42 | 12/20 | 73,7% | 0,549 | 90,9% |
| 43 | 13/20 | 80,0% | 0,628 | 95,7% |
| 44 | 12/20 | 80,0% | 0,628 | 90,9% |
| media | 12,33/20 | 77,9% | 0,602 | 92,5% |

Comparada a v51, a v53 melhorou eventos, localizacao e F1 EDF em duas seeds,
mas reduziu a precisao media por evento e elevou FA/h de 0,497 para 0,602.

## Decisao

**PROMISSORA, MAS NAO SUPERIOR.** A v53 nao supera a referencia historica v43
em eventos e localizacao (`13,8/20`, `81,9%`), embora mantenha FA/h menor.
Nenhum modelo foi promovido para producao.

## Proximo gargalo

Os erros restantes nao devem ser atacados com novo ajuste livre de threshold.
O proximo experimento deve:

1. produzir diagnostico por evento no ponto exato de `0,75 FA/h`;
2. separar eventos com score baixo de eventos fragmentados;
3. minerar hard negatives dos 7-8 falsos alarmes restantes;
4. testar uma perda de continuidade/borda somente se os erros forem de
   fragmentacao;
5. ampliar treino apenas com pacientes/morfologias que cubram os eventos ainda
   invisiveis, mantendo a avaliacao reservada intacta.

Artefatos principais:

- `modelos/v49_multiresolution_full_calibration.json`;
- `modelos/v50_multiresolution_diverse_training.json`;
- `modelos/v51_multiresolution_event_checkpoint_*.json`;
- `modelos/v52_audit_chb20_*.json`;
- `modelos/v53_targeted_morphology_*.json`.
