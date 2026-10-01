# Experimento v30: confirmador FROC sobre gerador v29

## Auditoria anterior a mudanca

O confirmador anterior recebia 12 features: duracao, numero de janelas, media,
maximo, desvio padrao, percentil 90, minimo, evidencia do primeiro estagio,
contrastes globais, cobertura do EDF e densidade de runs. Os candidatos eram
positivos quando tinham ao menos 1 segundo de overlap e razao de overlap de
0,25. O Random Forest era treinado com candidatos da calibracao em LOPO, mas o
threshold era ordenado primeiro por F1 de evento, sem impor o teto de FA/h. O
candidato principal era escolhido pela probabilidade do confirmador e, em
empate, pela media do primeiro estagio.

## Mudanca controlada

- gerador v29 mantido com seed, arquitetura, features, amostragem, pacientes e
  limite de 48 sequencias normais por EDF;
- 40 features do confirmador cobrindo distribuicao, continuidade, forma e
  contexto local dos scores;
- hard negatives exclusivamente da calibracao;
- probabilidades de calibracao estritamente LOPO por paciente;
- selecao FROC sob `FA/h <= 1,0`, com analise secundaria em 0,50, 0,75, 1,00,
  1,25 e 1,50;
- comparacao de ranking por probabilidade, duracao, continuidade e combinacao;
- threshold e ranking escolhidos somente na calibracao OOF;
- modelo ativo e artefatos anteriores preservados.

## Comparacao

| Versao | F1 EDF | F1 localizado | Eventos | Principais corretos | FA/h |
|---|---:|---:|---:|---:|---:|
| v23b | 86,7% | 66,7% | 6/9 | 5/9 | 1,36 |
| v27 | 76,7% | 55,6% | 7/9 | 4/9 | 1,69 |
| v28 | 82,2% | 61,1% | 6/9 | nao registrado | 0,58 |
| v29 | 86,7% | 61,1% | 7/9 | 5/9 | 1,30 |
| v30 | 93,3% | 61,1% | 5/9 | 4/9 | 0,78 |

## Resultado no formato de decisao

Versao: v30

Alteracao: confirmador Random Forest enriquecido, calibracao LOPO, selecao FROC
restrita e quatro politicas de ranking, sem alterar o gerador v29.

Hipotese: features temporais e contexto local reduziriam falsos alarmes e
priorizariam candidatos ictais sem perder os 7/9 eventos do primeiro estagio.

Resultados:
- F1 EDF: 93,3%
- precisao EDF: 93,3%
- sensibilidade EDF: 93,3%
- F1 localizado: 61,1%
- eventos detectados: 5/9
- eventos principais corretos: 4/9
- falsos alarmes: 12
- FA/h: 0,778

Comparacao com v29:
- melhorou: F1 EDF de 86,7% para 93,3% e FA/h de 1,30 para 0,78.
- piorou: eventos de 7/9 para 5/9 e principais corretos de 5/9 para 4/9; F1
  localizado permaneceu em 61,1%.

Casos:
- chb09_06: evento continuou presente, mas o falso trecho 1236-1438 s seguiu
  como principal; ranking nao corrigido.
- chb15_10: nenhum candidato confirmado; evidencia do gerador permanece fraca.
- chb15_20: candidato ictal foi eliminado e um falso trecho 0-146 s foi
  mantido como principal.
- chb18_30: nenhum evento ictal confirmado; dois falsos candidatos foram
  mantidos, com principal em 2918-3038 s.

Decisao: REJEITAR.

Justificativa: embora a restricao agregada de FA/h tenha sido atingida e o F1
por EDF tenha subido, a versao perdeu dois dos sete eventos da v29. O criterio
metodologico proibe promover uma reducao para 6/9; 5/9 e ainda pior.

Proximo gargalo: as probabilidades LOPO do confirmador variam entre pacientes e
nao separam candidatos ictais dificeis de falsos candidatos persistentes. O
proximo experimento deve manter o teste intocado e melhorar a diversidade da
coorte de calibracao/hard negatives, ou usar um confirmador temporal que modele
a sequencia completa do candidato. Os casos chb15_10 e chb18_30 continuam sendo
falhas de representacao do primeiro estagio, nao de threshold.

## Artefato

`modelos/sequence_lopo_tf_hybrid_segmentation_v30_confirmifier_froc_fixed_cal_reserved_test.json`
