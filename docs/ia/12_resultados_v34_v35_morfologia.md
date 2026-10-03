# v34-v35: morfologia temporal por canal

## Objetivo

Testar se descritores de forma de onda invariantes a amplitude recuperariam
crises de baixa potencia sem reproduzir os falsos alarmes dos deltas da v33.
Foram adicionados por canal: comprimento de linha normalizado, mobilidade e
complexidade de Hjorth, cruzamentos por zero, fator de crista, skewness,
kurtosis e entropia simbolica.

O espectro existente foi reutilizado; a morfologia foi calculada no EDF inteiro
e normalizada por mediana/IQR antes da amostragem. `chb09`, `chb15` e `chb18`
nao foram carregados. O Random Forest permaneceu desligado.

## Ablacao

- v34: concatenacao precoce das 15 features espectrais e oito morfologicas;
- v35: branch morfologica residual separada, inicializada em zero, somada a
  branch espectral antes da segunda convolucao da CNN-BiLSTM.

Media de seeds 42, 43 e 44 na calibracao fixa:

| Variante | Eventos | F1 localizado | F1 EDF | FA/h |
|---|---:|---:|---:|---:|
| Baseline | 12,67/20 | 70,2% | 87,9% | 0,89 |
| v34 fusao precoce | 9,00/20 | 69,0% | 90,9% | 0,73 |
| v35 branch residual | 10,00/20 | 71,3% | 89,2% | 0,76 |

## Decisao

**REJEITAR v34 e v35.** As duas variantes reduziram falsos alarmes, mas a
queda de sensibilidade por evento e inaceitavel. Nenhuma foi avaliada no teste
reservado ou integrada ao fluxo de producao.

Os resultados mostram que estatisticas morfologicas agregadas ajudam a
rejeitar atividade normal, mas descartam detalhes necessários para representar
algumas crises. A proxima mudanca justificavel e uma branch curta de forma de
onda bruta, com convolucao depthwise por canal e forte regularizacao, fundida
com a branch espectral. Ela deve ser validada primeiro por grupos de pacientes;
nao ha justificativa para novo ajuste de threshold ou para promover as features
manuais.

Artefatos: `modelos/v34_morphology_calibration.json` e
`modelos/v35_morphology_residual_calibration.json`.
