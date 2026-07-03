# Relatorio do experimento inter-paciente CNN-LSTM
## Objetivo
Avaliar o objetivo principal do projeto: treinar um modelo global com varios pacientes e testar em pacientes nunca vistos.
## Metodo
- Validacao leave-one-patient-out: cada fold remove um paciente inteiro do treino.
- A calibracao de threshold e duracao minima foi feita somente nos pacientes de treino.
- O paciente de teste nao foi usado para escolher threshold/duracao.
- Modelo: CNN-LSTM sequencial com janelas de 4s, passo de 2s e sequencias de 8 janelas.
## Resultado resumido
| Paciente teste | Thr | Duracao | EDFs crise | EDFs normais | Crises localizadas | FP | FN | EDF F1 | Window F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chb01 | 0.95 | 10s | 3 | 1 | 2 | 1 | 0 | 0.857 | 0.160 |
| chb03 | 0.95 | 180s | 2 | 1 | 0 | 0 | 2 | 0.000 | 0.237 |
| chb11 | 0.8 | 10s | 1 | 1 | 0 | 1 | 0 | 0.667 | 0.000 |
| chb13 | 0.7 | 10s | 8 | 3 | 2 | 3 | 0 | 0.842 | 0.035 |
| chb14 | 0.9 | 10s | 6 | 1 | 1 | 1 | 0 | 0.923 | 0.007 |
| chb15 | 0.95 | 10s | 5 | 1 | 0 | 1 | 5 | 0.000 | 0.000 |

## Totais
- EDFs com crise avaliados: 25
- EDFs normais avaliados: 8
- Crises localizadas com sobreposicao real: 5
- Falsos positivos: 7
- Falsos negativos: 7
- F1 medio por EDF: 0.548
- F1 medio por janela: 0.073

## Interpretacao
O modelo global ainda nao esta pronto para substituir um classificador clinico confiavel. Ele detecta alguns EDFs com crise, mas localiza poucas crises reais e ainda gera falsos positivos. Isso confirma que a maior dificuldade do projeto e a generalizacao para pacientes nunca vistos.
## Decisao tecnica
O proximo trabalho deve focar em melhorar a generalizacao inter-paciente, antes de promover esse modelo como principal na aplicacao. Caminhos provaveis: aumentar a base com mais EDFs normais e ictais por paciente, usar features por canal ou sinal bruto multicanal, calibrar em conjunto de validacao separado e reportar sensibilidade/falso positivo por EDF.
