# v36: branch curta de sinal bruto

## Objetivo

Testar se uma representacao curta da forma de onda bruta por canal recuperaria
eventos que recebem score ictal baixo nas features tempo-frequencia. O
experimento manteve a segmentacao temporal e adicionou uma segunda entrada com
os dois segundos centrais de cada janela, reamostrados a 64 Hz.

A branch usa convolucoes separaveis por canal, sem misturar a identidade dos
canais no inicio. Depois do pooling apenas na dimensao de amostras, sua saida e
projetada como residuo sobre a branch espectral. A projecao residual foi
inicializada em zero para que o treino comecasse equivalente a representacao
espectral. O modelo possui 137.178 parametros.

Foram mantidos o split interpaciente, a BCE ponderada, o sampler por sequencia,
o pos-processamento e a busca FROC usados na comparacao controlada. O Random
Forest permaneceu desligado. `chb09`, `chb15` e `chb18` nao foram carregados.

## Resultado na calibracao

Resultados com o limite principal de ate 1,0 falso alarme por hora:

| Seed | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h |
|---:|---:|---:|---:|---:|---:|
| 42 | 4/20 | 24,2% | 50,0% | 80,0% | 0,71 |
| 43 | 10/20 | 48,8% | 73,7% | 90,9% | 0,86 |
| 44 | 10/20 | 47,6% | 66,7% | 90,9% | 0,94 |
| **Media v36** | **8,0/20** | **40,2%** | **63,5%** | **87,3%** | **0,84** |
| Baseline v32 | 12,67/20 | - | 70,2% | 87,9% | 0,89 |

A v36 reduziu discretamente os falsos alarmes, mas detectou em media 4,67
eventos a menos que a baseline e piorou o F1 localizado. A seed 42 tambem
evidencia instabilidade relevante.

Na FROC, afrouxar o limite de falsos alarmes nao resolveu de forma robusta:

- seed 42: 12/20 eventos somente com 1,49 FA/h;
- seed 43: 12/20 eventos com 1,02 FA/h;
- seed 44: 12/20 eventos somente com 1,49 FA/h.

Portanto, o sinal bruto curto contem alguma evidencia adicional, mas esta
fusao nao consegue explora-la mantendo o limite de falsos alarmes.

## Decisao

**REJEITAR v36.** A variante nao sera promovida, integrada a inferencia ou
avaliada nos pacientes reservados. O modelo ativo permanece inalterado.

O resultado indica que anexar uma branch bruta residual a uma representacao ja
agregada nao basta. O proximo experimento deve investigar a representacao do
primeiro estagio sem consultar o teste reservado, priorizando uma destas
hipoteses controladas:

1. pretreino auto-supervisionado por canal no conjunto de treino;
2. sampler por evento com cobertura explicita de inicio, centro, fim e crises
   curtas;
3. frontend espacial que combine canais apenas depois de extrair padroes
   temporais locais.

Antes de aumentar novamente a arquitetura, deve-se gerar uma analise por evento
na calibracao para identificar quais crises ficam sem score e se o problema e
topografia, duracao ou baixa relacao sinal-ruido.

Artefato: `modelos/v36_raw_waveform_calibration.json`.
