# Investigacao arquitetural v31

## Base e limite estatistico

O manifesto local contem 93 EDFs, 87 eventos anotados e 4.687 segundos ictais.
O split fixo preserva 17 pacientes, 68 EDFs e 58 eventos para treino; quatro
pacientes, 13 EDFs e 20 eventos para calibracao; tres pacientes, 12 EDFs e nove
eventos para desenvolvimento reservado. Os tempos nao ictais aproximados sao
358.923 s no treino, 45.949 s na calibracao e 55.540 s no teste reservado.

A v29 treina com 4.053 sequencias selecionadas. Sua tabela de alvos temporais
contem 1.684 passos ictais e 8.534 normais apos consolidar sobreposicoes;
esses passos nao equivalem a eventos independentes. A base e pequena para uma
rede de alta capacidade ou um transformer treinado do zero.

## Auditoria da representacao atual

`time_frequency_per_channel` produz 15 features por canal para 23 canais de
referencia. O vetor de 345 features conserva a ordem dos canais, mas a primeira
`Conv1D` da CNN-BiLSTM recebe todos os 345 valores de uma vez. Ela aprende
misturas entre canais antes de qualquer projecao por canal. A BiLSTM usa oito
passos e preve oito probabilidades, sem pooling temporal global. Assim, a
identidade espacial existe na entrada, mas nao e preservada como eixo explicito
pela arquitetura.

O experimento v26 demonstrou que aumentar diretamente a sequencia para 16
passos diminui a quantidade de exemplos positivos curtos. A v31 expande apenas
o *input* ao redor de um alvo central de oito passos; o numero de alvos e a
frequencia de amostragem permanecem iguais. As bordas do EDF sao preenchidas
com o primeiro/ultimo passo observado.

## Comparacao arquitetural

| Opcao | Dados/overfit | Temporal | Espacial | Custo | Integracao |
|---|---|---|---|---|---|
| CNN + TCN | baixo/moderado | dilatacao | mistura cedo | baixo | direta |
| CNN + TCN + atencao | moderado | dilatacao | depende da atencao | medio | direta |
| Multiscale CNN + TCN | moderado | curto/medio/longo | preservavel | medio | direta |
| Frontend EEGNet-like + TCN | moderado | bom | forte no sinal bruto | medio | requer nova entrada bruta |
| CNN + BiGRU | moderado | recorrente curto | mistura cedo | medio | direta |
| Transformer temporal leve | alto | contexto amplo | depende da projecao | alto | direta, mas dados escassos |
| U-Net temporal | moderado/alto | multiresolucao | depende da entrada | alto | altera alinhamento temporal |
| Conv-TasNet-like | alto | multiescala fina | orientado ao sinal bruto | alto | requer nova entrada |
| TCN residual dilatada | baixo/moderado | receptivo ampliado | necessita frontend | baixo | direta |
| CNN-TCN-BiLSTM | alto | ampla | depende do frontend | alto | aumenta capacidade sem evidencia |

Escolha inicial: projecao compartilhada por canal, tres ramos temporais com
dilatacao 1/2/4, gate por canal e tres blocos TCN residuais compactos. A saida
e uma probabilidade por passo do alvo central de oito passos. Nao ha GNN nem
transformer: as features disponiveis ja sao resumos por janela, e 58 eventos
de treino nao sustentam sua capacidade adicional.

## Referencias tecnicas

- Bai, Kolter e Koltun, *An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling* (2018):
  https://arxiv.org/abs/1803.01271
- Lawhern et al., *EEGNet: A Compact Convolutional Network for EEG-based
  Brain-Computer Interfaces* (2018): https://arxiv.org/abs/1611.08024
- Hu, Shen e Sun, *Squeeze-and-Excitation Networks* (2018):
  https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html
- Perslev et al., *U-Time: A Fully Convolutional Network for Time Series
  Segmentation Applied to Sleep Staging* (2019):
  https://proceedings.neurips.cc/paper_files/paper/2019/hash/57bafb2c2dfeefba931bb03a835b1fa9-Abstract.html

## Protocolo

Ablações A-F usam somente os 17 pacientes de treino e quatro de calibracao.
O criterio de escolha e: `FA/h <= 1`, maior recall de eventos, maior F1
localizado, maior precisao de eventos e menor FA/h. Todas as configuracoes
usam a mesma grade de threshold, duracao e gap da baseline; histerese fica em
1,0. O Random Forest permanece desligado. Somente a arquitetura escolhida sera
medida nos tres pacientes reservados. Esses pacientes ja foram vistos em
experimentos anteriores e, portanto, continuam sendo desenvolvimento, nao um
teste clinico final independente.

As perdas comparadas sao weighted BCE, BCE+erro de borda, BCE+Tversky temporal
e BCE+penalidade de oscilacao em passos com mesmo rotulo. As tres combinacoes
usam coeficiente pequeno (0,1) predefinido; nenhum coeficiente sera ajustado
no teste reservado. O sampler por evento seleciona indices distintos para
inicio, meio e fim de cada crise e respeita o teto de 32 positivos por EDF.

## Limitacoes da analise de erros

O artefato v29 conserva score maximo e medio por evento, mas nao guarda a
serie temporal completa nem atribuicoes por canal. E possivel comparar seus
intervalos e scores agregados, mas nao afirmar qual canal contribuiu mais para
a predicao v29. A analise de energia/topografia por canal pode ser extraida
dos EDFs como desvio de features entre periodo ictal e contexto normal; isso
nao equivale a explicacao da rede. As categorias clinicas de hard negatives
tambem requerem anotacao propria: amplitude, ritmo e transiente derivados de
heuristicas nao devem ser tratados como rotulos clinicos validados.
