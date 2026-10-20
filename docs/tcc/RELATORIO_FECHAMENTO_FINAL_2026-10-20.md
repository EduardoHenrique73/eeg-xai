# Fechamento final do EEG-XAI e do TCC II

Data: 20/10/2026.

## Estado consolidado

- Modelo final: v43, seed 42, arquitetura CNN-BiLSTM com fusao tardia de 23 canais.
- Entrada: sequencias de oito janelas, com 15 caracteristicas tempo-frequencia por canal.
- Ponto operacional: threshold 0,60, duracao minima 18 s e fechamento de lacunas de 8 s.
- Treino: 77 EDFs de 17 grupos.
- Calibracao: 13 EDFs de chb16, chb19, chb20 e chb24.
- Avaliacao historica: chb09, chb15 e chb18, sem ajuste de parametros nessa avaliacao.

## Reproducao executada

A avaliacao historica congelada foi reexecutada em 20/10/2026. Os hashes do modelo, scaler, metadados e calibracao coincidiram com o manifesto. Os resultados foram reproduzidos:

| Metrica | Resultado |
|---|---:|
| Precisao por EDF | 87,5% |
| Sensibilidade por EDF | 77,8% |
| F1 por EDF | 82,4% |
| F1 localizado | 46,2% |
| Eventos detectados | 4/9 |
| Precisao por evento | 11,1% |
| Recall por evento | 44,4% |
| F1 por evento | 17,8% |
| Falsos alarmes por hora | 2,08 |
| IoU media dos eventos detectados | 73,7% |
| Erro absoluto medio de inicio | 3,0 s |
| Erro absoluto medio de termino | 23,5 s |

Os resultados confirmam reproducibilidade computacional no ambiente atual. O conjunto permanece uma avaliacao historica de desenvolvimento, e nao um teste externo intocado.

## Evidencias atualizadas

- Graficos de metricas, eventos/FA-h e matrizes de confusao foram regenerados a partir dos artefatos atuais.
- As imagens regeneradas coincidem com as incorporadas ao projeto Overleaf.
- O capitulo de resultados registra a configuracao reproduzivel, a distincao entre calibracao e historico e as limitacoes do protocolo.
- O roteiro de defesa foi ampliado para doze slides e inclui as respostas conceituais mais importantes.

## Mensagem correta para a banca

O trabalho entrega um prototipo funcional integrado e uma avaliacao experimental transparente. A v43 apresentou desempenho forte na calibracao, mas perdeu generalizacao na avaliacao historica. Portanto, a contribuicao e valida como Engenharia de Software e IA explicavel aplicada a EEG, sem alegar validacao clinica ou diagnostico autonomo.

## Pendencias antes da entrega

1. Fazer a conferencia visual final do PDF compilado: quebras de pagina, legendas, sumario e legibilidade das figuras.
2. Conferir referencias bibliograficas e URLs que ainda dependem de verificacao manual da fonte original.
3. Montar os slides usando o roteiro final e as imagens ja existentes em `docs/tcc/assets`.
4. Ensaiar a defesa mantendo os mesmos numeros da tabela de resultados.

Nao se recomenda iniciar novo treinamento antes da entrega, pois isso abriria outra etapa experimental sem conjunto externo intocado e poderia tornar o texto e a apresentacao inconsistentes.
