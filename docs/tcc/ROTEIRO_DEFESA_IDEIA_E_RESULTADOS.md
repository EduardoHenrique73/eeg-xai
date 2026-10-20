# Roteiro final para defesa do EEG-XAI

## Tese central em uma frase

O EEG-XAI demonstra que e possivel integrar visualizacao multicanal, localizacao temporal por CNN-BiLSTM, explicabilidade por Gradient SHAP e emissao de laudo em um unico prototipo, mas os resultados historicos mostram que a generalizacao entre pacientes ainda limita o uso clinico.

## Mensagem de abertura

O EEG-XAI integra a importacao do EEG, a visualizacao dos canais, a indicacao automatica de trechos suspeitos, uma explicacao computacional e o registro do parecer em uma plataforma web. O detector sugere onde revisar; o especialista interpreta o exame. O trabalho entrega o software integrado e mede os acertos e as limitacoes do detector.

## Ordem sugerida de apresentacao

1. Problema: registros extensos exigem navegacao entre muitos canais e intervalos.
2. Objetivo: reunir visualizacao, localizacao temporal e explicabilidade no mesmo fluxo.
3. Demonstracao visual: uma tela final, destacando entrada, intervalo suspeito e parecer.
4. Metodo: EDF -> janelas -> caracteristicas por canal -> CNN-BiLSTM -> escores -> intervalos.
5. Avaliacao: separar treino, calibracao e avaliacao historica; explicar as unidades EDF e evento.
6. Resultados: apresentar ambos os conjuntos com o mesmo ponto operacional.
7. Exemplos: um acerto temporal e uma falha ja presentes na avaliacao.
8. Conclusao: contribuicao de software, resultado experimental e limite de generalizacao.

## Sequencia sugerida de slides

### Slide 1 - Problema e objetivo

Mostrar um EEG com muitos canais e explicar: "O desafio nao e apenas dizer se um arquivo contem crise; e indicar em que intervalo o especialista deve revisar." Em seguida, apresentar o objetivo de integrar o fluxo em uma plataforma web.

### Slide 2 - O que foi entregue

Usar uma captura da interface final. Destacar apenas quatro pontos: upload EDF, visualizacao multicanal, trechos suspeitos com SHAP e laudo. Evitar listar todas as tecnologias neste momento.

### Slide 3 - Fluxo da IA

Apresentar: EDF -> janelas de 4 s -> 15 caracteristicas por canal -> sequencias de 8 passos -> CNN-BiLSTM -> escores temporais -> trechos suspeitos. Explicar que a saida e um escore operacional, nao uma certeza diagnostica.

### Slide 4 - Evolucao em relacao ao TCC I

Explicar que Entropia de Shannon, Dinamica Simbolica, Bonn e k-fold foram investigados no planejamento ou em ensaios exploratorios. A versao final adotou CHB-MIT, caracteristicas tempo-frequencia, separacao por grupos antes do janelamento e avaliacao temporal por evento.

### Slide 5 - Protocolo final

Mostrar treino com 77 EDFs de 17 grupos, calibracao com 13 EDFs de quatro grupos e avaliacao historica em chb09, chb15 e chb18. Informar seed 42, limiar 0,60, duracao minima 18 s e gap 8 s. Dizer explicitamente que os pacientes historicos ficaram fora do treino e da calibracao da v43, mas ja haviam sido consultados durante o desenvolvimento.

### Slide 6 - Como ler as metricas

Usar tres perguntas: "O EDF recebeu alerta?", "A crise foi encontrada?" e "O trecho principal ficou no lugar certo?" Associar respectivamente F1 por EDF, recall/F1 por evento e F1 localizado. FA/h mede alarmes sem correspondencia por hora nao ictal.

### Slide 7 - Resultado na calibracao

Apresentar F1 EDF 95,7%, F1 localizado 90,9%, 15/20 eventos e 0,55 FA/h. Ressaltar que calibracao participa da escolha do ponto operacional e nao equivale a desempenho externo.

### Slide 8 - Avaliacao historica

Apresentar F1 EDF 82,4%, F1 localizado 46,2%, 4/9 eventos e 2,08 FA/h. A mensagem principal e a queda de generalizacao interpaciente, nao esconder o resultado menor.

### Slide 9 - Um acerto e uma falha

No chb09_19, mostrar o evento real 5.298--5.362 s e a previsao 5.300--5.354 s, com IoU de 84,4%. No chb15_10, explicar que o evento 1.080--1.114 s nao teve candidato valido. Isso torna concreta a diferenca entre classificar e localizar.

### Slide 10 - Explicabilidade

Mostrar o overlay SHAP. Explicar que ele destaca contribuicoes positivas das caracteristicas por canal e janela para a sequencia de pico. Nao explica cada amostra bruta e nao prova causalidade fisiologica.

### Slide 11 - Contribuicoes e limites

Contribuicoes: plataforma integrada, detector temporal, avaliacao por evento e overlay explicavel. Limites: cinco de nove eventos historicos perdidos, 2,08 FA/h, ausencia de validacao externa e ausencia de estudo com neurologistas.

### Slide 12 - Conclusao

Encerrar com: "O sistema cumpre a proposta como prototipo funcional e experimento de Engenharia de Software com IA explicavel. O principal proximo passo cientifico e validar em uma coorte externa mantida intocada."

## Numeros que devem permanecer identicos

| Contexto | F1 EDF | F1 localizado | Eventos | FA/h |
|---|---:|---:|---:|---:|
| Calibracao | 95,7% | 90,9% | 15/20 | 0,55 |
| Avaliacao historica | 82,4% | 46,2% | 4/9 | 2,08 |

A IoU temporal media de 73,7%, o erro absoluto medio de inicio de 3,0 s e o erro absoluto medio de termino de 23,5 s descrevem somente os quatro eventos historicos detectados.

## Perguntas provaveis

### Qual e a contribuicao?

A integracao de ingestao, navegacao multicanal, detector temporal e explicacao no mesmo ambiente, acompanhada de avaliacao quantitativa. Nao foi demonstrada melhoria da produtividade de medicos, pois nao houve estudo com usuarios.

### O modelo usa o sinal bruto?

A entrada final usa 15 caracteristicas tempo-frequencia por canal, organizadas em sequencias de oito janelas. A CNN aprende padroes nessa representacao, e a BiLSTM considera sua evolucao temporal.

### Por que mudou em relacao ao TCC I?

As decisoes evoluiram durante os experimentos. Entropia de Shannon e Dinamica Simbolica foram investigadas; a representacao final utiliza caracteristicas espectrais. O k-fold inicial foi exploratorio. A avaliacao final separou grupos antes do janelamento. Bonn nao foi misturada a avaliacao temporal de EDFs continuos.

### Por que nao apresentar so o F1 de 95,7%?

Esse valor e por EDF e foi obtido na calibracao, usada na selecao do modelo e dos parametros. Na avaliacao historica, o F1 EDF foi 82,4%, com quatro de nove eventos detectados e 2,08 falsos alarmes por hora. Ambos precisam aparecer para descrever o comportamento observado.

### Como acertar um arquivo e errar a localizacao?

Um arquivo pode receber alerta em um trecho normal e ainda ser classificado como positivo. No historico, sete EDFs ictais receberam alerta, quatro eventos foram encontrados e tres candidatos principais corresponderam a crises. Cada metrica responde a uma pergunta diferente.

### Um escore de 99% significa 99% de certeza?

Nao. E uma saida do modelo utilizada para ordenar e filtrar suspeitas. Nao foi demonstrada equivalencia entre esse escore e uma probabilidade clinica calibrada. O limiar 0,60 e uma regra de decisao escolhida na calibracao.

### O que o SHAP explica?

Contribuicoes das caracteristicas, por canal e janela, para um passo temporal de uma sequencia que contribui para o pico. Nao explica cada amostra bruta, nao demonstra causa fisiologica e nao decompoe exatamente a media de todas as sequencias sobrepostas.

### E um teste independente?

Os grupos historicos ficaram fora do treino e da calibracao da versao final, mas ja haviam sido consultados durante o desenvolvimento. Portanto, a denominacao correta e avaliacao historica interpaciente.

### O sistema esta pronto para uso clinico?

O fluxo de software foi implementado e verificado tecnicamente. Os resultados ainda mostram crises perdidas e falsos alarmes, e nao houve validacao clinica com especialistas. Essa delimitacao faz parte da conclusao do trabalho.

## Conferencia antes da defesa

- Usar os valores da v43 seed 42 e identificar quando um numero for media entre seeds.
- Explicar que FA/h usa tempo nao ictal e que F1 localizado e uma metrica por EDF com criterio temporal.
- Distinguir screenshots, diagramas conceituais e graficos derivados de resultados.
- Nao interpretar IoU media dos eventos detectados como resultado de todas as crises.
- Apresentar os exemplos como ilustracoes do conjunto ja avaliado.
- Encerrar com o que foi entregue e o que a avaliacao demonstrou.
