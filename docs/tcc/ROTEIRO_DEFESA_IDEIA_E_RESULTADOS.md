# Roteiro para explicar a ideia e os resultados

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
