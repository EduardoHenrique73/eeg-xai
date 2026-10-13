# Alinhamento textual do TCC II

## Finalidade

Este documento orienta a atualizacao do projeto de TCC I para o texto final do TCC II, preservando a estrutura, o tom academico e a terminologia empregados no trabalho original. Os dados desta proposta devem ser incorporados ao arquivo-fonte editavel do TCC, quando ele estiver disponivel.

## Alteracoes estruturais necessarias

O TCC I descreve uma proposta de implementacao e emprega predominantemente verbos no futuro, como "sera utilizado", "serao realizados" e "sera implementado". O TCC II deve registrar o trabalho efetivamente executado. Portanto, ainda sao necessarias as seguintes alteracoes:

1. atualizar introducao, objetivos e metodologia para refletir o escopo efetivamente entregue;
2. converter a descricao metodologica do futuro para o passado;
3. substituir previsoes de implementacao pelas configuracoes realmente utilizadas;
4. acrescentar um capitulo de resultados;
5. acrescentar um capitulo de discussao;
6. acrescentar limitacoes e ameacas a validade;
7. acrescentar a conclusao e os trabalhos futuros;
8. substituir os wireframes pelas capturas da aplicacao implementada;
9. atualizar diagramas de arquitetura, fluxo e modelo de dados conforme o codigo final;
10. revisar referencias, numeracao de figuras, quadros, sumario e conformidade com o modelo institucional.

## Texto academico sobre as bases de dados

### Base de dados e delimitacao experimental

Para o desenvolvimento e a avaliacao experimental do modelo de deteccao temporal de crises epilepticas, foi utilizada a base publica CHB-MIT Scalp EEG Database, disponibilizada por meio da plataforma PhysioNet. A base contem registros continuos de eletroencefalograma de pacientes pediatricos, armazenados no formato European Data Format (EDF), acompanhados de anotacoes temporais que identificam o inicio e o termino dos eventos ictais. Essas caracteristicas permitem avaliar nao apenas a classificacao do exame, mas tambem a capacidade do modelo de localizar temporalmente as crises ao longo do sinal.

Os registros foram processados com separacao por paciente, com o objetivo de reduzir o vazamento de informacao entre os subconjuntos experimentais e aproximar a avaliacao do cenario de aplicacao em individuos nao observados durante o treinamento. Foram utilizados 77 arquivos EDF de 17 grupos de pacientes no treinamento. A calibracao foi realizada com 13 arquivos dos grupos `chb16`, `chb19`, `chb20` e `chb24`. Os registros dos grupos `chb09`, `chb15` e `chb18` foram empregados em uma avaliacao interpaciente historica de desenvolvimento. Como esses pacientes haviam sido consultados em experimentos anteriores, seus resultados nao foram caracterizados como teste externo completamente intocado.

A Bonn University EEG Database foi considerada durante o planejamento inicial do trabalho, conforme descrito no projeto de TCC I. Entretanto, ela nao foi incorporada ao protocolo experimental final. A base de Bonn e composta por segmentos curtos e previamente selecionados, com configuracao de canais e organizacao distintas dos registros continuos e multicanais do CHB-MIT. Consequentemente, sua combinacao direta com o CHB-MIT introduziria diferencas de dominio e nao permitiria a aplicacao uniforme das metricas temporais adotadas, como sobreposicao dos intervalos, erro de inicio e termino, sensibilidade por evento e falsos alarmes por hora.

Dessa forma, optou-se por delimitar o experimento final ao CHB-MIT, mantendo coerencia entre o objetivo de localizar eventos epilepticos em exames continuos, o formato dos dados e as metricas de avaliacao. A base de Bonn permanece mencionada como referencia considerada na etapa de planejamento, mas nao constitui fonte dos resultados apresentados neste trabalho.

## Texto academico para a metodologia executada

### Processamento dos sinais e representacao

Os arquivos EDF foram lidos e processados com a biblioteca MNE-Python. Cada registro foi dividido em janelas temporais de 4 segundos, com deslocamento de 2 segundos. Para cada canal, foram extraidas 15 caracteristicas que representam propriedades estatisticas, informacoes nos dominios do tempo e da frequencia, Entropia de Shannon e medidas derivadas de Dinamica Simbolica. A normalizacao robusta foi aplicada individualmente em cada arquivo EDF, com o objetivo de reduzir diferencas de escala entre registros sem utilizar informacoes dos pacientes destinados a calibracao ou avaliacao.

O janelamento temporal tambem desempenhou a funcao de ampliacao da quantidade de exemplos disponiveis para o treinamento. A amostragem foi realizada no nivel de sequencia e balanceada por evento ictal, evitando que crises longas dominassem o processo de otimizacao e preservando exemplos das regioes de inicio, centro e termino dos eventos.

### Arquitetura da rede neural

O modelo final corresponde a uma arquitetura hibrida CNN-BiLSTM com fusao espacial tardia. A entrada preserva a identidade de 23 canais, cada um representado por 15 caracteristicas. As operacoes convolucionais iniciais processam a dimensao temporal sem realizar agregacao espacial prematura. Posteriormente, as representacoes dos canais sao combinadas e encaminhadas a uma camada BiLSTM, responsavel por modelar as dependencias temporais bidirecionais. A saida consiste em uma probabilidade ictal para cada passo temporal, sem a aplicacao de pooling temporal global antes da predicao.

O treinamento utilizou Binary Cross-Entropy ponderada, regularizacao e selecao de checkpoint no conjunto de calibracao. A escolha considerou a sensibilidade por evento sob restricao de falsos alarmes por hora, em vez de depender exclusivamente de acuracia ou metricas calculadas por janela.

### Pos-processamento e explicabilidade

As probabilidades produzidas para sequencias sobrepostas foram agregadas na linha temporal do exame. O ponto operacional final utilizou limiar de 0,60, duracao minima de 18 segundos e fechamento de lacunas de ate 8 segundos. Esses parametros foram definidos exclusivamente no conjunto de calibracao.

Para apoiar a interpretacao das predicoes, foi aplicado o metodo Gradient SHAP sobre a sequencia de maior probabilidade. Os valores positivos de relevancia foram associados aos canais e intervalos temporais correspondentes e projetados sobre o tracado exibido na interface. Essa explicacao representa a contribuicao das caracteristicas janeladas para a sequencia analisada; portanto, nao deve ser interpretada como uma atribuicao independente para cada amostra bruta do EEG.

## Capitulos que devem ser acrescentados

### Resultados

O capitulo deve apresentar separadamente:

- caracterizacao da base e dos subconjuntos;
- resultados da calibracao da v43 seed 42;
- avaliacao interpaciente historica;
- resultados por EDF e por evento;
- metricas temporais e falsos alarmes por hora;
- curva FROC;
- comparacao das principais ablacoes;
- capturas do sistema implementado;
- exemplo do SHAP sobreposto ao sinal.

### Discussao

A discussao deve explicar que a calibracao alcancou F1 por EDF de 95,7%, F1 localizado de 90,9%, deteccao de 15 de 20 eventos e 0,55 falso alarme por hora. Tambem deve confrontar esses valores com a avaliacao interpaciente historica, na qual foram localizados 4 de 9 eventos, com F1 localizado de 46,2% e 2,08 falsos alarmes por hora. Essa queda demonstra que o principal gargalo e a generalizacao entre pacientes, e nao a integracao da aplicacao ou a reproducibilidade dos artefatos.

### Limitacoes

Devem ser declaradas explicitamente:

- inexistencia de uma coorte local totalmente intocada, pois todos os 24 grupos CHB-MIT disponiveis foram consultados em alguma fase;
- ausencia de validacao clinica com neurologistas;
- uso de uma unica base no experimento final;
- variabilidade interpaciente e dificuldade concentrada em alguns padroes do paciente `chb15`;
- SHAP aplicado a features janeladas, e nao a cada amostra bruta;
- ausencia de certificacao para uso diagnostico;
- dependencia do ambiente TensorFlow/Keras e necessidade de concluir a suite integral de testes.

### Conclusao

O texto final deve afirmar que o objetivo de implementar um prototipo web de apoio a analise de EEG foi atingido. A plataforma integra ingestao de EDF, processamento temporal, CNN-BiLSTM, Entropia de Shannon, Dinamica Simbolica, explicabilidade e suporte ao laudo. Entretanto, os resultados nao autorizam caracterizar o sistema como ferramenta diagnostica validada. A contribuicao comprovada e a viabilidade tecnica de uma plataforma de segunda opiniao e pesquisa, acompanhada de uma avaliacao transparente de suas limitacoes de generalizacao.

## Pendencias para considerar o TCC completo

### Obrigatorias

1. obter o arquivo editavel do TCC e incorporar as secoes atualizadas;
2. incluir resultados, discussao, limitacoes e conclusao;
3. atualizar figuras e diagramas com o sistema realmente implementado;
4. gerar e inserir a curva FROC final e as tabelas consolidadas;
5. inserir evidencias visuais de um EDF normal, um EDF com crise e o SHAP sobreposto;
6. revisar o protocolo experimental e declarar corretamente os subconjuntos;
7. executar a suite completa de testes em um ambiente TensorFlow estavel;
8. revisar citacoes e acrescentar a referencia especifica da CHB-MIT, alem da referencia geral da PhysioNet;
9. realizar revisao gramatical, normalizacao ABNT e atualizacao automatica do sumario;
10. preparar a apresentacao e uma demonstracao reproduzivel.

### Recomendadas, mas nao bloqueantes para um prototipo academico

- obter uma coorte externa completamente intocada;
- conduzir avaliacao de usabilidade com especialistas;
- medir desempenho e tempo de processamento em diferentes computadores;
- ampliar diversidade de pacientes e padroes ictais;
- avaliar calibracao probabilistica e robustez a montagens diferentes.

## Regra de comunicacao dos resultados

O texto e a apresentacao devem empregar as expressoes "apoio a analise", "triagem experimental", "trecho suspeito" e "segunda opiniao". Devem ser evitadas afirmacoes como "diagnostica epilepsia", "garante robustez", "pronto para uso clinico" ou "validado clinicamente". Os resultados de calibracao e da avaliacao interpaciente historica devem permanecer em tabelas distintas.
