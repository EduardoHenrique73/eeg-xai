# Fechamento tecnico e cientifico do TCC

## Escopo entregue

O sistema implementa uma plataforma web de apoio a analise de EEG em EDF. O backend FastAPI gerencia autenticacao, pacientes, exames, processamento assincrono, laudos e configuracoes medicas. O frontend React oferece visualizacao multicanal, selecao de canais, trechos suspeitos, score, parametros da decisao e sobreposicao SHAP no sinal.

O sistema deve ser apresentado como apoio experimental a revisao medica. Ele nao realiza diagnostico autonomo e nao possui validacao clinica.

## Metodologia final

### Dados

A base principal e CHB-MIT. O protocolo usa separacao por paciente:

- treino: 77 EDFs de 17 grupos de paciente;
- calibracao: 13 EDFs de `chb16`, `chb19`, `chb20` e `chb24`;
- desenvolvimento historico: 12 EDFs de `chb09`, `chb15` e `chb18`.

Todos os 24 pacientes CHB-MIT locais ja foram consultados em alguma fase. Logo, o projeto nao possui atualmente uma coorte local interpaciente totalmente intocada.

O Bonn foi citado na proposta inicial, mas nao foi combinado ao experimento temporal principal. Seus segmentos, canais e protocolo diferem do CHB-MIT. Misturar diretamente as bases comprometeria a interpretacao de localizacao por evento. No texto final, Bonn deve ser descrito como base estudada/planejada, nao como fonte dos resultados finais, salvo se um experimento separado for acrescentado.

### Representacao e modelo

Cada EDF e dividido em janelas de 4 s com passo de 2 s. Sao calculadas features tempo-frequencia por canal, estatisticas, entropia de Shannon e medidas de dinamica simbolica. A normalizacao robusta e feita por EDF antes da amostragem.

A v43 usa 23 canais e 15 features por canal. Convolucoes temporais sao aplicadas mantendo a identidade dos canais; a fusao espacial ocorre posteriormente. Uma BiLSTM produz probabilidade ictal para cada passo da sequencia. Nao existe pooling temporal global antes da saida.

O treino usa weighted BCE, amostragem por sequencia e balanceamento ictal por evento. O checkpoint e escolhido na calibracao por recall de evento, sujeito a limite de falsos alarmes por hora.

### Decisao temporal

O modelo final usa:

- threshold: 0,60;
- duracao minima: 18 s;
- fechamento de lacunas: 8 s;
- limite de cobertura suspeita: 70% do exame.

Esses parametros foram escolhidos somente na calibracao. A aplicacao agrega as saidas de sequencias sobrepostas e retorna probabilidade por janela temporal, trecho principal, candidatos secundarios e janela de pico.

### Explicabilidade

Gradient SHAP explica as features da sequencia de pico. A interface projeta os valores positivos por canal e intervalo de segundos sobre o tracado. A explicacao nao equivale a SHAP por amostra bruta e deve ser descrita com essa limitacao.

## Resultados

### Calibracao da v43 seed 42

| Metrica | Resultado |
|---|---:|
| F1 por EDF | 95,7% |
| F1 localizado | 90,9% |
| Eventos detectados | 15/20 |
| Precisao por evento | 68,2% |
| Sensibilidade por evento | 75,0% |
| F1 por evento | 71,4% |
| Falsos alarmes por hora | 0,55 |

### Desenvolvimento historico interpaciente

| Metrica | Resultado |
|---|---:|
| Precisao por EDF | 87,5% |
| Sensibilidade por EDF | 77,8% |
| F1 por EDF | 82,4% |
| F1 localizado | 46,2% |
| Eventos detectados | 4/9 |
| F1 por evento | 17,8% |
| Falsos alarmes por hora | 2,08 |

As crises `chb09_06`, `chb09_19`, `chb15_31` e `chb18_29` foram encontradas. O paciente `chb15` concentrou crises perdidas e falsos alarmes. O resultado mostra queda de generalizacao entre pacientes, apesar do desempenho forte na calibracao.

### Ablacoes finais

Hard-negative mining foi testado sem usar pacientes reservados. A v55 forte caiu para 12,67/20 eventos e F1 localizado de 71,3%. A v56 leve igualou 15/20 eventos da v43 seed 42, mas reduziu F1 localizado para 85,7% sem alterar FA/h. Ambas foram rejeitadas.

## Resposta aos objetivos do TCC

| Objetivo | Situacao |
|---|---|
| Plataforma web para EEG | Implementado |
| Upload e leitura EDF | Implementado |
| CNN-LSTM temporal | Implementado como CNN-BiLSTM |
| Entropia e dinamica simbolica | Implementado nas features |
| Janelamento/data augmentation | Implementado |
| SHAP por canais e segundos | Implementado para a sequencia de pico |
| Configuracoes medicas | Implementado |
| Avaliacao interpaciente | Implementada como desenvolvimento historico |
| Teste externo intocado | Pendente |
| Validacao clinica | Nao realizada |

## Conclusao sugerida

O trabalho demonstra a viabilidade tecnica de integrar processamento de EEG, segmentacao temporal por CNN-BiLSTM e explicabilidade em uma aplicacao web. A avaliacao controlada mostrou que o modelo consegue localizar parte dos eventos e produzir explicacoes temporais e espaciais. Entretanto, a queda no conjunto interpaciente historico evidencia que desempenho de calibracao nao implica generalizacao clinica. O sistema deve ser considerado prototipo de apoio a pesquisa e revisao medica. Trabalhos futuros devem incluir coorte externa intocada, maior diversidade de crises, confirmacao temporal independente e estudo com especialistas.

## Checklist de apresentacao

1. Demonstrar login, paciente, upload EDF e visualizacao.
2. Mostrar um EDF normal e um EDF com crise conhecida.
3. Exibir trecho suspeito, parametros da decisao e SHAP sobre o tracado.
4. Apresentar calibracao e desenvolvimento historico em tabelas separadas.
5. Explicar por que accuracy de janela nao e suficiente.
6. Declarar explicitamente as limitacoes de generalizacao e validacao clinica.
7. Verificar os artefatos com `scripts/verify_release_artifacts.py`.

## Estado da verificacao tecnica

- build de producao do frontend: aprovado (`tsc` e Vite, 2.580 modulos);
- testes focados de inferencia temporal/checkpoint: 22 aprovados;
- carregamento seguro do `.keras`: aprovado;
- smoke test em EDF real completo: aprovado;
- hashes SHA-256 dos quatro artefatos: aprovados;
- `git diff --check`: aprovado;
- suite pytest completa: nao concluida nesta sessao por bloqueio do runtime local durante a primeira inicializacao do TensorFlow/Keras; nenhuma falha de assercao foi observada antes da interrupcao.

Antes da apresentacao, reiniciar o ambiente e repetir `.venv\Scripts\python.exe -m pytest -q`. O resultado so deve ser registrado como aprovado se o processo terminar com codigo zero.
