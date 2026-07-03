# Relatorio do experimento intra-paciente CNN-LSTM
## Objetivo
Avaliar se o pipeline CNN-LSTM sequencial consegue identificar trechos ictais em exames EDF do mesmo paciente, usando exames anteriores do proprio paciente para treino e um EDF separado para teste.
## Escopo do experimento
- Base: arquivos EDF locais do CHB-MIT em `dataset_amostra`.
- Modelo: CNN-LSTM sequencial sobre janelas consecutivas de features EEG.
- Janela: 4 segundos; passo: 2 segundos; sequencia: 8 janelas.
- Criterio de decisao: score minimo por janela + duracao minima de trecho suspeito continuo.
- Validacao: intra-paciente, com um EDF com crise e um EDF normal separados para teste por paciente.
## Resultado resumido
| Paciente | Threshold | Duracao minima | Teste com crise | Crise real (s) | Trecho detectado (s) | Sobreposicao (s) | Teste normal | FP | FN | Localizou crise real |
|---|---:|---:|---|---:|---:|---:|---|---:|---:|---|
| chb01 | 0.5 | 45s | chb01_15.edf | 1716-1786 | 1724-1786 | 62 | chb01_01.edf | 0 | 0 | sim |
| chb03 | 0.5 | 10s | chb03_02.edf | 716-810 | 736-806 | 70 | chb03_05.edf | 0 | 0 | sim |
| chb13 | 0.5 | 60s | chb13_62.edf | 836-2738 | 2660-2826 | 78 | chb13_02.edf | 0 | 0 | sim |
| chb14 | 0.85 | 45s | chb14_18.edf | 1024-1078 | 948-1078 | 54 | chb14_01.edf | 0 | 0 | sim |
| chb15 | 0.5 | 90s | chb15_31.edf | 1736-1886 | 1712-1926 | 150 | chb15_01.edf | 0 | 0 | sim |

## Leitura dos resultados
Nos cinco pacientes avaliados, a calibracao intra-paciente localizou a crise real no EDF de teste e nao marcou o EDF normal como crise. O caso `chb14`, que anteriormente nao localizava a crise real, passou a localizar depois que o treino foi corrigido para usar todos os EDFs de crise disponiveis do paciente.
## Estado atual do projeto
O experimento intra-paciente esta consolidado como evidencia para o TCC. Isso nao significa que o sistema esteja clinicamente finalizado ou que exista um modelo universal pronto para qualquer paciente. A validacao entre pacientes ainda apresentou instabilidade e deve ser tratada como limitacao ou trabalho futuro.
## Limitacoes
- A amostra ainda e pequena: cinco pacientes com splits controlados.
- A calibracao foi feita por paciente; isso e adequado para analise intra-paciente, mas nao prova generalizacao universal.
- O modelo usa sequencias de features por janela, nao o sinal EEG bruto multicanal completo.
- O resultado deve ser interpretado como apoio a triagem/analise, sempre com validacao medica.
## Conclusao
A etapa experimental principal do TCC esta em bom estado para ser documentada: o pipeline carrega EDFs, rotula janelas com base nos summaries do CHB-MIT, treina CNN-LSTM sequencial, calibra threshold/duracao e retorna trechos suspeitos auditaveis. O proximo trabalho de engenharia e decidir se a aplicacao vai operar como modelo intra-paciente ou se sera desenvolvido um modelo global com mais dados e validacao entre pacientes.
