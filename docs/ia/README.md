# Documentacao da IA do EEG-XAI

Atualizado em: 2026-10-01

Esta pasta documenta o estado atual da IA do projeto, o que foi feito, quais experimentos foram executados, quais problemas ainda existem e quais sao os proximos passos recomendados.

Arquivos principais:

- [01_historico_do_trabalho.md](01_historico_do_trabalho.md): linha do tempo do que foi feito no sistema e na IA.
- [02_estado_atual_da_ia.md](02_estado_atual_da_ia.md): como a IA funciona hoje dentro do backend/frontend.
- [03_experimentos_e_metricas.md](03_experimentos_e_metricas.md): modelos testados, metricas e conclusoes.
- [04_problemas_conhecidos.md](04_problemas_conhecidos.md): gargalos atuais, falsos positivos e erros de localizacao.
- [05_proximos_passos.md](05_proximos_passos.md): plano tecnico para melhorar a rede neural.
- [06_comandos_e_arquivos_importantes.md](06_comandos_e_arquivos_importantes.md): comandos usados, artefatos e variaveis de ambiente.

Resumo executivo:

O sistema ja possui integracao completa para o modelo sequencial CNN-LSTM, com fallback para o fluxo legado. A API e a interface ja exibem score, trecho suspeito, top trechos suspeitos e metadados do modelo.

O gargalo atual nao e mais integracao. O gargalo e qualidade do modelo: ele ainda tem falsos positivos e principalmente dificuldade para localizar corretamente o trecho real da crise em pacientes novos.

O melhor pipeline experimental atual e a `v11` hibrida com normalizacao robusta,
evidencia acumulada e quatro pacientes de calibracao. Ela obteve F1 por EDF
`0.8222` e F1 localizado `0.6111`, mas localizou somente 4 de 9 crises no
conjunto de desenvolvimento. Nenhum novo modelo foi promovido; ainda falta o
teste final em pacientes intocados.
