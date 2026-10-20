# TCC e evidencias

## Mensagem central

O EEG-XAI e uma plataforma web de apoio a analise de EEG. O sistema integra ingestao EDF, processamento de sinais, rede neural temporal, visualizacao multicanal, explicabilidade e emissao de laudo. A IA fornece indicios para revisao medica; nao substitui o neurologista.

## O que mostrar

- problema e motivacao da triagem de EEG;
- fluxo completo do sistema;
- arquitetura CNN-BiLSTM e extracao das features;
- separacao por paciente antes do janelamento;
- diferenca entre calibracao e avaliacao historica;
- tabela de metricas por EDF, evento e localizacao;
- exemplos de trechos detectados e perdidos;
- limites: falsos alarmes, crises com score baixo e queda interpaciente;
- evolucao controlada das versoes, incluindo a decisao de nao promover a v57.

## O que evitar

- chamar score de probabilidade diagnostica;
- apresentar a v57 como melhoria global;
- misturar os 15/20 eventos da calibracao com os 4/9 da avaliacao historica;
- afirmar que SHAP explica cada amostra bruta se o mapa esta associado a janelas de features;
- usar resultados exploratorios como validacao clinica.

## Fontes principais

- `docs/tcc/overleaf_eeg_xai/main.tex`
- `docs/tcc/overleaf_eeg_xai/conteudo.tex`
- `docs/tcc/RELATORIO_FECHAMENTO_FINAL_2026-10-20.md`
- `docs/tcc/ROTEIRO_DEFESA_IDEIA_E_RESULTADOS.md`
- [[02 - Estado atual v43]]
- [[03 - Experimento v57]]
