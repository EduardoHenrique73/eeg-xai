# Documentacao da IA do EEG-XAI

Atualizado em: 2026-10-03

Esta pasta documenta o estado atual da IA do projeto, o que foi feito, quais experimentos foram executados, quais problemas ainda existem e quais sao os proximos passos recomendados.

Arquivos principais:

- [18_estado_consolidado_e_plano.md](18_estado_consolidado_e_plano.md): estado
  canonico atual, todos os resultados principais, aprendizados e plano.

- [01_historico_do_trabalho.md](01_historico_do_trabalho.md): linha do tempo do que foi feito no sistema e na IA.
- [02_estado_atual_da_ia.md](02_estado_atual_da_ia.md): como a IA funciona hoje dentro do backend/frontend.
- [03_experimentos_e_metricas.md](03_experimentos_e_metricas.md): modelos testados, metricas e conclusoes.
- [04_problemas_conhecidos.md](04_problemas_conhecidos.md): gargalos atuais, falsos positivos e erros de localizacao.
- [05_proximos_passos.md](05_proximos_passos.md): plano tecnico para melhorar a rede neural.
- [06_comandos_e_arquivos_importantes.md](06_comandos_e_arquivos_importantes.md): comandos usados, artefatos e variaveis de ambiente.
- [06_auditoria_v29.md](06_auditoria_v29.md): auditoria temporal da v29.
- [07_experimento_v30_confirmador_froc.md](07_experimento_v30_confirmador_froc.md): confirmador v30.
- [08_investigacao_arquitetural_v31.md](08_investigacao_arquitetural_v31.md): auditoria arquitetural.
- [09_resultado_v31_multiescala.md](09_resultado_v31_multiescala.md): resultado multiescala.
- [10_auditoria_representacao_v32.md](10_auditoria_representacao_v32.md): mascara e saturacao.
- [11_resultado_v33_deltas_temporais.md](11_resultado_v33_deltas_temporais.md): deltas temporais.
- [12_resultados_v34_v35_morfologia.md](12_resultados_v34_v35_morfologia.md): features morfologicas.
- [13_resultado_v36_branch_sinal_bruto.md](13_resultado_v36_branch_sinal_bruto.md): branch bruta.
- [14_resultado_v37_dados_expandidos.md](14_resultado_v37_dados_expandidos.md): primeiro lote expandido.
- [15_resultados_v38_v39_expansao_eventos.md](15_resultados_v38_v39_expansao_eventos.md): sampler por evento.
- [16_resultados_v40_v41_checkpoint_eventos.md](16_resultados_v40_v41_checkpoint_eventos.md): checkpoint por evento.
- [17_auditoria_estabilidade_v40.md](17_auditoria_estabilidade_v40.md): auditoria por crise e seed.

Resumo executivo:

O sistema ja possui integracao completa para o modelo sequencial CNN-LSTM, com fallback para o fluxo legado. A API e a interface ja exibem score, trecho suspeito, top trechos suspeitos e metadados do modelo.

O gargalo atual nao e mais integracao. O gargalo e qualidade do modelo: ele ainda tem falsos positivos e principalmente dificuldade para localizar corretamente o trecho real da crise em pacientes novos.

O candidato experimental mais promissor atual e a v40/v40b. Na calibracao,
ela atingiu 13,33/20 eventos, F1 localizado de 75,0%, F1 EDF de 92,9% e 0,58
FA/h no teto 0,75. Ela ainda nao foi promovida. `chb09`, `chb15` e `chb18` ja
foram consultados em experimentos anteriores e devem ser tratados como
desenvolvimento, nao como teste final intocado.
