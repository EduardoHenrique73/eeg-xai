# Documentacao da IA do EEG-XAI

Atualizado em: 2026-10-07

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
- [19_resultado_v42_detector_duplo.md](19_resultado_v42_detector_duplo.md): detector temporal de dois caminhos e decisao de rejeicao.
- [20_resultado_v43_fusao_tardia_canais.md](20_resultado_v43_fusao_tardia_canais.md): frontend por canal, cinco seeds e auditoria de estabilidade.
- [21_resultado_v44_estabilidade_otimizacao.md](21_resultado_v44_estabilidade_otimizacao.md): learning rate, checkpoint e resultado negativo da v44.
- [22_auditoria_v45_crises_curtas.md](22_auditoria_v45_crises_curtas.md): lacuna de crises curtas, novos EDFs e resultado da v45.
- [23_auditoria_v46_resolucao_temporal.md](23_auditoria_v46_resolucao_temporal.md): cobertura e separabilidade com janelas de 2 s/1 s.
- [24_resultados_v47_v48_multirresolucao.md](24_resultados_v47_v48_multirresolucao.md): fusao 4 s + 2 s, correcao do pareamento e decisao das v47/v48.

Resumo executivo:

O sistema ja possui integracao completa para o modelo sequencial CNN-LSTM, com fallback para o fluxo legado. A API e a interface ja exibem score, trecho suspeito, top trechos suspeitos e metadados do modelo.

O gargalo atual nao e mais integracao. O gargalo e qualidade do modelo: ele ainda tem falsos positivos e principalmente dificuldade para localizar corretamente o trecho real da crise em pacientes novos.

O candidato experimental mais promissor atual e a v40/v40b. Na calibracao,
ela atingiu 13,33/20 eventos, F1 localizado de 75,0%, F1 EDF de 92,9% e 0,58
FA/h no teto 0,75. Ela ainda nao foi promovida. `chb09`, `chb15` e `chb18` ja
foram consultados em experimentos anteriores e devem ser tratados como
desenvolvimento, nao como teste final intocado.

A v42 adicionou um caminho curto de alta confianca, mas foi rejeitada: elevou
o FA/h medio de 0,58 para 0,86 e reduziu o F1 localizado de 75,0% para 71,0%,
sem ganho consistente de eventos.

A v43 com fusao espacial tardia e o candidato arquitetural mais promissor:
13,8/20 eventos, F1 localizado de 81,9%, F1 EDF de 96,4% e 0,64 FA/h em cinco
seeds. Ainda nao foi promovida devido a pior seed de 12/20 e eventos
persistentes com score baixo.

A v44 tentou estabilizar a v43 com learning rate menor e reducao por plateau,
mas caiu para 13,4/20 eventos e F1 localizado de 76,7%. Foi rejeitada; a v43
permanece como candidata experimental.

A v45 adicionou tres crises curtas do `chb06`, mas nao recuperou os eventos
invisiveis e piorou localizacao e F1 EDF. Foi rejeitada. A proxima investigacao
deve avaliar resolucao temporal de 2 s/1 s antes de gerar um novo cache completo.

A auditoria v46 mostrou ganho da resolucao curta nos eventos 3-4, mas piora
nos eventos 1-2. O proximo passo e um prototipo multirresolucao que preserve a
branch de 4 s, e nao uma substituicao global do janelamento.

A v47 confirmou informacao complementar: no alvo exploratorio, o score maximo
medio dos eventos 2/3/4 passou de 0,317/0,543/0,837 para
0,666/0,719/0,988. O percentil 99 normal, porem, subiu de 0,948 para 0,991.
Ela e promissora, mas nao foi promovida. A v48 residual foi rejeitada por
instabilidade. O proximo passo e medir a v47 em toda a calibracao com FROC.
