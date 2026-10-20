# Auditoria resolutiva v58

Relatorio completo: `docs/ia/33_auditoria_resolutiva_v58.md`.

## O que foi comprovado

- Treino: 77 EDFs, 16 pessoas distintas, 72 crises, 71,53 minutos ictais.
- Calibracao: quatro pessoas, 20 crises e somente um EDF inteiramente normal.
- Nenhuma nova avaliacao nos pacientes historicos nesta rodada.
- Existe queda importante entre calibracao e historico; mais seeds nao substituem novos pacientes.
- O F1 chamado localizado e por EDF com acerto do trecho principal, nao F1 por segundo.
- A busca FROC do treino testa thresholds apenas de 0,10 a 0,60.

## Tentativa de modelo

v58 = v43 com dropout de 10% dos mapas por canal, somente durante treino.
20 epocas, seed 42; checkpoint na epoca 16. Eventos cairam de 15/20 para
13/20; F1 localizado de 90,9% para 80%; FA/h subiu de 0,549 para 0,706.
**REJEITAR.** Modelo ativo continua [[02 - Estado atual v43]].

## Correcao real aplicada

O treino reconstruia derivacoes bipolares a partir de referencia comum, mas a
inferencia nao. A regra agora e compartilhada. Em quatro segundos do EDF de
treino `chb12_27.edf`, a inferencia recuperou 20 slots, antes sem nenhum slot
bipolar direto. As 345 features ficaram exatamente iguais as do treino.
Isso demonstra paridade da entrada; nao comprova ganho de sensibilidade.
Suite completa de verificacao: 209 testes aprovados. Artefatos oficiais da
v43 conferidos por hash e preservados.

## Proximas decisoes orientadas por evidencia

1. Validacao agrupada que separe escolha de checkpoint, calibracao e avaliacao.
2. Mais negativos completos no conjunto de calibracao.
3. Auditar/normalizar dominio espectral em experimento separado com retreino.
4. Avaliador suplementar por evento 1:1 e bordas nas anotacoes originais.
5. Preservar resultados negativos e evitar nova escolha nos pacientes historicos.
