# EEG-XAI | Inicio

> Cofre de acompanhamento do sistema, dos experimentos e do TCC.

## Navegacao

- [[01 - Mapa do projeto]]
- [[02 - Estado atual v43]]
- [[03 - Experimento v57]]
- [[04 - TCC e evidencias]]
- [[05 - Atualizar grafo]]
- [[06 - Auditoria resolutiva v58]]

## Estado em 20/10/2026

- Baseline oficial do produto e do TCC: [[02 - Estado atual v43]].
- v57 foi um experimento controlado de balanceamento por paciente e **nao foi promovida**.
- v58 testou regularizacao por canal e foi rejeitada: 13/20 eventos contra 15/20 da baseline na calibracao.
- A auditoria v58 corrigiu a reconstrucao da montagem referencial na inferencia, validada com features iguais as do treino.
- O ganho da v57 foi reduzir alarmes, mas ela nao recuperou eventos historicos perdidos.
- O objetivo cientifico permanece deteccao temporal interpaciente com apoio explicavel, nao diagnostico autonomo.

## Fontes no repositorio

- `docs/ia/README.md`: indice completo dos experimentos.
- `docs/ia/20_resultado_v43_fusao_tardia_canais.md`: relatorio arquitetural da v43.
- `docs/ia/32_experimento_v57_balanceamento_pacientes.md`: experimento mais recente.
- `docs/tcc/overleaf_eeg_xai/`: fonte LaTeX do TCC.
- `docs/tcc/ROTEIRO_DEFESA_IDEIA_E_RESULTADOS.md`: roteiro da apresentacao.

## Regra de leitura

Resultados de calibracao e avaliacao historica nao devem ser misturados. A calibracao escolhe o ponto operacional; a avaliacao historica mede o comportamento em registros reservados ao protocolo daquela versao.
