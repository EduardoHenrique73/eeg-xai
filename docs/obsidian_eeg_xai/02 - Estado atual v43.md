# Estado atual | v43

## Papel

A v43 e a baseline congelada usada no produto, no TCC e na comparacao das tentativas posteriores.

## Configuracao congelada

| Item | Valor |
| --- | --- |
| Modelo | CNN-BiLSTM com fusao tardia de canais |
| Entrada | 23 slots de canais x 15 features por timestep |
| Janela / passo | 4 s / 2 s |
| Sequencia | 8 timesteps, stride 2 |
| Threshold documentado | 0,60 |
| Duracao minima | 18 s |
| Fechamento de lacunas | 8 s |
| Seed final | 42 |
| Treino | 77 EDFs de 17 grupos CHB |
| Calibracao | 13 EDFs de CHB16, CHB19, CHB20 e CHB24 |

## Resultados oficiais

### Calibracao

- Precisao por EDF: 100%
- Sensibilidade por EDF: 91,7%
- F1 por EDF: 95,7%
- F1 localizado: 90,9%
- Eventos: 15/20
- FA/h: 0,55

### Avaliacao historica

- Precisao por EDF: 87,5%
- Sensibilidade por EDF: 77,8%
- F1 por EDF: 82,4%
- F1 localizado: 46,2%
- Eventos: 4/9
- FA/h: 2,08

## Interpretacao correta

O modelo apresenta bom comportamento no conjunto de calibracao, mas cai ao enfrentar os registros historicos interpaciente. Isso evidencia a dificuldade de generalizacao temporal e topografica entre pessoas. A v43 e adequada como resultado final documentado porque o protocolo, os artefatos e os limites estao registrados; ela nao deve ser apresentada como diagnostico clinico pronto.

## Artefatos

- Modelo: `modelos/v43_final_seed42.keras`
- Scaler: `modelos/v43_final_seed42_scaler.pkl`
- Metadata: `modelos/v43_final_seed42_metadata.json`
- Calibracao: `modelos/v43_final_seed42_calibration.json`
- Relatorio: `docs/ia/20_resultado_v43_fusao_tardia_canais.md`
