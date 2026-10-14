# Comparação controlada da v43: threshold 60% × 90%

> Avaliação pós-hoc. Nenhum parâmetro foi escolhido ou promovido com base nestes arquivos.

## Parâmetros congelados

- Modelo: `C:\Users\Foco\Documents\DEV\eeg-xai\modelos\v43_final_seed42.keras`
- SHA-256 modelo: `7aa86c97ba50226bd24214e57027977ef1f451a5b7e0eaecd81c25e76863ada4`
- SHA-256 scaler: `e4d1e1d2aa5630ae1b68578698c20b88f5a4eff892992d9ea7ebc6d6dda44ab7`
- Duração mínima: 18 s
- Gap: 8 s

## Amostra

- `chb15_01.edf`: avaliacao_historica_reservada; 0 evento(s) anotado(s).
- `chb18_01.edf`: avaliacao_historica_reservada; 0 evento(s) anotado(s).
- `chb18_36.edf`: edf_inedito_paciente_historico; 1 evento(s) anotado(s).
- `chb24_07.edf`: edf_inedito_paciente_de_calibracao; 1 evento(s) anotado(s).

## Resultados

| Métrica | 60% | 90% |
|---|---:|---:|
| Precisão EDF | 0.5000 | 0.5000 |
| Sensibilidade EDF | 0.5000 | 0.5000 |
| F1 EDF | 0.5000 | 0.5000 |
| F1 localizado | 0.5000 | 0.5000 |
| Precisão por evento | 0.0526 | 0.1250 |
| Recall por evento | 0.5000 | 0.5000 |
| F1 por evento | 0.0952 | 0.2000 |
| FA/h | 4.5207 | 1.7581 |
| IoU médio detectado | 0.8571 | 0.9200 |

## Resultados por EDF

| EDF | Classe | Pred. 60% | Pred. 90% | Eventos 60% | Eventos 90% | Alarmes 60% | Alarmes 90% |
|---|---:|---:|---:|---:|---:|---:|---:|
| chb15_01.edf | 0 | 1 | 1 | 0 | 0 | 10 | 3 |
| chb18_01.edf | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| chb18_36.edf | 1 | 1 | 1 | 1 | 1 | 8 | 4 |
| chb24_07.edf | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

## Limitações

- Amostra pequena; não permite inferência estatística robusta.
- CHB15 e CHB18 foram consultados historicamente; CHB24 participou da calibração.
- A comparação não constitui nova calibração nem teste clínico externo independente.
- O threshold é um corte operacional, não uma porcentagem de certeza diagnóstica.

## Conclusão

- Melhor para sensibilidade: **empate (60% e 90%)**.
- Melhor para precisão por evento: **90%**.
- Melhor para localização temporal nesta amostra: **90%**.
- Melhor equilíbrio nesta amostra reduzida: **90%**.
- Decisão científica: manter **60%** como ponto operacional calibrado da v43.
- Interface: 90% pode permanecer somente como comparação experimental identificada, sem substituir a calibração.
