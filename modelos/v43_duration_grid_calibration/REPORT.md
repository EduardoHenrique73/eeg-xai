# v43 — experimento de duração mínima na calibração

> Análise exploratória restrita aos pacientes de calibração. O modelo e o teste histórico não foram alterados.

## Grade

| Threshold | Duração | F1 EDF | F1 localizado | Eventos | Recall evento | F1 evento | FA/h | Teto atendido |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 60% | 10s | 1.0000 | 0.9091 | 15/20 | 0.7500 | 0.4615 | 2.3548 | não |
| 60% | 14s | 0.9565 | 0.9091 | 15/20 | 0.7500 | 0.6000 | 1.1774 | não |
| 60% | 18s | 0.9565 | 0.9091 | 15/20 | 0.7500 | 0.7143 | 0.5495 | sim |
| 90% | 10s | 0.9565 | 0.8000 | 14/20 | 0.7000 | 0.5714 | 1.1774 | não |
| 90% | 14s | 0.8571 | 0.7368 | 13/20 | 0.6500 | 0.6500 | 0.5495 | sim |
| 90% | 18s | 0.8571 | 0.7368 | 11/20 | 0.5500 | 0.6286 | 0.3140 | sim |

## Decisão

Melhor combinação dentro do teto de FA/h: **threshold 60%, duração 18s**.

Esse resultado é apenas candidato de calibração. Uma eventual promoção exige nova avaliação congelada que não reutilize os EDFs pós-hoc já consultados.

## Salvaguardas

- Gap fixo em 8 s.
- Uma única inferência reutilizada em toda a grade.
- CHB09, CHB15 e CHB18 não foram usados nesta escolha.
- O artefato ativo da v43 não foi modificado.
