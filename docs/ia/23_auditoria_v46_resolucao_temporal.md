# v46: auditoria de resolucao temporal

Atualizado em: 2026-10-07

## Objetivo

Verificar se janelas menores representam melhor as crises de 6-8 segundos que
continuam invisiveis na v43, antes de gerar um novo cache completo e executar
treino caro.

Foram comparadas:

- baseline: janela de 4 s, passo de 2 s, sequencia de 8 passos, contexto 18 s;
- candidata: janela de 2 s, passo de 1 s, sequencia de 16 passos, contexto 17 s.

A auditoria usou `chb16_17` e 11 EDFs de treino representativos, incluindo as
tres novas crises curtas. Pacientes reservados nao foram acessados.

## Cobertura temporal

- `chb16_17`: 10 para 16 sequencias ictais;
- `chb02_19`: 3 para 5;
- `chb21_22`: 3 para 7;
- `chb06_10`, `chb06_13` e `chb06_18`: 3-4 para 7 cada.

## Similaridade com eventos do treino

| Evento | Janelas 4 s | Janelas 2 s | Melhor similaridade 4 s | Melhor similaridade 2 s | Melhor curta 4 s | Melhor curta 2 s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 5 | 0,719 | 0,709 | 0,599 | 0,585 |
| 2 | 2 | 3 | 0,757 | 0,694 | 0,601 | 0,598 |
| 3 | 3 | 4 | 0,662 | 0,719 | 0,517 | 0,637 |
| 4 | 3 | 4 | 0,602 | 0,648 | 0,465 | 0,583 |

## Separacao contra EEG normal local

| Evento | 4 s/2 s | 2 s/1 s | Mudanca |
|---:|---:|---:|---:|
| 1 | 2,487 | 2,235 | piora |
| 2 | 2,172 | 2,009 | piora |
| 3 | 1,470 | 1,793 | melhora |
| 4 | 1,208 | 1,698 | melhora |

## Conclusao

A resolucao de 2 s/1 s nao deve substituir globalmente a de 4 s/2 s. Ela
melhora precisamente os eventos 3 e 4, mas reduz separacao ou similaridade nos
eventos 1 e 2. O resultado sustenta uma hipotese multirresolucao: preservar a
branch de 4 s da v43 e adicionar uma branch curta de 2 s como fonte de
representacao, com fusao antes da saida temporal.

Antes de treinar toda a coorte, o prototipo deve alinhar as duas resolucoes no
mesmo eixo, usar fusao pequena e demonstrar ganho nos eventos 3-4 sem perder
os eventos 1-2.

Artefatos:

- `modelos/v46_temporal_resolution_baseline_subset.json`;
- `modelos/v46_temporal_resolution_audit.json`.
