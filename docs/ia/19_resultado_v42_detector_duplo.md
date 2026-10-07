# Resultado v42: detector temporal de dois caminhos

Atualizado em: 2026-10-06

## Hipotese

A v42 testou se uma regra temporal adicional, curta e de alta confianca,
recuperaria crises breves eliminadas pela duracao minima do caminho sustentado.
Nenhuma arquitetura, loss, amostragem ou dado foi alterado.

- caminho sustentado: ponto da FROC v40b sob teto de 0,75 FA/h;
- caminho transiente: threshold em 0,70, 0,80, 0,90 ou 0,95;
- duracao transiente: 6, 8 ou 10 segundos;
- gap transiente: zero;
- decisao final: uniao dos trechos aceitos pelos dois caminhos;
- seeds: 42, 43 e 44;
- conjunto: calibracao fixa com 20 eventos;
- pacientes historicamente reservados: nao acessados (`test_accessed=false`).

O criterio previo exigia superar 13,33/20 eventos, manter F1 localizado de pelo
menos 75%, F1 EDF de pelo menos 90% e FA/h de no maximo 0,75.

## Resultado por seed

| Seed | Metodo | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h | Falsos alarmes |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | sustentado | 13/20 | 61,9% | 63,2% | 91,7% | 0,71 | 9 |
| 42 | duplo | 13/20 | 56,5% | 63,2% | 91,7% | 1,02 | 13 |
| 43 | sustentado | 12/20 | 66,7% | 80,0% | 90,9% | 0,31 | 4 |
| 43 | duplo | 13/20 | 61,9% | 73,7% | 90,9% | 0,71 | 9 |
| 44 | sustentado | 15/20 | 68,2% | 81,8% | 96,0% | 0,71 | 9 |
| 44 | duplo | 15/20 | 65,2% | 76,2% | 96,0% | 0,86 | 11 |

### Media das tres seeds

| Metodo | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h | Falsos alarmes |
|---|---:|---:|---:|---:|---:|---:|
| sustentado v40b | 13,33/20 | 65,6% | 75,0% | 92,9% | 0,58 | 7,33 |
| detector duplo | 13,67/20 | 64,1% | 71,0% | 92,9% | 0,86 | 11,00 |

A media do detector duplo nao representa um ponto operacional valido sob o
teto de 0,75: as seeds 42 e 44 ultrapassaram esse limite. Somente a seed 43
produziu um ponto combinado viavel no teto solicitado.

## Comparacao por crise

A auditoria foi recalculada com os scores mascarados pelos trechos realmente
aceitos pela uniao dos caminhos. Apenas uma diferenca de deteccao apareceu:

| Seed | Arquivo | Evento | Sustentado | Duplo | Score maximo | Sobreposicao |
|---:|---|---:|---|---|---:|---:|
| 43 | `chb24_04.edf` | 3 | perdido | detectado | 0,9968 | 16 s |

Nas seeds 42 e 44, o caminho transiente nao recuperou evento adicional. Em
contrapartida, adicionou quatro e dois falsos alarmes, respectivamente. Na
seed 43, recuperou um evento, mas adicionou cinco falsos alarmes e reduziu o
F1 localizado de 80,0% para 73,7%.

## Decisao

**REJEITAR.** A v42 nao superou os criterios predefinidos de forma consistente:

- ganho medio de apenas 0,34 evento em 20;
- nenhum ganho em duas das tres seeds;
- FA/h medio aumentou de 0,58 para 0,86;
- F1 localizado caiu de 75,0% para 71,0%;
- F1 de evento caiu de 65,6% para 64,1%;
- duas seeds violaram o teto de 0,75 FA/h.

Nenhum modelo ou regra da v42 deve ser promovido para producao.

## Aprendizado

A duracao minima global explica parte dos eventos perdidos, mas nao e o
gargalo dominante. Um caminho curto baseado apenas em score e duracao nao
distingue crises breves de transientes normais de alta confianca. O primeiro
estagio ainda precisa representar melhor os eventos invisiveis; alternativamente,
um futuro confirmador de transientes precisaria ser treinado com categorias de
hard negatives, e nao ser apenas uma segunda regra de threshold.

## Proximo passo

Priorizar reprodutibilidade e uma ablacao de representacao para eventos com
score ictal baixo:

1. salvar checkpoint, scaler, metadata, calibracao e traces com hashes;
2. identificar eventos persistentemente invisiveis em varias seeds;
3. testar frontend temporal por canal com fusao espacial tardia;
4. manter arquitetura, loss e pos-processamento restantes fixos;
5. avaliar no mesmo protocolo, com pelo menos cinco seeds antes de promover.

Artefato completo: `modelos/v42_dual_path_calibration.json`.
