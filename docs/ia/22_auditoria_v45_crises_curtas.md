# v45: auditoria e treino com crises curtas

Atualizado em: 2026-10-07

## Auditoria de representacao

Foram comparados os quatro eventos de `chb16_17` com 72 eventos do treino,
usando a mediana das 15 features tempo-frequencia em 23 canais e similaridade
de cosseno.

- evento 1: 9 s, maior similaridade 0,719;
- evento 2: 6 s, maior similaridade 0,757;
- evento 3: 8 s, maior similaridade 0,662;
- evento 4: 8 s, maior similaridade 0,602;
- nenhum evento do treino atingiu similaridade 0,8;
- apenas um dos 72 eventos do treino tinha ate 10 segundos;
- 15 tinham ate 20 segundos e 57 tinham mais de 20 segundos.

Os vizinhos morfologicamente mais proximos duravam 52-143 segundos. Entre os
eventos curtos, as melhores similaridades ficaram entre 0,46 e 0,60. A forca
das features tambem caiu do evento 1 para o 4, principalmente nas bandas beta,
alpha, delta e theta.

## Dados adicionados

Foram baixados do espelho publico oficial do PhysioNet:

- `chb06_10.edf`: crise de 12 s, tres sequencias positivas;
- `chb06_13.edf`: crise de 13 s, quatro sequencias positivas;
- `chb06_18.edf`: crise de 12 s, tres sequencias positivas.

Os arquivos foram validados por tamanho, leitura MNE, 23 canais e duracao. O
paciente `chb06` ja pertencia ao treino. Nenhum paciente de calibracao ou
reservado foi movido entre conjuntos.

## Experimento controlado

A v45 manteve arquitetura, loss, sampler, checkpoint e pos-processamento da
v43. Mudou apenas o manifesto, acrescentando as tres crises curtas. Foram
executadas seeds 42-44; a rodada nao foi ampliada porque nao houve ganho no
criterio principal.

| Modelo | Eventos | F1 evento | F1 localizado | F1 EDF | FA/h |
|---|---:|---:|---:|---:|---:|
| v43, seeds 42-44 | 14,0/20 | 66,7% | 83,4% | 97,1% | 0,63 |
| v45, seeds 42-44 | 14,0/20 | 67,2% | 75,4% | 92,5% | 0,60 |

Resultados v45 por seed sob `FA/h <= 0,75`:

| Seed | Eventos | F1 localizado | F1 EDF | FA/h |
|---:|---:|---:|---:|---:|
| 42 | 15/20 | 85,7% | 95,7% | 0,71 |
| 43 | 13/20 | 66,7% | 90,9% | 0,63 |
| 44 | 14/20 | 73,7% | 90,9% | 0,47 |

## Eventos invisiveis

Os scores de `chb16_17` eventos 2-4 nao melhoraram consistentemente:

- evento 2: v43 `0,038/0,002/0,574`; v45 `0,082/0,002/0,001`;
- evento 3: v43 `0,022/0,005/0,503`; v45 `0,010/0,013/0,002`;
- evento 4: v43 `0,051/0,409/0,723`; v45 `0,003/0,030/0,128`.

A v45 recuperou alguns eventos de outros arquivos, mas perdeu eventos antes
detectados. Isso caracteriza troca de erros, nao generalizacao melhor.

## Decisao

**REJEITAR.** A falta de crises curtas era real, mas duracao semelhante nao
implica morfologia semelhante. Adicionar tres crises curtas do mesmo paciente
nao resolveu a representacao dos eventos invisiveis e piorou localizacao e F1
EDF. A v43 permanece como baseline experimental.

## Proximo passo

Auditar resolucao temporal antes de novo treino completo. O pipeline atual usa
janelas de 4 s com passo de 2 s; crises de 6-8 s produzem apenas 2-3 alvos
ictais. Comparar, em uma amostra dirigida, features de janelas de 2 s com passo
de 1 s. Somente criar novo cache e retreinar toda a coorte se essa resolucao
separar melhor os eventos 2-4 de seus trechos normais e aproximar sua assinatura
de crises do treino.

Artefatos:

- `modelos/v45_invisible_event_audit.json`;
- `modelos/v45_short_seizures_calibration.json`;
- `dataset_amostra/manifests/v45_short_seizures_train.txt`.
