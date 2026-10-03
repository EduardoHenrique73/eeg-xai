# v31: ablação arquitetural multiescala

## Decisão

**REJEITAR para promoção.** A arquitetura multiescala reduziu falsos alarmes na
calibração, mas não generalizou aos três pacientes reservados. O modelo ativo e o
Random Forest não foram alterados. O artefato reproduzível de seleção e avaliação
é `modelos/v31_multiscale_ablation_calibration.json`.

## Protocolo e diagnóstico arquitetural

O manifesto fixo contém 93 EDFs, 87 crises e 4.687 s ictais. Treino: 17
pacientes, 68 EDFs e 58 crises; calibração: quatro pacientes, 13 EDFs e 20
crises; desenvolvimento reservado: `chb09`, `chb15`, `chb18`, 12 EDFs e nove
crises. O treino amostrado contém 4.053 sequências, das quais 789 têm alvo
positivo e 3.264 são negativas. Os tempos não ictais aproximados são 358.923 s,
45.949 s e 55.540 s, respectivamente. Sequências sobrepostas não são
observações independentes.

A CNN-BiLSTM de referência recebe 345 features (23 canais x 15 features) e
mistura canais já na primeira convolução. Seu alvo tem oito passos. A v31
testou uma projeção compartilhada por canal, convoluções temporais em três
escalas (dilatações 1/2/4), TCN residual compacta e saída sigmoide por passo.
O contexto de entrada sobe a 24 passos, mas o alvo continua nos mesmos oito
passos centrais: não há pooling temporal global nem perda de exemplos por
alongamento do alvo. A variante selecionada **não** usa channel attention.
Ela tem 24.993 parâmetros, contra 125.185 da baseline. A comparação das dez
famílias arquiteturais e as referências estão em
`docs/ia/08_investigacao_arquitetural_v31.md`.

Cada ablação usou seed 42, limite de 20 épocas, early stopping por PR-AUC na
calibração, weighted BCE e o mesmo grid de limiar/duração/gap, salvo as losses
explicitamente indicadas. A escolha foi feita exclusivamente na calibração:
FA/h <= 1, depois recall de evento, F1 localizado, precisão de evento e FA/h.
O confirmador permaneceu desligado. As reavaliações reservadas usam seeds
43/44/45 e recalibram cada modelo somente nos mesmos quatro pacientes de
calibração. O split foi verificado por paciente.

## Ablação na calibração

| Variante | Mudança | Eventos | F1 localizado | FA/h | F1 EDF |
|---|---|---:|---:|---:|---:|
| A | CNN-BiLSTM, weighted BCE | 13/20 | 73,7% | 0,94 | 90,9% |
| B | TCN compacta | 12/20 | 85,7% | 0,86 | 95,7% |
| C | B + multiescala/contexto auxiliar | 13/20 | 80,0% | 0,71 | 95,7% |
| D | C + channel attention | 10/20 | 73,7% | 0,71 | 85,7% |
| E1 | D + boundary loss | 12/20 | 80,0% | 0,86 | 95,7% |
| E2 | D + Tversky temporal | 13/20 | 80,0% | 0,94 | 95,7% |
| E3 | D + continuidade | 11/20 | 66,7% | 0,94 | 90,9% |
| F | D + boundary + sampler por evento | 6/20 | 40,0% | 0,71 | 80,0% |

C foi selecionada antes de abrir o desenvolvimento reservado. B teve F1
localizado maior, mas 12/20 eventos, contra 13/20 de C. F mistura sampler e
loss: a queda não pode ser atribuída isoladamente ao sampler. O sampler mantém
o teto de 32 sequências positivas por EDF e privilegia início/meio/fim; nesta
base ele trocou índices, não aumentou as 789 sequências positivas.

## FROC na calibração

Melhor ponto de C sob cada teto de falsos alarmes/hora, sem consultar teste:

| Teto FA/h | FA/h obtido | Eventos | Precisão por evento | F1 localizado |
|---:|---:|---:|---:|---:|
| 0,50 | 0,24 | 6/20 | 66,7% | 58,8% |
| 0,75 | 0,71 | 13/20 | 59,1% | 80,0% |
| 1,00 | 0,71 | 13/20 | 59,1% | 80,0% |
| 1,25 | 0,71 | 13/20 | 59,1% | 80,0% |
| 1,50 | 1,33 | 14/20 | 45,2% | 80,0% |

Essa é uma FROC da **calibração**. No desenvolvimento reservado medimos apenas
os pontos operacionais previamente definidos por fold; não escolhemos um novo
ponto olhando seu resultado.

## Resultado interpaciente reservado

| Métrica | v23b | v29 | v31 C |
|---|---:|---:|---:|
| F1 EDF macro | 86,7% | 86,7% | 88,9% |
| Precisão EDF macro | - | - | 91,7% |
| Sensibilidade EDF macro | - | - | 86,7% |
| FP / FN por EDF | - | - | 1 / 2 |
| Eventos detectados | 6/9 | 7/9 | **5/9** |
| Precisão / F1 por evento (micro) | - | - | 15,6% / 24,4% |
| F1 localizado macro | 66,7% | 61,1% | **65,1%** |
| Trechos principais corretos | 5/9 | 5/9 | **4/9** |
| Falsos alarmes | - | - | **27** |
| FA/h | 1,36 | 1,30 | **1,75** |

Para v31, o denominador de FA/h é 15,416 horas não ictais. A sensibilidade
por evento foi 55,6%. Entre os cinco eventos detectados, IoU temporal média
0,803, erro absoluto médio de início 2,8 s, erro absoluto médio de fim 18,4 s,
latência média 1,6 s; duração prevista média 90,4 s versus real 72,4 s.
Contando os quatro eventos perdidos como IoU zero, a IoU média dos nove é
0,446. As médias condicionais aos detectados não descrevem as crises perdidas.

## Comparação por crise

`Detectado` requer >= 1 s e >= 25% de cobertura real por um candidato válido.
O erro de início é apresentado apenas para eventos detectados. Score máximo
é medido dentro do intervalo ictal anotado, não é probabilidade clínica
calibrada.

| EDF | Duração real | Score max v29 | Score max v31 | Detectado v29/v31 | Sobreposição v31 | IoU v31 | Erro início v31 | Principal v31 |
|---|---:|---:|---:|---|---:|---:|---:|---|
| chb09_06 | 66 s | 1,000 | 1,000 | sim/sim | 66 s | 0,971 | -2 s | sim |
| chb09_19 | 64 s | 1,000 | 1,000 | sim/sim | 64 s | 0,762 | -2 s | sim |
| chb15_06 | 128 s | 0,969 | 0,917 | sim/não | 0 s | 0 | - | não |
| chb15_10 | 34 s | 0,317 | 0,269 | não/não | 0 s | 0 | - | não |
| chb15_17 | 38 s | 0,936 | 0,433 | sim/não | 0 s | 0 | - | não |
| chb15_20 | 58 s | 0,969 | 0,998 | sim/sim | 58 s | 0,935 | -2 s | não |
| chb15_31 | 122 s | 0,996 | 0,999 | sim/sim | 114 s | 0,919 | +8 s | sim |
| chb18_29 | 52 s | 0,996 | 1,000 | sim/sim | 52 s | 0,426 | 0 s | sim |
| chb18_30 | 32 s | 0,243 | 0,043 | não/não | 0 s | 0 | - | não |

`chb15_10` e `chb18_30` continuam invisíveis ao primeiro estágio; o score
ictal máximo até diminuiu. `chb15_17` perdeu um candidato válido que a v29
encontrava. `chb15_06` tem pico alto na v31, mas sem duração/continuidade
suficiente para virar evento válido. `chb15_20` aparece em candidato secundário,
enquanto o trecho principal do EDF continua incorreto. `chb09_06` melhorou a
escolha do trecho principal frente à v29. `chb18_29` cobre a crise, mas a
previsão longa piora a IoU.

O relatório JSON registra, por evento, o canal cuja **feature de potência
normalizada** teve maior contraste ictal-versus-normal, além dos cinco
contrastes por banda. Nos casos perdidos `chb15_10`, `chb15_17` e `chb18_30`,
esse canal foi `T7-P7`. Isso é uma pista descritiva, **não** atribuição de
importância da rede nem evidência de foco clínico. A normalização robusta por
EDF e clipping em [-10, 10] podem saturar contrastes; valores próximos de 10
não representam energia física em unidades do sinal.

## Limitações e próximo gargalo

Os três pacientes reservados já foram vistos em versões anteriores: esta é
comparação controlada de desenvolvimento, não validação externa independente.
Os mesmos quatro pacientes de calibração servem para early stopping e seleção
de ponto operacional, o que pode otimizar demais a calibração. Variância entre
seeds é grande: os limiares escolhidos por fold foram 0,20, 0,55 e 0,60. Com
apenas nove eventos reservados, uma crise altera muito o resultado.

A máscara de canal ausente é inferida após o `StandardScaler`; zero original
pode deixar de ser zero, portanto essa máscara ainda precisa ser auditada.
Os hard negatives não têm categorias clínicas anotadas; não é honesto relatar
FA/h por artefato, ritmo não ictal ou transiente sem rotulagem. A tabela por
crise mostra associação de bandas/canais, mas não prova causalidade. Também
não há modelo v31 salvo para promoção: este foi um experimento, não deployment.

O gargalo agora é representação de crises interpaciente difíceis e rejeição de
alarmes sustentados. A próxima investigação deve, **somente em treino e
calibração**, auditar saturação/ausência de canais, conservar máscara antes da
escala, revisar potência absoluta versus relativa e anotar uma amostra dos
falsos alarmes por categoria. Só então testar uma alteração de representação
por vez, mantendo este split congelado e acrescentando pacientes inéditos
para validação final. Não ajustar limiar ou arquitetura aos nove eventos acima.

Meta não atingida: >= 7/9 eventos, F1 localizado > 66,7%, FA/h <= 1,0 e
F1 EDF >= 86,7% simultaneamente. Apenas a última condição foi satisfeita.
