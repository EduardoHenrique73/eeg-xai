# Problemas conhecidos

Atualizado em: 2026-07-03

## 1. Falsos positivos

O modelo ainda marca trechos normais como suspeitos.

Isso aparece principalmente em:

- arquivos normais classificados como crise;
- top trechos suspeitos em regioes sem anotacao real;
- predicoes positivas sem sobreposicao com a crise real.

No modelo `expanded_20260702`, o teste separado mostrou:

```text
Falsos positivos: 4
Predicoes positivas sem overlap: 11
```

Impacto:

- pode induzir o medico a revisar trechos desnecessarios;
- reduz confianca no sistema;
- piora a utilidade clinica mesmo quando o F1 geral parece aceitavel.

## 2. Falsos negativos

O modelo ainda deixa algumas crises passarem.

No modelo `expanded_20260702`:

```text
Falsos negativos: 3
```

Impacto:

- risco mais grave do ponto de vista clinico;
- o sistema nao pode ser usado como unica triagem;
- precisa melhorar recall sem explodir falsos positivos.

## 3. Erro de localizacao temporal

Este e o problema mais importante atualmente.

O modelo pode acertar que o arquivo contem crise, mas apontar o trecho errado.

Exemplo do `expanded_20260702`:

```text
F1 geral:        0.8108
F1 localizacao:  0.3077
```

Isso mostra que a metrica geral esconde o problema real.

Impacto:

- a interface pode mostrar um trecho suspeito errado;
- o medico pode perder tempo revisando area normal;
- o sistema nao cumpre completamente o objetivo de apoiar a localizacao da crise.

## 4. Generalizacao para paciente novo

O objetivo do projeto e receber exame de uma pessoa nunca vista antes.

Por isso, validacao intra-paciente nao e suficiente.

Problema:

- EEG varia muito entre pacientes;
- montagem/canais variam;
- duracao e morfologia das crises variam;
- um modelo pode aprender caracteristicas de paciente em vez de padrao de crise.

Consequencia:

- metricas boas em split aleatorio podem ser enganosas;
- o correto e separar pacientes entre treino, calibracao e teste.

## 5. Base ainda pequena para o objetivo

Apesar de existirem 93 EDFs completos localmente, isso ainda e pouco para uma CNN-LSTM robusta inter-paciente.

O problema nao e so quantidade de arquivos, mas composicao:

- precisa de mais pacientes;
- precisa de mais crises;
- precisa de mais arquivos normais completos;
- precisa de pre-crise, pos-crise e normal distante;
- precisa manter teste totalmente separado.

## 6. Balanceamento dificil

Crise ocupa pouco tempo dentro de arquivos longos.

Se amostrar janelas de forma ingenua:

- o modelo ve muito mais normal que crise;
- ou, se superamostrar crise demais, passa a achar crise em tudo;
- falsos positivos aumentam;
- localizacao fica instavel.

O balanceamento precisa considerar:

- janelas durante crise;
- janelas imediatamente antes da crise;
- janelas imediatamente depois da crise;
- janelas normais distantes;
- arquivos normais completos.

## 7. Calibracao pode enganar

O modelo `expanded_20260702` teve calibracao muito boa:

```text
F1 localizado calibracao: 0.9333
```

Mas teste separado ruim:

```text
F1 localizado teste: 0.3077
```

Conclusao:

Nao basta escolher threshold/duracao que funciona na calibracao. E necessario confirmar em teste separado por paciente.

## 8. Threshold nao resolve tudo

Aumentar threshold pode reduzir falso positivo, mas tambem pode aumentar falso negativo.

Aumentar duracao minima pode remover picos falsos, mas pode perder crises curtas ou trechos pequenos corretamente detectados.

Portanto, threshold/duracao ajudam, mas nao resolvem de vez um modelo mal generalizado.

## 9. Sinal bruto nao foi melhor

Foi testado caminho com `raw_signal`, mas os resultados foram fracos nos experimentos registrados.

Isso nao significa que sinal bruto nunca funcionara. Significa que, na arquitetura/base atual, ele nao foi o melhor caminho.

## 10. Dinamica simbolica agregada por media tambem nao foi suficiente

O caminho `symbolic_mean` apresentou F1 geral razoavel, mas localizacao muito ruim.

Conclusao:

Se dinamica simbolica continuar no projeto, deve ser usada com cuidado, talvez por canal e combinada com tempo-frequencia, nao como media simples que perde informacao espacial/temporal.

## 11. Fragmentacao de janelas antes da CNN-LSTM

Foi identificado que o amostrador podia selecionar, por exemplo, 12 janelas de
crise em quatro blocos de 3. Como a CNN-LSTM exige sequencias de 8 janelas,
nenhum desses blocos gerava uma amostra positiva. Um EDF com crise podia chegar
ao treino apenas com sequencias normais.

Correcao aplicada:

- selecao por classe preserva blocos com no minimo `sequence_length` janelas;
- existe teste automatizado garantindo que uma crise amostrada gera sequencia positiva;
- o cache inclui o comprimento e o modo de alvo na chave.

## 12. Desvio de score entre pacientes

No teste do `center_v2`, os scores mudaram muito por paciente. O paciente
`chb11` teve score maximo medio aproximado de 0.55 e gerou falsos positivos,
enquanto `chb14` ficou perto de 0.17 e perdeu varias crises.

Isso indica que um threshold global nao consegue compensar sozinho diferencas
de amplitude, montagem, distribuicao espectral e morfologia entre pacientes.
As features atuais incluem potencias absolutas por banda, que podem contribuir
para esse desvio mesmo com `StandardScaler` global.

Foi implementado um modo relativo e normalizacao robusta por EDF. O `v5`
melhorou a localizacao para F1 `0.4828`, mas o teste continuou muito pior que a
calibracao (`0.9231`). Portanto, variacao de paciente e composicao do conjunto
de calibracao continuam sendo problemas abertos.

O conjunto de teste atual ja foi consultado em varios experimentos. Proximos
ajustes de modelo precisam de pacientes adicionais nunca usados, senao ha risco
de otimizar indiretamente para esse teste.

## 13. Normalizacao treino/inferencia corrigida

Os experimentos robustos antigos estimavam mediana e IQR com janelas normais
rotuladas durante treino/validacao. Essa informacao nao existe ao analisar um
EDF novo. O pipeline foi corrigido para usar todas as janelas tanto no treino
quanto na inferencia. Por isso, resultados antigos de `v4` e `v5` nao sao
diretamente comparaveis com producao e devem ser tratados como exploratorios.

## 14. Teste antigo nao esta mais intacto

Os pacientes `chb03`, `chb11`, `chb13` e `chb14` foram consultados repetidas
vezes para comparar candidatos. Eles podem continuar como benchmark historico,
mas nao como evidencia final imparcial do TCC. E necessario reservar novos
pacientes antes de qualquer nova escolha de arquitetura, threshold ou duracao.

## 15. Gargalo atual de desempenho

O `v7` reduziu predicoes positivas espurias no nivel de EDF, mas perdeu muitas
crises. O problema atual e o compromisso especificidade/sensibilidade entre
pacientes, nao falta de epocas. A validacao amostrada tambem nao mede de forma
confiavel alarmes por hora; essa metrica exige EDF completo.

## 16. Qualidade heterogenea dos EDFs

Parte do CHB-MIT possui nomes de canais duplicados, canais auxiliares com nome
invalido e ao menos um aviso de divergencia entre cabecalho e tamanho do arquivo.
O sistema ignora canais auxiliares invalidos e fixa a montagem de referencia,
mas o relatorio final deve registrar esses avisos e os criterios de exclusao.

## 17. Normalizacao robusta apos amostragem

Foi identificado que o treino calculava mediana/IQR apenas sobre as janelas ja
balanceadas. Como essa amostra super-representa crise e vizinhanca da crise, a
escala era diferente da inferencia, que processa o EDF completo. O fluxo foi
corrigido para normalizar todas as janelas antes da amostragem. Resultados com
`all_sampled_windows_v3` continuam historicos e nao devem ser comparados como
se usassem a mesma entrada da producao.

## 18. Criterio de calibracao excessivamente conservador

A validacao leave-one-patient-out priorizava numero de falsos positivos antes
de F1 e recall. Isso reduzia alarmes, mas selecionava thresholds que perdiam
muitas crises. Existem agora dois modos explicitos:

- `balanced`: prioriza F1 localizado e recall;
- `conservative`: prioriza reduzir falsos positivos.

Para o TCC, ambos podem ser relatados, mas o modo usado deve ser declarado. O
modo balanceado melhorou a localizacao de 2/9 para 4/9 crises na `v10`, ainda
insuficiente para uso clinico.

## 19. Pico curto podia ocultar crise sustentada

O trecho principal era escolhido pela media dos scores. No `chb18_29`, isso
colocou um falso trecho de 12 segundos acima do trecho de 52 segundos que
sobrepunha 44 segundos da crise real. O sistema agora ranqueia os candidatos
pela evidencia acumulada acima do threshold, depois de verificar a duracao
minima. A mudanca melhorou o F1 localizado da configuracao `v11` com calibracao
ampliada, mas nao aumentou o total de crises localizadas alem de 4/9.

## 20. Calibracao pequena e instavel entre pacientes

Dois pacientes de calibracao (`chb20` e `chb24`) nao representavam bem todas as
distribuicoes de teste. Na `v11`, ampliar para quatro pacientes reduziu os
falsos positivos de 3 para 1 e elevou o F1 localizado medio de `0.5000` para
`0.6111`, mas reduziu a sensibilidade em relacao a configuracao mais agressiva.
Ainda e necessario congelar uma coorte de calibracao maior e diversa antes do
teste final intocado.

## 21. Poucas crises independentes e perda sem objetivo temporal

O manifesto expandido possui 54 EDFs, 15 pacientes e 55 eventos de crise. Nas
dobras `v11`, entretanto, o treino contem somente 182 a 214 sequencias positivas,
muitas sobrepostas e derivadas do mesmo evento. O numero efetivo de exemplos
independentes e, portanto, muito menor que o numero de janelas.

Alem disso, a CNN-LSTM atual produz um unico rotulo para a janela central de
cada sequencia. A loss nao otimiza diretamente inicio/fim, continuidade ou
overlap do evento. Isso ajuda a explicar por que 6/9 eventos aparecem entre os
candidatos, mas apenas 4/9 sao escolhidos como trecho principal.

## 22. Segmentacao detecta o EDF, mas ainda fragmenta alguns eventos

A `v14` eliminou falsos negativos por EDF nos tres pacientes de desenvolvimento,
mas manteve sensibilidade por evento em 6/9. No `chb18_30`, 8 das 15 janelas
ictais ficaram acima do threshold, porem nao formaram um trecho continuo com a
duracao minima calibrada. Isso indica que o proximo ajuste deve tratar pequenas
lacunas temporais e artefatos sem simplesmente reduzir o threshold.

O mesmo modelo gerou 6.493 falsos alarmes por hora nos EDFs normais do `chb18`.
Qualquer suavizacao, histerese ou uniao de lacunas deve ser escolhida apenas na
calibracao e limitada por uma meta explicita de falsos alarmes por hora.

## 23. Duas causas diferentes para as crises nao localizadas

A auditoria das `v16` a `v19` separou as falhas restantes:

- algumas crises, como `chb15_10` e `chb15_17`, recebem score baixo durante o
  intervalo real; pos-processamento nao corrige uma representacao nao aprendida;
- outras, como `chb18_30`, produzem picos dentro da crise, mas curtos ou
  fragmentados, que perdem para candidatos falsos mais longos.

Fechamento de lacunas e histerese ajudam apenas o segundo grupo. Aumentar
indiscriminadamente as bordas positivas (`v19`) piorou a generalizacao. O
primeiro grupo exige mais eventos independentes e diversidade interpaciente,
nao apenas mais janelas sobrepostas dos mesmos eventos.

## 24. A coorte de desenvolvimento esta esgotada para escolha imparcial

Os mesmos tres pacientes foram usados para comparar varias arquiteturas,
perdas, thresholds e regras temporais. As metricas continuam uteis para depurar,
mas escolher outra regra a partir delas aumenta o risco de sobreajuste ao teste.
O proximo ganho precisa ser selecionado em treino/calibracao ampliados e medido
uma unica vez em pacientes ainda intocados.

## 25. Confirmador nao recupera candidato inexistente ou nao transferivel

O detector em dois estagios confirmou o limite esperado: hard negatives ajudam
a filtrar candidatos, mas o segundo modelo nao consegue recuperar uma crise que
o primeiro estagio nao representou adequadamente. Em `chb15`, reduzir o limiar
de geracao de `0.10` para `0.05` nao recuperou eventos e aumentou a densidade de
falsos candidatos. Antes de retestar o confirmador, o gerador precisa ser
retreinado com mais crises independentes e validado em pacientes separados.

## 26. Avaliacoes historicas tinham isolamento incompleto da coorte

Nas `v16/v18`, somente o paciente testado na dobra corrente era removido. Os
outros pacientes declarados como teste podiam entrar no treino ou na
calibracao. O validador agora aceita `reserved_test_patients` e impede que toda
a coorte reservada participe do ajuste. A referencia justa recalculada e menor
que o numero historico e deve ser usada em comparacoes futuras.

## 27. Expansao melhorou o gerador, mas aumentou alarmes

A `v23b`, com 92 EDFs e calibracao fixa, melhorou o F1 localizado de `0.5460`
para `0.6667` e as crises principais de 4/9 para 5/9. Entretanto, os falsos
alarmes subiram de `1.1027/h` para `1.3622/h`. Isso impede tratar o ganho como
prontidao clinica e exige calibracao futura em pacientes ainda intocados.

## 28. Contexto temporal pode ser curto para eventos fragmentados

Com oito passos, janelas de quatro segundos e passo de dois segundos, cada
sequencia cobre aproximadamente 18 segundos. Isso pode ser insuficiente para
distinguir atividade ictal sustentada de transientes e para manter continuidade
em crises mais longas. A hipotese deve ser testada alterando apenas o tamanho da
sequencia, sem escolher o resultado final na coorte de desenvolvimento.

## 29. Ensemble e confirmador atuais nao melhoraram a v23b

A media de tres redes (`v24`) diluiu alguns eventos reconhecidos por modelos
individuais. O confirmador (`v25`) manteve 6/9 eventos, mas elevou os falsos
alarmes para `1.7514/h`. Ambos foram rejeitados. A infraestrutura permanece
util para pesquisas futuras, mas nao deve ser ativada no produto atual.

## 30. Amostragem anterior desperdicava sequencias continuas

O limite por classe era aplicado nas janelas antes da montagem das sequencias.
Blocos com lacunas eram descartados depois, fazendo um limite de 96 janelas
normais produzir apenas cerca de 20 a 34 sequencias em varios EDFs. O modo
`sampling-level=sequence` corrige a ordem e preserva mais dados validos. Apesar
de aumentar a sensibilidade por evento para 7/9 na `v27`, os limites atuais
aumentaram falsos alarmes e reduziram a qualidade do trecho principal.

## 31. Confirmador ainda e excessivamente conservador

Na `v28`, o confirmador reduziu os falsos alarmes de `1.6865/h` para `0.5838/h`,
mas voltou de 7/9 para 6/9 eventos. Em algumas dobras, a sensibilidade OOF da
calibracao ficou entre 25% e 60%. Antes de usar o segundo estagio, e necessario
ampliar a coorte de calibracao ou treina-lo com objetivo explicito de recall sob
um limite de alarmes por hora.

## 32. Seeds efetivas variavam entre dobras

O validador usa `random_state + fold_idx` para evitar inicializacoes identicas,
mas os artefatos antigos registravam apenas parametros gerais e nao a seed
efetiva. O JSON agora salva `random_state_base` e `random_seeds` em cada dobra.
Na `v29`, a selecao do limite usou seed 42, enquanto as dobras reservadas usaram
43, 44 e 45. Isso nao causa vazamento, mas adiciona variancia e deve ser tratado
em repeticoes futuras ou em um unico modelo congelado para toda a coorte final.
