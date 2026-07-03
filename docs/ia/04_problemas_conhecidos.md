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

