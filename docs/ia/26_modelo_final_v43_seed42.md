# Modelo final reproduzivel v43 seed 42

## Objetivo

Transformar a melhor configuracao controlada da v43 em artefatos carregaveis pela aplicacao, sem usar os pacientes reservados `chb09`, `chb15` e `chb18` durante treino, selecao de checkpoint ou calibracao.

## Configuracao

- arquitetura: fusao tardia por canal com CNN-BiLSTM;
- entrada: `time_frequency_per_channel`;
- 23 canais e 15 features por canal;
- sequencia de 8 passos, stride 2, alvo temporal central;
- janela de 4 s e passo de 2 s;
- normalizacao robusta por EDF;
- amostragem balanceada por evento;
- seed 42;
- checkpoint selecionado por recall de evento sob `FA/h <= 0,75`;
- melhor checkpoint: epoca 14.

## Calibracao reproduzida

No conjunto formado pelos pacientes `chb16`, `chb19`, `chb20` e `chb24`:

| Metrica | Resultado |
|---|---:|
| Eventos detectados | 15/20 |
| Recall por evento | 75,0% |
| Precisao por evento | 68,2% |
| F1 por evento | 71,4% |
| F1 localizado | 90,9% |
| F1 por EDF | 95,7% |
| Falsos alarmes por hora | 0,55 |

Ponto operacional: threshold `0,60`, duracao minima `18 s` e fechamento de lacunas de ate `8 s`.

## Artefatos

- `modelos/v43_final_seed42.keras`;
- `modelos/v43_final_seed42_scaler.pkl`;
- `modelos/v43_final_seed42_metadata.json`;
- `modelos/v43_final_seed42_calibration.json`;
- `modelos/v54_v43_final_seed42_training.json`.

O `.env` local aponta explicitamente para os quatro artefatos de inferencia.

## Correcoes de integracao

1. A aplicacao passou a agregar corretamente a saida temporal `[sequencia, passo, 1]`, calculando a media das janelas sobrepostas.
2. O `max_gap_seconds` calibrado passou a ser aplicado na inferencia e exposto pela API/interface.
3. A mascara de canais ausentes da fusao tardia passou a usar uma camada Keras serializavel, sem habilitar desserializacao insegura.
4. Zeros de canais ausentes voltam a ser preservados depois da aplicacao do scaler.

## Smoke test

O EDF normal `chb11_01.edf` foi processado integralmente. O maior pico foi `0,902`, mas o trecho persistiu por apenas `12 s`; como a duracao minima e `18 s`, o resultado conclusivo permaneceu negativo. Foram agregadas 1.798 janelas temporais.

## Interpretacao correta

Estes resultados demonstram reproducibilidade no conjunto de calibracao e funcionamento ponta a ponta. Eles nao substituem a avaliacao final nos pacientes reservados. A divulgacao no TCC deve separar claramente treino, calibracao e teste e evitar linguagem de validacao clinica.

A avaliacao congelada posterior no reservado historico esta documentada em `27_avaliacao_congelada_v43_reservado_historico.md`. Ela mostrou queda para 4/9 eventos e 2,08 FA/h, confirmando que a generalizacao interpaciente ainda e o principal gargalo.
