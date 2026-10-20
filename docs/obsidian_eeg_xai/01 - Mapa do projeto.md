# Mapa do projeto

```mermaid
flowchart LR
    A[EDF CHB-MIT] --> B[MNE e janelamento]
    B --> C[15 features tempo-frequencia por canal]
    C --> D[CNN-BiLSTM v43]
    D --> E[score por janela]
    E --> F[trechos suspeitos]
    F --> G[interface e SHAP]
    G --> H[apoio ao laudo]
    D --> I[metricas por EDF, evento e tempo]
    I --> J[TCC e apresentacao]
```

## Camadas

- **Dados:** EDFs CHB-MIT com anotacoes oficiais de crise.
- **Processamento:** MNE-Python, janelas temporais e normalizacao baseada somente no treino.
- **Representacao:** caracteristicas espectrais e temporais por canal; a v43 nao usa o sinal bruto diretamente como entrada principal.
- **Modelo:** CNN-BiLSTM com fusao tardia dos canais e saida temporal.
- **Pos-processamento:** threshold, duracao minima e uniao de lacunas para formar eventos.
- **Aplicacao:** FastAPI, React/TypeScript, PostgreSQL na configuracao final e visualizacao multicanal.
- **Explicabilidade:** Gradient SHAP associado as janelas de features da sequencia de pico.

## Relacoes importantes

- [[02 - Estado atual v43]] e a referencia oficial congelada.
- [[03 - Experimento v57]] mostra uma tentativa posterior que nao deve substituir a v43.
- [[04 - TCC e evidencias]] transforma os resultados em narrativa academica transparente.
