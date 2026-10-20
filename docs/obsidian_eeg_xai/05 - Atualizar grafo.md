# Atualizar o grafo

## 1. Gerar o grafo local do codigo

No PowerShell, na raiz do projeto:

```powershell
.venv\Scripts\graphify.exe extract . --code-only --out docs\graphify --no-cluster
```

Esse modo indexa a estrutura do codigo sem enviar documentacao ou EDFs a um modelo externo.

## 2. Exportar para o Obsidian

```powershell
.venv\Scripts\graphify.exe export obsidian `
  --graph docs\graphify\graphify-out\graph.json `
  --dir docs\obsidian_eeg_xai\graphify-code
```

Depois, abra `docs/obsidian_eeg_xai` como cofre no Obsidian. O canvas tecnico fica em `graphify-code/graph.canvas`.

## 3. Consultas locais

```powershell
.venv\Scripts\graphify.exe god-nodes --graph docs\graphify\graphify-out\graph.json
.venv\Scripts\graphify.exe path "exame_pipeline" "sequence_inference" --graph docs\graphify\graphify-out\graph.json
.venv\Scripts\graphify.exe explain "exame_pipeline" --graph docs\graphify\graphify-out\graph.json
```

## Limites

O modo `--code-only` gera relacoes estruturais do codigo. Ele nao faz leitura semantica completa dos relatorios Markdown. Para isso, seria necessario configurar um backend de LLM no Graphify. O conteudo academico curado deste cofre foi escrito manualmente para manter metricas e conclusoes sob controle.
