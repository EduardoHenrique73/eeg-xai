# EEG-XAI — Frontend (React + TypeScript)

SPA do Visualizador Clínico de Sinais EEG (wireframe Figura 9 — TCC 1).

## Stack

- **React 19** + **TypeScript** (Vite)
- **Tailwind CSS 4** (`tailwind.config.js` + `postcss.config.js`)
- **Recharts** — plotagem de sinais EEG e overlay SHAP
- **Axios** — consumo da API FastAPI

## Estrutura

```
src/
├── api/              # Cliente HTTP (upload, diagnóstico)
├── components/
│   ├── layout/       # ClinicalLayout (3 colunas)
│   ├── paciente/     # PacienteCard
│   ├── upload/       # EdfDropzone (.edf)
│   ├── visualizador/ # EegSignalChart (Recharts)
│   └── laudo/        # Score IA + parecer médico
├── hooks/            # useDiagnosticoPolling
├── pages/            # VisualizadorClinico
├── types/            # Interfaces alinhadas ao backend
└── data/             # Mock clínico (paciente)
```

## Desenvolvimento

```bash
cd frontend
npm install
npm run dev
```

O Vite faz proxy de `/api` e `/static` para `http://localhost:8000`.

Suba o backend em paralelo:

```bash
uvicorn app.main:app --reload
```

## Build

```bash
npm run build
```

> Requer Node.js **20.19+** (ou use Vite 6 conforme `package.json`).
