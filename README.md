# EEG-XAI — Plataforma de Diagnóstico Neurológico com IA Explicável

Evolução do sistema legado PIBITI para o **TCC 2**, com arquitetura moderna assíncrona.

## Stack

| Camada | Tecnologia |
|--------|------------|
| API | FastAPI (assíncrono) |
| ORM | SQLAlchemy 2.0 + asyncpg |
| Banco | PostgreSQL |
| Testes | pytest + pytest-asyncio (TDD) |
| IA (futuro) | mne-python, TensorFlow CNN-LSTM, SHAP |

## Estrutura

```
eeg-xai/
├── app/
│   ├── main.py              # FastAPI
│   ├── config.py            # Settings (pydantic-settings)
│   ├── database.py          # Engine assíncrono
│   ├── models/              # Entidades ORM
│   └── ai_engine/           # Motor de IA (placeholders)
├── tests/                   # TDD — testes antes da implementação
├── storage/
│   ├── edf/                 # Arquivos .edf (disco, não no banco)
│   └── shap/                # Mapas SHAP (disco, não no banco)
└── requirements.txt
```

## Modelagem de dados

| Tabela | Descrição |
|--------|-----------|
| `usuarios` | Médico (CRM, email, senha_hash) |
| `pacientes` | Sujeito do exame (vinculado ao médico) |
| `exames` | Metadados + `arquivo_path` do .edf |
| `predicoes_ia` | Score + `mapa_shap_path` |

**Regra de ouro:** arrays de sinais e imagens ficam no disco; o PostgreSQL guarda apenas `_path`.

## Setup

```powershell
cd C:\Users\Foco\Documents\DEV\eeg-xai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Testes (TDD)

```powershell
pytest -v
```

## Executar API

```powershell
uvicorn app.main:app --reload
```

Documentação interativa: http://localhost:8000/docs

## Próximas fases

1. Rotas de upload `.edf` com HTTP 202 + Background Tasks
2. Implementação do `ai_engine` (mne, CNN-LSTM, SHAP)
3. Autenticação JWT para médicos
4. Alembic migrations
