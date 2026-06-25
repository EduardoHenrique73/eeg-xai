# EEG-XAI - Plataforma de Diagnostico Neurologico com IA Explicavel

Sistema web para o TCC de apoio a triagem e analise de exames de EEG em formato EDF, com backend FastAPI, frontend React e pipeline de IA explicavel.

## Estado Atual

O projeto ja possui:

- API FastAPI assincrona com documentacao OpenAPI.
- Autenticacao medica com JWT.
- Gestao basica de pacientes.
- Upload e armazenamento de arquivos `.edf` em disco.
- Leitura de EDF real com MNE-Python.
- Visualizacao de sinais EEG com downsampling para o frontend.
- Selecao de canais EEG para analise.
- Extracao de 19 features por dinamica simbolica, estatisticas e entropia de Shannon.
- Inferencia com modelo CNN-LSTM Keras.
- Geracao de grafico SHAP salvo em `storage/shap`.
- Emissao de laudo medico final.
- Testes automatizados com pytest.

Pontos ainda pendentes para aderencia completa ao TCC:

- CRUD completo de pacientes: editar e excluir.
- Zoom/pan real no grafico EEG.
- Persistencia automatica de metadados extraidos do EDF, como taxa de amostragem e canais.
- SHAP temporal sobreposto ao sinal EEG por tempo/canal, ou ajuste do texto do TCC para explicar que o prototipo usa importancia de features.
- Pipeline de treino, janelamento temporal/data augmentation e validacao cruzada com metricas clinicas.
- Tela de configuracoes: threshold, montagem padrao e controle de exibicao SHAP.

## Stack

| Camada | Tecnologia |
| --- | --- |
| API | FastAPI |
| ORM | SQLAlchemy 2.0 async |
| Banco | PostgreSQL em producao; SQLite em desenvolvimento/testes |
| IA/sinais | MNE-Python, TensorFlow/Keras, NumPy, SHAP |
| Frontend | React, Vite, TypeScript, Recharts |
| Testes | pytest, pytest-asyncio |

## Estrutura

```text
eeg-xai/
├── app/
│   ├── main.py              # Aplicacao FastAPI
│   ├── config.py            # Settings via pydantic-settings
│   ├── database.py          # Engine e sessao assincrona
│   ├── models/              # Entidades ORM
│   ├── api/routes/          # Rotas REST
│   ├── services/            # Regras de aplicacao
│   └── ai_engine/           # EDF, features, CNN-LSTM e SHAP
├── frontend/                # SPA React/Vite
├── dataset_amostra/         # EDFs de exemplo para testes locais
├── modelos/                 # Artefatos Keras/PKL
├── storage/
│   ├── edf/                 # Arquivos EDF
│   └── shap/                # Imagens SHAP
└── tests/                   # Testes automatizados
```

## Modelagem de Dados

| Tabela | Descricao |
| --- | --- |
| `usuarios` | Medico, CRM, email e hash de senha |
| `pacientes` | Dados cadastrais do paciente |
| `exames` | Metadados do exame e caminho do EDF |
| `predicoes_ia` | Score da IA e caminho do mapa SHAP |

Regra do projeto: sinais brutos e imagens ficam no disco; o banco guarda metadados e caminhos.

## Setup

```powershell
cd C:\Users\Foco\Documents\DEV\eeg-xai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Para desenvolvimento local sem PostgreSQL, use SQLite no `.env`:

```env
DATABASE_URL=sqlite+aiosqlite:///./eeg_xai_dev.db
```

## Testes

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Pipeline Cientifico Inicial

O projeto inclui uma primeira versao simples de data augmentation por janelamento temporal e k-fold sobre as 19 features do EEG.

```powershell
.venv\Scripts\python.exe scripts\simple_kfold_windows.py
```

O script:

- le os EDFs em `dataset_amostra/`;
- usa o `chb01-summary.txt` para localizar intervalos de crise;
- divide o sinal em janelas temporais sobrepostas;
- rotula janelas como normal/crise por sobreposicao temporal;
- extrai as 19 features ja usadas pela IA;
- roda k-fold estratificado com um classificador leve;
- salva metricas em `modelos/metrics_kfold_simple.json`.

Esta etapa ainda nao substitui o treinamento clinico final da CNN-LSTM; ela serve como base reprodutivel para documentar janelamento, data augmentation e validacao cruzada no TCC.

## Executar API

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Documentacao interativa: http://localhost:8000/docs

## Executar Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Observacao Sobre o Modelo

O arquivo em `modelos/cnn_lstm_hybrid.keras` deve ser substituido por um modelo treinado e validado para uso cientifico. Em ambiente de desenvolvimento, o pipeline pode criar um modelo dummy se o artefato estiver ausente; esse comportamento serve apenas para testes tecnicos e nao deve ser apresentado como modelo clinicamente validado.
