# EEG-XAI - Plataforma de Diagnostico Neurologico com IA Explicavel

Sistema web para o TCC de apoio a triagem e analise de exames de EEG em formato EDF, com backend FastAPI, frontend React e pipeline de IA explicavel.

## Estado Atual

O projeto ja possui:

- API FastAPI assincrona com documentacao OpenAPI.
- Autenticacao medica com JWT.
- Gestao basica de pacientes.
- Upload e armazenamento de arquivos `.edf` em disco.
- Leitura de EDF real com MNE-Python.
- Visualizacao multicanal de sinais EEG com downsampling e trechos suspeitos.
- Selecao de canais EEG para analise.
- Extracao tempo-frequencia por canal, estatisticas, entropia de Shannon e dinamica simbolica.
- Inferencia temporal com CNN-BiLSTM e fusao tardia de 23 canais.
- Gradient SHAP do modelo sequencial salvo em `storage/shap`.
- Emissao de laudo medico final.
- Configuracoes medicas de threshold, montagem e exibicao SHAP.
- Isolamento de pacientes, exames e estatisticas por medico autenticado.
- Testes automatizados com pytest.

Pontos cientificos ainda pendentes:

- validacao externa em pacientes nunca consultados durante o desenvolvimento;
- estudo formal de usabilidade com profissionais da area;
- validacao clinica, que esta fora do escopo dos resultados atuais.

Todos os 24 pacientes CHB-MIT disponiveis localmente ja foram usados em treino,
calibracao ou desenvolvimento historico. Por isso, `chb09/chb15/chb18` nao sao
descritos como teste final intocado.

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

Para transferir a instalacao atual completa (modelo, EDFs, banco e laudos) para
outro PC Windows, veja `docs/TRANSFERIR_PARA_OUTRO_PC.md`. O Git sozinho nao
inclui esses dados locais.

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

## Modelo Experimental Ativo

O ambiente local usa a v43 seed 42:

- features `time_frequency_per_channel`;
- 23 canais, com identidade espacial preservada ate a fusao tardia;
- CNN-BiLSTM com probabilidade ictal por passo temporal;
- janelas de 4 s, passo de 2 s e sequencias de 8 passos;
- threshold 0,60, duracao minima 18 s e gap closing de 8 s;
- checkpoint escolhido na calibracao por evento sob limite de FA/h.

Na calibracao, o artefato reproduziu F1 EDF de 95,7%, F1 localizado de 90,9%,
15/20 eventos e 0,55 FA/h. No desenvolvimento historico interpaciente, caiu
para F1 EDF de 82,4%, 4/9 eventos e 2,08 FA/h. Essa diferenca e documentada e
impede qualquer alegacao de validacao clinica.

Consulte `docs/ia/29_fechamento_tcc.md` para metodologia, resultados e limites.

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

Tambem ha um treino preliminar da arquitetura CNN-LSTM sobre as janelas:

```powershell
.venv\Scripts\python.exe scripts\train_cnn_lstm_windows.py --epochs 5
```

Por padrao, ele salva:

- `modelos/cnn_lstm_hybrid_prelim.keras`
- `modelos/cnn_lstm_hybrid_prelim_scaler.pkl`
- `modelos/metrics_cnn_lstm_windows.json`

Esse modelo preliminar fica separado do `cnn_lstm_hybrid.keras` usado pela aplicacao, para evitar substituir o motor atual sem validacao maior.

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

O fluxo sequencial e um sistema de apoio experimental, nao um dispositivo de
diagnostico. O fallback/dummy existe somente para testes tecnicos. O pacote da
v43 nao e versionado diretamente porque `.keras` e `.pkl` estao no `.gitignore`;
use `scripts/verify_release_artifacts.py` para conferir os arquivos locais e o
manifesto SHA-256 antes da demonstracao.
