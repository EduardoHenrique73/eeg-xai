# Relatório de transferência técnica — EEG-XAI

Data de consolidação: 14/10/2026  
Branch: `main`  
Objetivo: permitir que outra IA ou desenvolvedor continue o trabalho sem reconstruir o histórico do projeto.

## 1. Visão geral

O EEG-XAI é uma aplicação web de apoio à análise de EEG para localização temporal de atividade compatível com crise epiléptica. O sistema não produz diagnóstico clínico autônomo. Ele oferece:

- cadastro de médicos e pacientes;
- upload de EDF;
- visualização multicanal do EEG;
- inferência temporal por CNN-BiLSTM;
- apresentação de candidatos temporais suspeitos;
- Gradient SHAP projetado por canal e janela temporal;
- elaboração e persistência de laudo médico;
- configurações de montagem, exibição de SHAP e preferências do perfil.

Stack principal:

- backend: Python, FastAPI, SQLAlchemy assíncrono, MNE-Python e TensorFlow/Keras;
- frontend: React, TypeScript, Vite, Tailwind e Recharts;
- banco final declarado: PostgreSQL;
- banco de compatibilidade e testes: SQLite;
- dataset científico: CHB-MIT Scalp EEG Database.

## 2. Modelo de IA ativo

O modelo final adotado é a v43, seed 42:

- arquitetura: CNN-BiLSTM com fusão tardia de 23 canais;
- entrada: sequências de 15 características tempo-frequência por canal;
- normalização: robusta por EDF;
- saída: probabilidade ictal por passo temporal;
- explicabilidade: Gradient SHAP;
- modelo: `modelos/v43_final_seed42.keras`;
- scaler: `modelos/v43_final_seed42_scaler.pkl`;
- metadados: `modelos/v43_final_seed42_metadata.json`;
- calibração: `modelos/v43_final_seed42_calibration.json`.

Hashes registrados no experimento mais recente:

- modelo SHA-256: `7aa86c97ba50226bd24214e57027977ef1f451a5b7e0eaecd81c25e76863ada4`;
- scaler SHA-256: `e4d1e1d2aa5630ae1b68578698c20b88f5a4eff892992d9ea7ebc6d6dda44ab7`.

## 3. Ponto operacional científico congelado

O ponto oficial da v43 é:

- threshold: `0,60`;
- duração mínima: `18 s`;
- fechamento de lacunas: `8 s`;
- máximo de falsos alarmes na seleção: `0,75 FA/h`;
- seed: `42`.

Correção importante realizada em 14/10/2026:

- a aplicação enviava `usuario.threshold_confianca` ao modelo sequencial;
- um perfil configurado em 90% substituía silenciosamente os 60% calibrados;
- o ramo sequencial agora ignora preferências pessoais e usa o artefato científico;
- a preferência do médico permanece apenas para o fluxo legado;
- a interface exibe o limiar efetivamente usado no resultado.

Arquivos envolvidos:

- `app/services/exame_pipeline.py`;
- `app/api/routes/exames.py`;
- `frontend/src/pages/VisualizadorClinico.tsx`;
- `frontend/src/components/laudo/ScoreConfianca.tsx`;
- `frontend/src/pages/Configuracoes.tsx`;
- `.env.example`.

Análises antigas persistidas com 90% não são recalculadas automaticamente. É necessário criar uma nova análise para aplicar 60%.

## 4. Separação dos dados da v43

Treinamento:

- 77 EDFs;
- 17 grupos de pacientes;
- arquivos listados em `modelos/v54_v43_final_seed42_training.json`.

Calibração:

- 13 EDFs;
- pacientes `chb16`, `chb19`, `chb20` e `chb24`;
- realizada separadamente do treinamento.

Avaliação histórica reservada:

- pacientes `chb09`, `chb15` e `chb18`;
- esses pacientes não treinaram a v43, mas foram consultados durante o desenvolvimento;
- por isso o material chama essa avaliação de histórica, e não de teste externo independente.

Todos os grupos disponíveis no CHB-MIT participaram de treino, calibração ou avaliação histórica. É possível testar um EDF novo, mas não obter um paciente CHB-MIT completamente intocado pelo processo.

## 5. Resultados oficiais da v43

Calibração:

- precisão EDF: 100%;
- sensibilidade EDF: 91,7%;
- F1 EDF: 95,7%;
- F1 localizado: 90,9%;
- eventos detectados: 15/20;
- F1 por evento: 71,4%;
- falsos alarmes: 7;
- FA/h: 0,55.

Avaliação histórica:

- F1 EDF: 82,4%;
- F1 localizado: 46,2%;
- evidencia dificuldade de generalização interpaciente.

Documentos principais:

- `docs/ia/20_resultado_v43_fusao_tardia_canais.md`;
- `docs/ia/26_modelo_final_v43_seed42.md`;
- `docs/ia/27_avaliacao_congelada_v43_reservado_historico.md`;
- `docs/ia/29_fechamento_tcc.md`;
- `docs/ia/30_alinhamento_textual_tcc_ii.md`.

## 6. Comparação pós-hoc de threshold 60% × 90%

Experimento executado em quatro EDFs, reutilizando exatamente os mesmos scores:

- `chb15_01.edf`: normal, avaliação histórica;
- `chb18_01.edf`: normal, avaliação histórica;
- `chb18_36.edf`: EDF novo de paciente histórico, crise oficial em 463–509 s;
- `chb24_07.edf`: EDF novo de paciente de calibração, crise oficial em 38–60 s.

Parâmetros fixos:

- duração mínima: 18 s;
- gap: 8 s;
- modelo, scaler, features e montagem inalterados.

Resultados:

| Métrica | 60% | 90% |
|---|---:|---:|
| Precisão EDF | 50,0% | 50,0% |
| Sensibilidade EDF | 50,0% | 50,0% |
| F1 EDF | 50,0% | 50,0% |
| F1 localizado | 50,0% | 50,0% |
| Precisão por evento | 5,3% | 12,5% |
| Recall por evento | 50,0% | 50,0% |
| F1 por evento | 9,5% | 20,0% |
| FA/h | 4,52 | 1,76 |
| IoU médio da crise detectada | 85,7% | 92,0% |

Interpretação:

- 90% reduziu os falsos alarmes de 18 para 7 nessa amostra;
- não perdeu evento adicional em relação a 60%;
- a amostra é pequena e inclui pacientes já consultados/calibrados;
- 90% não foi promovido;
- 60% permanece o padrão científico.

Artefatos:

- `scripts/compare_v43_thresholds.py`;
- `modelos/v43_threshold_comparison/results.json`;
- `modelos/v43_threshold_comparison/REPORT.md`;
- `modelos/v43_threshold_comparison/*_thresholds.png`.

## 7. Descoberta no chb24_07

A anotação oficial da crise é 38–60 s.

- com 60%, a rede marcou 40–56 s, duração de 16 s;
- com 90%, marcou 40–54 s, duração de 14 s;
- ambos se sobrepõem à crise real;
- ambos foram rejeitados como eventos porque não atingiram os 18 s mínimos.

Isso mostra que a crise não ficou invisível para a rede. O pós-processamento impediu a conclusão positiva. Entretanto, um caso isolado não autoriza reduzir a duração global.

## 8. Grade de duração na calibração

Para investigar o caso acima sem usar os EDFs pós-hoc na seleção, foi executada uma grade exclusivamente nos 13 EDFs de calibração:

- thresholds: 60% e 90%;
- durações: 10, 14 e 18 s;
- gap fixo: 8 s;
- uma única inferência reutilizada em todas as combinações.

| Threshold | Duração | Eventos | F1 evento | F1 localizado | FA/h |
|---:|---:|---:|---:|---:|---:|
| 60% | 10 s | 15/20 | 46,2% | 90,9% | 2,35 |
| 60% | 14 s | 15/20 | 60,0% | 90,9% | 1,18 |
| **60%** | **18 s** | **15/20** | **71,4%** | **90,9%** | **0,55** |
| 90% | 10 s | 14/20 | 57,1% | 80,0% | 1,18 |
| 90% | 14 s | 13/20 | 65,0% | 73,7% | 0,55 |
| 90% | 18 s | 11/20 | 62,9% | 73,7% | 0,31 |

Conclusão:

- reduzir a duração com threshold 60% não recuperou eventos;
- apenas aumentou os falsos alarmes;
- 90% perdeu eventos e piorou a localização na calibração;
- `60% + 18 s + 8 s` permanece o melhor equilíbrio dentro do teto de 0,75 FA/h.

Artefatos:

- `scripts/experiment_v43_duration_grid.py`;
- `modelos/v43_duration_grid_calibration/results.json`;
- `modelos/v43_duration_grid_calibration/REPORT.md`;
- `modelos/v43_duration_grid_calibration/duration_threshold_grid.png`.

## 9. Estado do backend

Fluxo principal:

1. upload e validação do EDF;
2. armazenamento do arquivo;
3. extração de sinais/features;
4. construção das sequências;
5. scaler e inferência v43;
6. agregação temporal;
7. classificação cautelosa;
8. Gradient SHAP opcional;
9. persistência da predição;
10. retorno do diagnóstico e dos trechos pela API.

Arquivos centrais:

- `app/config.py`;
- `app/ai_engine/feature_extractor.py`;
- `app/ai_engine/sequence_inference.py`;
- `app/ai_engine/sequence_shap.py`;
- `app/services/exame_pipeline.py`;
- `app/services/exame_sinais.py`;
- `app/api/routes/exames.py`;
- `app/database.py`.

Correções recentes incluem:

- descarte de canais auxiliares inválidos como `.` e `-`;
- tratamento de nomes duplicados;
- sobreposição SHAP por canal/tempo;
- dados do traçado servidos em janelas navegáveis;
- validação de campos de paciente;
- mensagens e notificações mais claras.

## 10. Estado do frontend

A interface atual inclui:

- design clínico azul e branco;
- campos com máscaras e validações;
- painel temporal da IA;
- destaque dos candidatos suspeitos;
- seleção direta de candidatos;
- gráfico EEG ampliável;
- modal de visualização em tela maior;
- zoom e navegação temporal;
- marcadores para saltar aos trechos candidatos;
- overlay SHAP alinhado às janelas explicadas;
- textos explícitos de apoio à decisão, sem alegação diagnóstica.

Arquivos centrais:

- `frontend/src/pages/VisualizadorClinico.tsx`;
- `frontend/src/components/visualizador/EegSignalChart.tsx`;
- `frontend/src/components/laudo/IaLaudoPanel.tsx`;
- `frontend/src/components/laudo/ScoreConfianca.tsx`;
- `frontend/src/pages/Configuracoes.tsx`;
- `frontend/src/pages/GestaoPacientes.tsx`;
- `frontend/src/utils/form.ts`.

## 11. Banco de dados e portabilidade

Decisão final do projeto/TCC: PostgreSQL.

Situação atual:

- `.env.example` contém URL PostgreSQL;
- a aplicação foi validada localmente com PostgreSQL 16, banco `EEG`;
- SQLite foi preservado para compatibilidade, testes e recuperação;
- segredos locais permanecem em arquivos ignorados pelo Git.

Pendência:

- `scripts/portable_installation.py` e `docs/TRANSFERIR_PARA_OUTRO_PC.md` ainda descrevem snapshot/restauração SQLite;
- para uma reprodução totalmente alinhada ao TCC, o instalador do segundo computador deve criar/configurar PostgreSQL, aplicar o esquema e importar os dados necessários;
- não remover o suporte SQLite usado pelos testes.

## 12. TCC II

Versão mais recente:

- `docs/tcc/TCC II - Eduardo Marques - Versao Final ABNT.docx`.

O documento inclui:

- folha de aprovação;
- agradecimentos;
- resumo e abstract;
- listas de figuras, quadros e siglas;
- organização do trabalho;
- metodologia e decisões de implementação;
- interface implementada;
- resultados da v43;
- limitações e ameaças à validade;
- referências e apêndice.

O DOCX mantém 19 imagens e campos automáticos de página para sumário/listas. A conferência estrutural passou, mas a revisão visual final em um editor compatível continua recomendada antes da entrega.

O experimento 60% × 90% e a grade de duração ainda não foram incorporados automaticamente ao TCC. Se forem adicionados, devem entrar como análise exploratória complementar, sem substituir as métricas oficiais da v43.

## 13. Reprodutibilidade em outro computador

Já existem:

- `scripts/portable_installation.py`;
- `scripts/setup_new_pc.ps1`;
- `scripts/create_transfer_archive.py`;
- `docs/TRANSFERIR_PARA_OUTRO_PC.md`;
- teste `tests/test_portable_installation.py`.

Procedimento científico sugerido:

1. congelar código, modelo, scaler e calibração;
2. registrar hashes;
3. instalar em segundo computador;
4. configurar PostgreSQL;
5. confirmar que a API carrega a v43;
6. processar um EDF conhecido para equivalência;
7. processar um EDF não usado no treino;
8. registrar hardware, versões, tempo e resultado;
9. não ajustar parâmetros após observar o resultado.

## 14. Validações recentes

- build do frontend: aprovado;
- 38 testes direcionados de treino/inferência: aprovados;
- teste do ponto operacional `0,60 / 18 s / 8 s`: aprovado;
- artefatos JSON/PNG/Markdown dos experimentos: validados;
- suíte integrada de duas rotas apresentou travamento de fixture em uma tentativa e foi interrompida; não declarar suíte completa como aprovada sem nova execução.

## 15. Problemas ainda relevantes

1. Generalização interpaciente ainda é o principal limite científico.
2. Alguns EDFs apresentam regiões longas de score alto e múltiplos falsos alarmes.
3. Crises curtas podem ser reconhecidas parcialmente, mas descartadas pela duração mínima.
4. Reduzir globalmente a duração não resolveu o recall e piorou FA/h na calibração.
5. A saída do modelo não é probabilidade clinicamente calibrada; scores próximos de 100% não significam certeza diagnóstica.
6. O SHAP explica features/janelas, não cada amostra bruta do EEG.
7. A avaliação pós-hoc de quatro EDFs é pequena e não deve promover 90%.
8. Falta finalizar a instalação portátil com PostgreSQL.
9. Falta executar e documentar o teste no segundo computador.
10. Falta decidir com o orientador se os experimentos exploratórios recentes entram no texto final.

## 16. Próximos passos recomendados

Prioridade 1 — reprodutibilidade:

- adaptar o setup do novo computador para PostgreSQL sem remover SQLite dos testes;
- executar a instalação em segundo ambiente;
- registrar evidências e tempos.

Prioridade 2 — avaliação final:

- baixar EDFs completos e validar tamanho/hash antes do uso;
- executar protocolo congelado sem ajustar parâmetros;
- documentar acertos e falhas.

Prioridade 3 — interface:

- confirmar visualmente que análises novas exibem 60%;
- impedir que resultados antigos de 90% sejam confundidos com o ponto atual;
- considerar exibir “ponto científico calibrado” na tela.

Prioridade 4 — TCC:

- incluir uma subseção curta sobre a influência do pós-processamento;
- apresentar a grade de duração como evidência de que 18 s foi mantido por equilíbrio, não por escolha arbitrária;
- manter explícito que 90% não foi promovido;
- revisar paginação e campos automáticos antes do PDF definitivo.

## 17. Comandos úteis

Backend:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Frontend:

```powershell
cd frontend
npm run dev
```

Build:

```powershell
cd frontend
npm run build
```

Testes:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Threshold pós-hoc:

```powershell
.venv\Scripts\python.exe scripts\compare_v43_thresholds.py
```

Grade de duração na calibração:

```powershell
.venv\Scripts\python.exe scripts\experiment_v43_duration_grid.py
```

## 18. Regras para continuação

- Não alterar treino, threshold, duração ou gap olhando os pacientes reservados.
- Não tratar score como certeza diagnóstica.
- Não promover configuração por resultado de amostra pequena.
- Não misturar EDFs do mesmo paciente entre treino e teste.
- Não substituir a v43 sem comparação controlada no mesmo protocolo.
- Não sobrescrever os artefatos finais; criar versões novas.
- Não remover o suporte SQLite necessário aos testes enquanto PostgreSQL é finalizado.
- Não versionar `.env`, senhas, bancos locais, EDFs ou dados clínicos.

