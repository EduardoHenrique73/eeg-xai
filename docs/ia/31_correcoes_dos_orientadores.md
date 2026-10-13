# Consolidacao das correcoes dos orientadores

## Documentos analisados

As observacoes presentes nos tres PDFs de revisao foram tratadas como comentarios editoriais e academicos. Sugestoes repetidas foram consolidadas. Nenhuma observacao foi interpretada como evidência de funcionalidade, validacao ou resultado que nao exista no projeto.

## Decisoes de escopo

O produto deve ser definido como **prototipo funcional de apoio a analise e triagem de EEG**, e nao como sistema diagnostico completo. O escopo essencial do TCC II compreende ingestao de EDF, visualizacao multicanal, processamento temporal, CNN-BiLSTM, explicabilidade, configuracoes medicas e emissao de laudo. Validacao clinica, certificacao regulatoria, integracao hospitalar por HL7 FHIR, recuperacao real de senha por e-mail e trilha completa de auditoria permanecem como trabalhos futuros.

## Matriz de correcoes

| Local | Sugestao consolidada | Correcao a aplicar | Situacao real |
|---|---|---|---|
| Capa/folha de rosto | Corrigir localidade | Usar `URUAÇU/GO` conforme modelo institucional | Textual |
| Introducao | Explicitar as barreiras enfrentadas | Identificar fragmentacao do fluxo, leitura manual extensa, baixa interoperabilidade e opacidade do modelo | Texto preparado abaixo |
| Introducao | Evitar promessa de produtividade e seguranca | Substituir por "com potencial para apoiar a produtividade e a revisao clinica" | Obrigatorio |
| Introducao | Incluir acessibilidade e regulacao | Mencionar acessibilidade como requisito de interface e a RDC Anvisa 657/2022 como contexto para eventual SaMD | Sem alegar conformidade |
| Objetivo geral | Definir o produto final | Declarar "desenvolver e avaliar um prototipo funcional" | Obrigatorio |
| Objetivos | Explicar como medir | Associar objetivos a metricas de software, EDF, evento, localizacao temporal e FA/h | Implementado |
| Objetivos | Validacao com medicos | Nao declarar como realizada; registrar como trabalho futuro | Nao realizada |
| Objetivos/metodologia | Evitar data leakage | Declarar divisao por paciente antes do janelamento e da amostragem | Aplicado no protocolo final |
| Justificativa | Reforcar problema vindo do PIBITI | Conectar o trabalho anterior ao problema de oferecer fluxo web completo, e nao somente classificador isolado | Texto preparado |
| Justificativa | Acessibilidade e custo | Explicar acesso via navegador e processamento centralizado, sem afirmar que elimina todo custo de infraestrutura | Parcialmente atendido |
| Referencial 2.1 | Verificar WHO 2024 | Remover da afirmacao sobre envelhecimento se a fonte nao sustentar diretamente o dado; usar fonte especifica | Revisao bibliografica pendente |
| Referencial 2.1 | Corrigir uso de LeCun et al. | Nao atribuir desempenho de EEG ao artigo geral de Deep Learning; usar estudo especifico de EEG | Obrigatorio |
| Referencial 2.2/Metodologia | Especificar metodo agil | Descrever desenvolvimento incremental individual, versionado por Git e orientado por ciclos experimentais; nao alegar Scrum/XP formal sem registros | Evidencia disponivel |
| Referencial 2.3 | Detalhar persistencia | Informar PostgreSQL para metadados e filesystem para EDF/modelos/mapas; nao foi usado pgvector, `.npy` ou `.h5` como persistencia clinica | Implementado |
| Referencial 2.3 | Corrigir referencia React | Atualizar referencia da documentacao oficial e ano de acesso | Pendente no texto final |
| Referencial 2.4 | XAI em neurociencia | Incluir fonte especifica sobre interpretabilidade aplicada a EEG; Craik et al. e revisao de EEG nao devem ser descritos como validacao do SHAP sem verificacao | Revisao bibliografica pendente |
| Referencial 2.5 | Interoperabilidade | Delimitar interoperabilidade ao EDF e API REST; HL7 FHIR e DICOM nao foram implementados | Trabalho futuro |
| Metodologia | Inserir figura das fases | Criar fluxo: requisitos -> dados -> preprocessamento -> treino -> calibracao -> avaliacao -> integracao -> verificacao | Pendente de figura |
| Metodologia | Ferramenta e papeis ageis | Registrar projeto individual, orientador como avaliador academico e Git como rastreabilidade; nao inventar Product Owner/Scrum Master | Texto preparado |
| Tecnologias | Informar versoes | Python 3.11.9; FastAPI 0.115.6; TensorFlow 2.16.2; MNE 1.8.0; SQLAlchemy 2.0.36; NumPy 1.26.4; SHAP 0.46.0; React 19.2.7; Vite 6.3.5 | Verificado em 13/10/2026 |
| Tecnologias | Ambiente | Ambiente virtual Python `.venv`; npm com lockfile no frontend; Docker nao e requisito atual | Verificado |
| Tecnologias | D3 ou Recharts | Definir Recharts 3.9.0 como biblioteca adotada | Implementado |
| Incrementos | Criterios de aceite | Definir criterios verificaveis para ingestao, IA e interface | Texto preparado |
| Testes | Quantificar dados e split | Informar 77 EDFs/17 grupos no treino; 13 EDFs de quatro grupos na calibracao; avaliacao historica em chb09/chb15/chb18 | Documentado |
| Testes | Bonn e CHB-MIT | Manter somente CHB-MIT no protocolo final; Bonn apenas como base considerada no planejamento | Justificado no documento 30 |
| Testes | Usabilidade | Nao afirmar teste com cinco especialistas; registrar ausencia como limitacao | Nao realizado |
| Testes | Tempo de resposta | Medir em EDF real e informar hardware; nao adotar limite arbitrario de 30 s sem requisito clinico validado | Medicao pendente |
| Implementacao | Definir implantacao | Descrever execucao local do prototipo; implantacao em nuvem nao foi validada | Estado real |
| Implementacao | Dados hospitalares | Substituir por "registros publicos reais da CHB-MIT"; nao foram usados dados de hospital local nem participantes recrutados | Obrigatorio |
| Implementacao | CEP | Explicar que o experimento usa base publica desidentificada; eventual coleta local exigiria avaliacao etica institucional | Texto metodologico |
| Requisitos | Como foram levantados | Revisao bibliografica, analise documental do EDF, benchmarking conceitual e refinamento com orientacao | Sem entrevista clinica formal |
| Requisitos | Ferramentas comparadas | So citar ferramentas efetivamente analisadas e documentadas; levantamento nominal ainda precisa de evidencia | Pendente |
| LGPD | Medidas concretas | Informar JWT, hash de senha e controle de acesso existente; declarar ausencia de consentimento, politica de retencao e auditoria completa | Parcial |
| Casos de uso | Importar exame como `include` | Tornar exame previamente importado uma pre-condicao de "Solicitar Analise Preditiva" | Corrigir diagrama e quadro |
| Casos de uso | Extrair metricas como acao medica | Tratar como etapa interna incluida pelo sistema, nao caso de uso iniciado pelo medico | Corrigir diagrama |
| Casos de uso | Referencia errada ao Quadro 7 | Corrigir numeracao e referencia cruzada apos reorganizar os quadros | Obrigatorio |
| Laudo | Rascunho, edicao e estrutura | O laudo atual e texto livre e torna-se imutavel apos emissao; nao existe rascunho persistente nem campos estruturados | Descrever estado real |
| Modelo de dados | Entidade Laudo e logs | O texto do laudo esta em `Exame`; `PredicaoIA` e entidade separada; nao ha entidade de auditoria | Atualizar MER ou implementar futuramente |
| Login | Recuperacao de senha | A interface e a rota existem, mas o envio e simulado | Nao apresentar como recuperacao operacional |
| Formatacao | Paragrafos, alinhamento e espacamento | Dividir paragrafos longos; justificar corpo; espacamento 1,5; citações longas em bloco ABNT | Aplicar no arquivo editavel |
| Figuras | Aumentar imagens | Substituir wireframes por capturas legiveis da aplicacao final | Pendente |
| Cronograma | Quadros 8 e 9 | Corrigir a remissao textual e atualizar atividades realizadas | Obrigatorio |
| Referencias | Itens incompletos | Revisar autoria, titulo, veiculo, ano, URL e data de acesso; remover fontes inadequadas | Obrigatorio |
| Fechamento | Consideracoes parciais/finais | No TCC II, substituir por Resultados, Discussao, Limitacoes e Conclusao | Estrutura preparada |

## Redacao revisada para a introducao

Embora modelos de aprendizado profundo tenham demonstrado potencial para a analise automatizada de sinais eletroencefalograficos, sua incorporacao ao fluxo de trabalho clinico ainda enfrenta barreiras tecnicas e operacionais. Entre elas, destacam-se a leitura manual de registros extensos, a fragmentacao entre ferramentas de armazenamento e visualizacao, a limitada interoperabilidade de solucoes proprietarias e a dificuldade de interpretar os fatores que influenciam a predicao de redes neurais profundas. Nesse contexto, um classificador isolado nao resolve integralmente o problema, pois o especialista tambem necessita importar o exame, navegar pelos canais, localizar temporalmente os eventos suspeitos, examinar a justificativa fornecida pelo modelo e registrar seu proprio parecer.

Diante dessa lacuna, este trabalho desenvolveu um prototipo funcional de plataforma web para apoio a triagem e a analise de exames de EEG. A solucao centraliza a ingestao de arquivos EDF, a visualizacao multicanal, a inferencia temporal por CNN-BiLSTM, a apresentacao de trechos suspeitos e a explicabilidade por SHAP. O sistema nao substitui o julgamento do especialista nem produz diagnostico autonomo; sua finalidade consiste em oferecer uma segunda opiniao computacional, com potencial para apoiar a produtividade e direcionar a revisao medica para regioes relevantes do exame.

Por se tratar de uma aplicacao web voltada a informacoes de saude, aspectos de acessibilidade, privacidade, rastreabilidade e seguranca devem integrar sua evolucao. No Brasil, a eventual disponibilizacao do produto com finalidade medica deve considerar o enquadramento regulatorio aplicavel a Software como Dispositivo Medico, disciplinado, entre outras normas, pela Resolucao da Diretoria Colegiada RDC nº 657, de 24 de marco de 2022, da Agencia Nacional de Vigilancia Sanitaria. O prototipo academico desenvolvido neste trabalho nao foi submetido a regularizacao sanitaria e nao deve ser apresentado como produto autorizado para uso clinico.

## Objetivo geral revisado

Desenvolver e avaliar um prototipo funcional de plataforma web para apoio a triagem e a analise de exames de Eletroencefalograma, integrando uma arquitetura hibrida CNN-BiLSTM e tecnicas de Inteligencia Artificial Explicavel para localizar temporalmente eventos suspeitos e fornecer informacoes de apoio a revisao do especialista.

## Objetivos especificos revisados

- levantar e documentar requisitos funcionais e nao funcionais relacionados a ingestao, processamento, visualizacao, seguranca e interpretabilidade de EEG;
- implementar uma interface web responsiva para importacao de EDF, navegacao temporal e selecao de canais;
- estruturar um pipeline de preprocessamento e janelamento aplicado depois da separacao dos pacientes entre treino e calibracao;
- extrair caracteristicas tempo-frequencia por canal, Entropia de Shannon e medidas de Dinamica Simbolica;
- treinar uma CNN-BiLSTM capaz de produzir probabilidade ictal ao longo do tempo;
- integrar o metodo SHAP para representar a relevancia das features por canal e por intervalo temporal;
- avaliar o modelo por EDF, evento e localizacao temporal, incluindo precisao, sensibilidade, F1, IoU e falsos alarmes por hora;
- verificar o funcionamento da aplicacao por testes automatizados, build do frontend e processamento de arquivos EDF reais da base CHB-MIT.

## Redacao revisada para levantamento de requisitos

O levantamento de requisitos foi conduzido por meio de revisao bibliografica sobre analise computacional de EEG, analise documental do formato EDF, estudo do fluxo proposto no projeto de iniciacao tecnologica anterior e refinamentos sucessivos realizados durante a orientacao academica. Nao foi conduzida entrevista estruturada nem validacao formal com neurologistas. Por esse motivo, os requisitos clinicos da versao atual devem ser compreendidos como requisitos de um prototipo academico, cuja validacao com profissionais da area permanece como etapa futura.

## Redacao revisada para desenvolvimento incremental

O desenvolvimento foi organizado em ciclos incrementais individuais, com rastreabilidade das alteracoes por meio do Git. O primeiro incremento foi considerado concluido quando a API passou a receber, validar, armazenar e ler arquivos EDF. O segundo incremento teve como criterio de aceite a geracao reproduzivel de features, o treinamento do modelo e a emissao de probabilidades temporais. O terceiro incremento foi concluido com a integracao entre backend e frontend, incluindo visualizacao multicanal, selecao de canais, trechos suspeitos, SHAP e emissao do laudo. Os ciclos de inteligencia artificial permaneceram iterativos, pois cada arquitetura ou estrategia de amostragem foi comparada com uma referencia controlada antes de ser promovida ou rejeitada.

## Redacao revisada para persistencia e laudo

O PostgreSQL armazena os metadados cadastrais e clinicos, enquanto os arquivos EDF, os artefatos dos modelos e as imagens geradas sao mantidos no sistema de arquivos e referenciados pelo banco. Nao foi empregado banco vetorial, pgvector ou persistencia clinica em formatos `.npy` ou `.h5`. O laudo medico e armazenado como texto livre associado ao exame e torna-se imutavel depois de sua emissao. A versao atual nao oferece rascunho persistente, campos clinicos estruturados ou trilha completa de auditoria, funcionalidades recomendadas para evolucao futura.

## Redacao revisada para seguranca, privacidade e regulacao

O prototipo implementa autenticacao por token JWT, armazenamento de senhas por hash e restricao das rotas clinicas a usuarios autenticados. Entretanto, essas medidas nao caracterizam conformidade integral com a Lei Geral de Protecao de Dados Pessoais. A versao atual nao implementa gestao de consentimento ou base legal, politica automatizada de retencao e descarte, registro completo de auditoria, criptografia especifica dos EDF em repouso ou recuperacao real de senha por e-mail. Esses controles devem anteceder qualquer implantacao com dados identificaveis de pacientes.

## Ajustes nos casos de uso

No caso de uso **Solicitar Analise Preditiva**, a existencia de um exame previamente importado deve constar como pre-condicao. A extracao de metricas deve ser representada como operacao interna obrigatoria do pipeline, e nao como uma acao iniciada diretamente pelo medico. A recuperacao de senha pode permanecer no diagrama apenas se identificada como simulada no prototipo ou como funcionalidade futura.

## Referencias regulatorias a acrescentar

- BRASIL. Agencia Nacional de Vigilancia Sanitaria. Resolucao da Diretoria Colegiada - RDC nº 657, de 24 de marco de 2022. Dispoe sobre a regularizacao de software como dispositivo medico. Brasilia, DF: Anvisa, 2022.
- BRASIL. Agencia Nacional de Vigilancia Sanitaria. Software como dispositivo medico: perguntas e respostas. Brasilia, DF: Anvisa, 2022.
- BRASIL. Agencia Nacional de Vigilancia Sanitaria. Guia nº 38/2020: principios e praticas de ciberseguranca em dispositivos medicos. Brasilia, DF: Anvisa, 2020.
- BRASIL. Governo Digital. Modelo de Acessibilidade em Governo Eletronico - eMAG, versao 3.1. Brasilia, DF: Governo Federal.

As referencias devem ser ajustadas ao manual institucional adotado pelo IFG, com URL e data de acesso verificadas no momento da edicao final.

## Pendencia documental

Os PDFs permitem ler o texto e os comentarios, mas nao constituem fonte adequada para uma revisao completa mantendo estilos, sumario, figuras e referencias automaticas. Para aplicar estas alteracoes diretamente ao TCC, e necessario obter o arquivo editavel original em `.docx`, `.odt` ou exportacao do Google Docs.
