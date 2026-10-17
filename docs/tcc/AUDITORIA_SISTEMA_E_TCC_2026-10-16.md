# Auditoria cruzada do EEG-XAI e do TCC II

Data: 16/10/2026. Referencia do repositorio: commit 3eb7abc.

## Atualizacao apos correcoes autorizadas

Os achados abaixo registram o estado auditado. Na rodada seguinte foram corrigidos A01, A02, A04, A05, a perda de extremos de A08 e o pacote E01:

- SHAP usa uma sequencia que contem o pico agregado e um alvo escalar no passo correspondente. O mapa continua sendo explicacao de uma sequencia contribuinte, nao uma decomposicao exata da media de todas as sequencias sobrepostas.
- Verificacao com pesos reais `v43_final_seed42.keras`, entradas sinteticas e Gradient SHAP real gerou PNG e 184 celulas de overlay. Nao equivale a nova avaliacao de desempenho com EDF.
- Indicadores de validade sao persistidos para novas analises; registros antigos precisam ser reanalisados.
- Cada solicitacao retorna UUID de execucao, salvo no JSON da predicao. O polling filtra por esse UUID e o preserva na sessao do navegador para retomar apos recarregar. Resultados anteriores nao encerram a espera. Estados persistentes de falha/fila continuam como pendencia A07.
- A visualizacao retém minimos e maximos por canal em blocos, com tempos reais e grade compartilhada. Foi retirada a segunda reducao uniforme para 600 pontos no frontend. Isso preserva extremos, mas a visao geral segue reduzida e nao reproduz toda a morfologia do sinal bruto.
- PNG sequencial recebe identificador da execucao para evitar sobrescrita entre novas analises.
- ZIP regenerado: `main.tex`, `conteudo.tex` e `pretextuais.tex` conferidos como identicos as fontes atuais.
- Validacao: 46 testes focados aprovados; build TypeScript/Vite aprovado. O travamento anterior foi localizado em `socketpair/asyncio` dentro do isolamento do Windows; a mesma rodada terminou fora dele em 16,74 s. Houve apenas aviso de restauracao do otimizador em modelo de teste.
- Nao houve retreino, alteracao de limiar cientifico, migracao de esquema ou alteracao do texto LaTeX. As correcoes ainda nao foram commitadas/enviadas nesta rodada.

## Escopo e limites

Revisao do LaTeX atual (`overleaf_eeg_xai/main.tex`, `conteudo.tex`, `pretextuais.tex`), backend, componentes principais do frontend, extracao/inferencia/XAI, avaliadores e artefatos finais da v43. As evidencias abaixo sao de leitura de codigo e artefatos, verificacoes locais e compilacao do frontend. Nao houve retreino, alteracao do modelo, migracao do banco nem modificacao do texto do TCC nesta auditoria.

Esta revisao nao certifica ausencia de outros bugs, conformidade normativa integral ou validacao clinica. Nao foi realizada uma nova sessao completa de navegador com EDF real, nem conferida cada referencia bibliografica em sua publicacao original. A paginacao ainda tem pendencias registradas na revisao visual anterior.

## Parecer

Existe um prototipo substancial e resultados experimentais documentados. O principal bloqueio imediato nao e criar outra arquitetura: e garantir que a demonstracao apresente a analise correta, que o SHAP explique a entrada/saida correta, que arquivos estejam protegidos e que o texto descreva exatamente o protocolo e suas limitacoes.

Os numeros principais do TCC coincidem com os artefatos finais: calibracao 15/20 eventos, F1 EDF 95,7%, F1 localizado 90,9%, FA/h 0,55; avaliacao historica 4/9 eventos, F1 EDF 82,4%, F1 localizado 46,2%, FA/h 2,08. Essa concordancia nao demonstra generalizacao externa.

## Nova analise apos as correcoes

| Achado | Estado atual | Observacao |
| --- | --- | --- |
| A01/A02 - mapeamento e alvo do SHAP | Resolvido | Teste real com a v43 gerou PNG e overlay; a explicacao permanece restrita a uma sequencia contribuinte. |
| A04 - resultado antigo encerrar reanalise | Resolvido | A solicitacao e o polling usam UUID de execucao. |
| A05 - indicadores de validade omitidos | Resolvido para novas analises | O default foi alterado para inconclusivo quando o campo nao existir; registros antigos devem ser reanalisados. |
| A08 - perda de extremos no grafico | Resolvido na visao reduzida | O envelope min/max preserva picos, mas nao substitui o sinal integral em alta resolucao. |
| E01 - ZIP Overleaf antigo | Resolvido | Os tres arquivos TeX do ZIP coincidem com as fontes atuais. |
| A03 - storage publico | Pendente critico | EDFs e artefatos continuam servidos por montagem estatica sem autenticacao. |
| A06 - modelo dummy/fallback silencioso | Pendente critico | Uma instalacao incompleta ainda pode produzir uma analise que parece normal. |
| A07 - estado persistente de execucao | Pendente alto | Nao ha fila, falha consultavel nem bloqueio de analises concorrentes. |
| T01/T02 - definicoes e reproducibilidade | Pendente no TCC | Metricas e receita experimental ainda precisam de definicao consolidada. |

Conclusao desta nova rodada: as correcoes retiraram os riscos imediatos de resultado obsoleto, flags contraditorias, pico perdido na visualizacao e SHAP tecnicamente incompativel. O sistema ainda nao deve ser tratado como finalizado enquanto arquivos clinicos permanecerem publicos, o fallback dummy puder parecer uma inferencia valida e a execucao de IA nao possuir estado persistente de sucesso ou falha.

## Achados prioritarios no sistema

### A01 - Alta: SHAP usa indices de dominios diferentes

Evidencia: `app/ai_engine/sequence_inference.py:498`, `:538`, `:566`, `:606`.

Na segmentacao, `indice_sequencia_xai` identifica uma sequencia, enquanto `indice_pico` identifica uma janela apos agregar saidas sobrepostas. Os intervalos de SHAP usam o primeiro; `_xai_input` usa o segundo para indexar `x_scaled`, cujo eixo e de sequencias. Pode explicar outra sequencia ou obter entrada vazia se o indice de janela exceder a quantidade de sequencias. O rotulo do pico agregado tambem precisa corresponder ao alvo efetivamente explicado.

Correcao: selecionar uma janela-alvo e a sequencia que a representa, com mapeamento explicito; usar esse mesmo mapeamento para entrada, saida escalar e intervalos. Aceite: teste com pico tardio e sequencias sobrepostas, verificando valores e tempos, seguido de EDF real.

### A02 - Alta: GradientExplainer recebe saida temporal incompativel

Evidencia: `app/ai_engine/sequence_shap.py:90`; `.venv/Lib/site-packages/shap/explainers/_gradient.py:189`; arquivo Keras v43 termina em `TimeDistributed(Dense(1))`.

A v43 produz tensor [lote, tempo, 1]. O SHAP instalado exige `len(model_output.shape) < 3`, mas o codigo passa diretamente o modelo temporal. O teste de overlay usa `ExplainerFake` e nao verifica essa integracao real. A excecao e capturada pelo pipeline, que ainda salva a predicao e pode informar `xai_method=gradient_shap` mesmo sem explicacao gerada.

Correcao: construir alvo escalar explicitamente definido para a janela/decisao explicada e registrar estado XAI (gerado, desativado, indisponivel, falhou). Aceite: Gradient SHAP real sobre a v43, entrada nao vazia e overlay coerente. Nao basta o teste com mock.

### A03 - Alta: arquivos de storage expostos sem autenticacao

Evidencia: `app/main.py:67` monta todo `settings.storage_root` em `/static` com `StaticFiles`. A autenticacao esta nos routers, nao nessa montagem.

Um cliente sem token que conheca o caminho pode solicitar EDF, imagem SHAP e outros arquivos desse diretorio. O nome UUID dificulta descoberta, mas nao verifica permissao. Mapas sequenciais usam nome previsivel por ID de exame.

Correcao: servir artefatos por rotas autenticadas com verificacao de propriedade do exame. Aceite: sem token e medico diferente nao acessam arquivos; proprietario acessa. Nao foi necessario baixar dados de pacientes para identificar essa montagem publica.

### A04 - Alta: reanalise pode mostrar a predicao antiga como nova

Evidencia: `app/api/routes/exames.py:106-115`, `:322`; `frontend/src/hooks/useDiagnosticoPolling.ts:63`, `:103`.

POST /analise agenda trabalho sem identificador de execucao. GET /diagnostico retorna a ultima predicao existente imediatamente. O polling para ao receber qualquer diagnostico concluido. Ao reanalisar um exame, o resultado antigo pode encerrar a espera antes da nova execucao, inclusive com canais diferentes.

Correcao: identificar cada execucao e consultar seu estado/resultado; preservar historico sem confundir versoes. Aceite: analisar, reanalisar com entrada diferente e garantir que a UI aguarda o novo resultado.

### A05 - Alta: flags de validade desaparecem antes da API

Evidencia: `sequence_inference.py:583-586`; `exame_pipeline.py:292`; `api/routes/exames.py:150-154`.

`cobertura_excessiva`, `resultado_conclusivo` e `resultado_positivo_conclusivo` sao calculados, mas omitidos por `_detalhes_sequenciais`. A rota assume respectivamente false, true e false. O frontend pode receber sinais contraditorios: classificacao textual limitada, flag conclusiva verdadeira e positividade falsa.

Verificacao executada: extracao AST e chamada isolada de `_detalhes_sequenciais` confirmou os tres campos AUSENTES mesmo quando fornecidos como entrada. Correcao: persistir e devolver esses campos sem defaults que convertam desconhecido em conclusivo; testar inferencia -> persistencia -> API.

### A06 - Alta: falta de artefato pode gerar analise com modelo dummy

Evidencia: `exame_pipeline.py:54-66`, `:429`; `config.py` usa development por padrao.

Se o sequencial nao estiver disponivel, ha fallback legado. Se o legado tambem faltar em desenvolvimento, um modelo dummy e criado e utilizado. Em instalacao nova isso pode produzir scores sem treinamento com aparencia de analise normal. Nao afirmo que a execucao atual usa dummy: o .env aponta para v43.

Correcao: modo demonstracao com dummy separado e identificado; bloquear analise cientifica sem artefatos validados. Aceite: remover artefatos apenas em ambiente temporario e receber erro claro, nunca resultado clinico comum.

### A07 - Media/alta: tarefa sem estado de falha e sem controle de concorrencia

Evidencia: `exame_pipeline.py:515-518`, `models/exame.py:13-14`, `api/routes/exames.py:322`.

Falha gera rollback/log, sem registro consultavel de erro; sem predicao, a API continua em 206. `pendente/concluido` representam o laudo, nao os estados da execucao IA. Chamadas repetidas podem iniciar analises simultaneas; reinicio perde BackgroundTasks. O PNG SHAP por exame tambem e sobrescrito por novas execucoes.

Correcao: estado persistido por execucao, exclusao mutua ou rejeicao de duplicata, erro recuperavel na UI, artefatos versionados. Para demonstracao local, nao e obrigatorio introduzir uma plataforma grande de filas.

### A08 - Media: fidelidade do grafico e limitada pela amostragem

Evidencia: `app/services/exame_sinais.py:71-74` usa `linspace` para selecionar ate 1500 pontos por canal, sem preservar extremos. O grafico solicita recortes de ate 120 s.

Em 120 s, 1500 pontos equivalem aproximadamente a 12,5 pontos/s; mesmo um recorte de 30 s tem cerca de 50 pontos/s. Picos entre amostras podem desaparecer e frequencias podem ser representadas incorretamente. Isso afeta a visualizacao, nao prova erro na inferencia, que usa outra entrada.

Correcao: detalhe em resolucao adequada para intervalos curtos e envelope min/max para visao geral; indicar claramente quando o traco e amostrado. Aceite: sinal sintetico com picos conhecidos permanece visivel ao navegar/alterar zoom.

### A09 - Media: upload integral em memoria e sem limite explicito

Evidencia: `app/services/exame_storage.py:26-27` faz `await arquivo.read()` e escrita sincrona de todo o conteudo. Nao ha limite de tamanho nesse caminho.

EDFs grandes ou uploads simultaneos podem consumir muita memoria e bloquear o processo durante escrita. Correcao: streaming em blocos, limite configurado e limpeza de arquivo parcial. A validacao EDF posterior ja existe e deve ser mantida.

### A10 - Media: consistencia do laudo e da analise

O frontend bloqueia nova analise quando o laudo foi emitido, mas a rota POST /analise nao verifica esse estado. O banco nao associa o laudo a uma predicao especifica nem guarda versao/hash do modelo em cada predicao. Novas execucoes podem alterar a analise apresentada ao lado de um laudo dito definitivo.

Evidencia: `api/routes/exames.py` em `salvar_laudo_exame` e `solicitar_analise_ia`; `models/predicao_ia.py`; `_detalhes_sequenciais`.

Correcao: definir regra de reanalise apos laudo e vincular laudo a execucao/modelo/parametros. Verificar tambem corrida entre duas emissoes: a verificacao atual e leitura seguida de escrita, sem atualizacao condicional atomica.

## Divergencias e lacunas no TCC

### T01 - Alta: definir exatamente as metricas

Em `conteudo.tex:142`, FA/h e descrito por hora de sinal analisado. Em `scripts/calibrate_sequence_cnn_lstm.py:500-544`, o denominador e tempo nao ictal. Explicitar essa escolha.

O F1 localizado e calculado sobre rotulos por EDF, exigindo sobreposicao do trecho principal: `:466-480` e `:524`. Nao e F1 de todos os instantes nem F1 por evento. Apresentar definicao, limiares de sobreposicao e exemplo. Isso explica como pode haver F1 localizado de 90,9% e apenas 15/20 eventos encontrados.

O avaliador percorre eventos e candidatos sem pareamento exclusivo um-a-um (`:391-410`). Alarmes fragmentados que sobrepoem o mesmo evento nao sao necessariamente penalizados como duplicatas. Documentar o criterio atual; qualquer mudanca exige nova avaliacao identificada, preservando numeros historicos.

Os intervalos reais do avaliador sao reconstruidos dos rotulos temporais (`intervalos_crise_reais`), portanto conferir e documentar discretizacao frente aos segundos oficiais. Nao confundir precisao de fronteira da grade com precisao da anotacao original.

### T02 - Alta: detalhar receita reproduzivel

O protocolo descreve canais, features, janelas e limiar, mas precisa de quadro consolidado com nomes das 15 features, bandas, normalizacao robusta, camadas/dimensoes, loss, pesos, otimizador, taxa de aprendizado, batch, epocas, selecao de checkpoint, amostragem e manifests.

Confirmado no artefato `v54_v43_final_seed42_training.json`: 77 EDFs de treino, 13 de calibracao, 4648 sequencias de treino, 952 positivas. Ha informacao no repositorio que pode complementar o texto sem inventar experimento.

### T03 - Media: ainda promete deteccao de descargas e explicacao fisiologica

`conteudo.tex:332` inclui descargas epileptiformes; `:361` chama SHAP de justificativa fisiologica. As evidencias apresentadas sao de eventos ictais anotados e atribuicoes do modelo. Falta validacao especifica de descargas interictais; o SHAP nao estabelece mecanismo fisiologico. Ajustar esses casos de uso a classificacao de trechos suspeitos e contribuicao computacional.

Em `:368`, a correspondencia com janela selecionada deve ficar restrita ao alvo efetivamente explicado. Hoje a proposta e a sequencia de pico; selecionar outro candidato na UI nao recalcula SHAP para ele.

### T04 - Media: configuracoes e requisitos precisam refletir o escopo final

O frontend ja identifica o controle de threshold como exclusivo do legado (`Configuracoes.tsx:235-255`). A inferencia sequencial usa calibracao congelada, com `threshold_override=None`. O TCC deve distinguir explicitamente preferencia do medico e ponto operacional da v43. A antiga suspeita de que o slider substitui 0,60 por 0,90 nao corresponde ao codigo atual.

Recuperacao de senha continua simulada (`auth.py`, endpoint recuperar-senha); a resposta promete instrucoes por e-mail, embora nenhum envio exista. O TCC ja reconhece a simulacao, mas a interface deve comunicar essa limitacao. Laudo em texto livre, sem assinatura digital ou campos estruturados, tambem ja esta delimitado no documento.

### T05 - Media: desempenho e usabilidade ainda sem medicao consolidada

O texto teorico diz que arquivos pesados sao processados em segundos (`:53`) e os requisitos falam em fluidez. Nao encontrei neste levantamento uma tabela final que sustente latencia do upload, extracao, inferencia, SHAP, memoria e navegacao por EDF/hardware. Registrar tempos da versao entregue e apresentar como verificacao tecnica; estudo formal com neurologistas permanece limitacao declarada.

### T06 - Media: acabamento editorial e referencias

Persistem quebras e espacos excessivos identificados na auditoria visual anterior. Nao foi feita nova inspeção de todas as paginas nesta rodada. O README do Overleaf diz conter folha de aprovacao, mas `pretextuais.tex` nao a implementa. Confirmar os elementos exigidos pelo instituto antes de acrescentar conteudo ou dados da banca.

Ha URLs com pontuacao final dentro de `\url{...}` e entradas genericas/malformadas, como `governodigitallogin`. A referencia ABOALAYON aponta para catalogo, nao diretamente para o artigo. Fazer verificacao bibliografica contra originais, inclusive titulo/autoria/ano/volume e correspondencia com citacoes. Nao foram conferidas externamente nesta auditoria.

## Entrega e documentacao

### E01 - Alta para entrega: ZIP do Overleaf desatualizado

Comparacao de texto das entradas do ZIP com arquivos atuais: `main.tex`, `conteudo.tex` e `pretextuais.tex` diferem, todos. O commit anterior enviou o ZIP antigo junto das fontes novas. Quem importar o ZIP pode recuperar problemas ja corrigidos. Regenerar pacote e verificar igualdade dos arquivos antes da entrega.

### E02 - Media: fonte de verdade e documentos de estado divergentes

`docs/ia/README.md` ainda descreve v43 como nao promovida em trechos antigos, embora depois aponte o pacote final. `PENDENCIAS_REVISAO.md` descreve outro DOCX/PDF, Arial e 47 paginas, enquanto o projeto atual usa tipografia Times. A auditoria de estilo do Overleaf lista defeitos ja corrigidos sem status atualizado.

Definir o LaTeX atual como fonte final de entrega e identificar documentos historicos. `scripts/export_tcc_to_latex.py` regenera conteudo/pretextuais a partir do DOCX: executar novamente pode apagar revisoes feitas somente no LaTeX. Documentar essa direcao de sincronizacao.

### E03 - Media: clone Git nao equivale a instalacao completa

`.gitignore` exclui `.keras`, `.pkl` e EDFs. O .env.example contem caminhos absolutos da maquina original. Ha scripts/testes de portabilidade, mas Git sozinho nao transporta pesos e dados. A entrega reproduzivel deve identificar pacote adicional, hashes, restauracao do PostgreSQL e caminhos relativos/configurados. A reproducao em segundo computador precisa de registro do que foi realmente executado.

## O que esta coerente e ja foi feito

- Autenticacao e verificacao de proprietario nas rotas clinicas de pacientes/exames.
- Validacao de CPF, telefone e data de nascimento no backend; controles no frontend.
- PostgreSQL suportado na configuracao, modelos e historico de migracao. Nao foi refeita conexao/migracao nesta auditoria.
- Modelo final v43 com 23 canais, 15 features por canal, 8 passos e saida temporal; metadados e configuracao local correspondem.
- Limiar final 0,60, duracao 18 s e gap 8 s no artefato de calibracao.
- Texto ja explica Bonn fora da avaliacao final, k-fold exploratorio, mudanca CNN-LSTM -> CNN-BiLSTM, features espectrais sem Dinamica Simbolica e limitacoes da avaliacao historica.
- Resultados historicos nao sao apresentados como teste clinico independente; a queda de generalizacao esta documentada.
- Interface possui selecao de candidatos, navegacao temporal, ampliacao do grafico e explicacao de que score nao e certeza diagnostica.

## Verificacoes desta rodada

- `npm run build`: aprovado (TypeScript + Vite); aviso de bundle JavaScript maior que 500 kB, sem erro de compilacao.
- Chamada isolada da funcao de serializacao: comprovada a perda das tres flags descritas em A05.
- Inspecao do config.json interno do Keras: saida final temporal `TimeDistributed`, consistente com o conflito descrito em A02.
- Comparacao do ZIP: os tres arquivos TeX principais nao correspondem aos arquivos atuais.
- `pytest -q`: interrompido depois de progresso parcial, sem resumo final.
- Selecao de testes de autenticacao, rotas, pipeline, SHAP e sinais: tambem interrompida sem resultado final. Nao se atribui essa demora especificamente ao TensorFlow sem diagnostico adicional. Nenhuma das duas execucoes conta como suite aprovada.
- Nenhuma alteracao em codigo de produto, pesos ou fonte do TCC; apenas este relatorio foi acrescentado. Nenhum commit/push realizado nesta rodada.

## Ordem de fechamento recomendada

1. Corrigir SHAP real, flags persistidas e identificacao de execucao/reanalise; verificar com EDF real.
2. Proteger arquivos, impedir dummy silencioso e registrar falhas da IA.
3. Vincular laudo/artefatos a execucao e melhorar fidelidade do grafico.
4. Definir metricas e completar receita/metodologia no TCC, mantendo historico de decisoes.
5. Fechar verificacoes automatizadas e roteiro ponta a ponta; registrar versoes, hashes e tempos.
6. Revisar referencias, elementos institucionais, paginacao e sincronizar ZIP/PDF/fontes.

Uma nova arquitetura ou promessa de mais precisao nao e requisito automatico para concluir o TCC. Os defeitos funcionais acima precisam de correcao; limitacoes experimentais devem ser expostas e discutidas com as evidencias existentes.
