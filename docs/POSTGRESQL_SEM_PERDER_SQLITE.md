# PostgreSQL sem perder o SQLite

O aplicativo aceita `DATABASE_URL=postgresql+asyncpg://...`. O SQLite atual nao
precisa ser apagado: a migracao copia os registros e nao altera o arquivo de
origem nem a configuracao `.env`.

1. Inicie o PostgreSQL e crie **um banco vazio** chamado `eeg_xai`. Use um
   usuario com permissao para criar tabelas e inserir registros. Nao use um
   banco que ja contenha prontuarios.
2. Mantenha o `DATABASE_URL` atual no `.env` apontando para o SQLite. No
   arquivo local `.env.postgres.local` (ignorado pelo Git), coloque:

   ```env
   EEG_XAI_POSTGRES_URL=postgresql+asyncpg://USUARIO:SENHA@127.0.0.1:5433/EEG
   ```

   Depois execute:

   ```powershell
   .venv\Scripts\python.exe scripts\migrate_sqlite_to_postgres.py
   ```

   Se a senha tiver caracteres especiais como `@`, `:`, `/` ou `#`, codifique-os
   como URL antes de inseri-la. Nao compartilhe o arquivo nem o inclua no Git.

3. O comando cria as tabelas no PostgreSQL, copia usuarios, pacientes, exames e
   predicoes, confere as contagens e ajusta as sequencias de IDs. Ele recusa
   destino com dados. Arquivos EDF, modelos e mapas SHAP continuam no disco,
   com os mesmos caminhos; confirme que esses arquivos existem no PC de destino.
4. Somente depois de validar cadastro, exames, predicoes e laudos na instancia
   PostgreSQL, altere `DATABASE_URL` no `.env` para a URL de destino e reinicie
   o backend. Guarde o arquivo SQLite como copia de seguranca; nao o exclua.

O exportador `scripts/portable_installation.py` continua especifico para
SQLite. Para transferir uma instalacao PostgreSQL a outro PC, use as ferramentas
de backup do proprio PostgreSQL; nao use o exportador SQLite nessa instalacao.

**Estado em 14/10/2026:** a migracao para o banco `EEG` no PostgreSQL 16 foi
executada e conferida: 1 usuario, 1 paciente, 14 exames e 14 predicoes foram
comparados campo a campo com o SQLite. O teste de inicializacao da API e de
consulta ORM com `DATABASE_URL` apontando para PostgreSQL passou. O arquivo
SQLite original e o backup em `transfer_data/` foram preservados, assim como
`.env.sqlite.backup`. A instancia final usa a porta 5433.

Para conferir novamente sem alterar dados:

```powershell
.venv\Scripts\python.exe scripts\migrate_sqlite_to_postgres.py --verify-only
.venv\Scripts\python.exe scripts\verify_postgres_app.py
```
