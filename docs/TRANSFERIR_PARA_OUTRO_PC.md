# Transferir a instalacao completa para outro PC Windows

Este procedimento preserva o codigo, o modelo v43, `dataset_amostra`, os EDFs
enviados pela interface, as imagens SHAP e o banco SQLite com pacientes, exames,
predicoes e textos dos laudos. O cache de features nao e copiado porque pode ser
regenerado. O pacote pode ultrapassar 7 GB.

**O pacote contem dados clinicos e credenciais em hash. Use apenas midia confiavel,
preferencialmente criptografada. Nunca envie o pacote ao GitHub.**

## No PC atual

1. Feche o backend e pare uploads ou analises em andamento.
2. Confira o tamanho previsto sem copiar nada:

```powershell
.venv\Scripts\python.exe scripts\portable_installation.py export --destination "E:\EEG-XAI-TRANSFERENCIA" --dry-run
```

3. Escolha uma pasta vazia com espaco suficiente e execute na raiz do projeto:

```powershell
.venv\Scripts\python.exe scripts\portable_installation.py export --destination "E:\EEG-XAI-TRANSFERENCIA"
```

O destino deve estar vazio e fora da pasta do projeto. O comando cria uma copia
do codigo versionado, dos dados locais e um snapshot consistente do SQLite.
Se faltar algum artefato, ele para com erro. Os arquivos `.part` e o cache de
features nao sao incluidos.

Para enviar pela nuvem, crie um ZIP AES-256 depois da exportacao. Instale a
dependencia opcional apenas no PC de origem:

```powershell
.venv\Scripts\python.exe -m pip install pyzipper==0.4.0
.venv\Scripts\python.exe scripts\create_transfer_archive.py --source "E:\EEG-XAI-TRANSFERENCIA" --output "E:\EEG-XAI-TRANSFERENCIA.zip" --password-file "E:\EEG-XAI-TRANSFERENCIA.senha.txt"
```

O script verifica o ZIP inteiro e salva a senha e o SHA-256 em um arquivo
separado. Envie apenas o ZIP; guarde ou comunique a senha por outro meio. No
outro PC, extraia com 7-Zip, que suporta ZIP com criptografia AES. Confira se
ha espaco livre suficiente na conta de armazenamento antes do envio.

## No PC novo

1. Instale Python 3.11 (com `py`), Node.js 20 e npm.
2. Copie ou extraia a pasta `EEG-XAI-TRANSFERENCIA` para um local permanente no PC novo.
   Nao execute diretamente da unidade removivel; os caminhos da aplicacao
   serao configurados para a pasta onde o pacote estiver.
3. Na pasta copiada, rode:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_new_pc.ps1
```

Esse comando cria `.venv`, instala backend e frontend, restaura o banco,
atualiza os caminhos dos EDFs/SHAP, cria um `.env` com chave JWT nova e
confere os artefatos do modelo contra o manifesto SHA-256.

4. Inicie em dois terminais, na raiz do projeto:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

```powershell
cd frontend
npm run dev
```

A interface abre em `http://localhost:5173`. A API usa a porta 8000.
O banco e o `.env` gerados nao sao sobrescritos por uma segunda configuracao.
Se mover a pasta depois de configurar, os caminhos armazenados no banco deixam
de apontar para os EDFs; mantenha-a no local definitivo.

O pacote nao retreina a IA nem baixa EDFs da internet. Ele copia os arquivos
presentes neste PC no momento da exportacao.
