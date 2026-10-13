$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $project

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Instale Python 3.11 com o Python Launcher (py) antes de continuar.'
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw 'Instale Node.js 20 com npm antes de continuar.'
}

& py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Falha ao criar o ambiente Python.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar requirements.txt.' }

Push-Location -LiteralPath (Join-Path $project 'frontend')
try {
    & npm ci
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar o frontend.' }
} finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath '.env') -and -not (Test-Path -LiteralPath 'eeg_xai_dev.db')) {
    & .\.venv\Scripts\python.exe scripts\portable_installation.py configure
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao configurar o pacote transferido.' }
} elseif (-not (Test-Path -LiteralPath '.env') -or -not (Test-Path -LiteralPath 'eeg_xai_dev.db')) {
    throw 'Instalacao parcial: .env e banco precisam existir juntos.'
}

& .\.venv\Scripts\python.exe scripts\verify_release_artifacts.py
if ($LASTEXITCODE -ne 0) { throw 'Artefatos do modelo nao conferem com o manifesto.' }

Write-Host 'Instalacao pronta. Inicie a API e o frontend em terminais separados:'
Write-Host '  .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload'
Write-Host '  cd frontend; npm run dev'
