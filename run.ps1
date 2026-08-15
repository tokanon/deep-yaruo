$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$frontend = Join-Path $root 'frontend\dist\index.html'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Python virtual environment is missing. See README.md for setup.'
}
if (-not (Test-Path -LiteralPath $frontend)) {
    throw 'Frontend build is missing. Run npm run build in the frontend directory.'
}

Write-Host 'Yaruo AA Studio: http://127.0.0.1:8000/'
& $python -m uvicorn backend.app:app --app-dir $root --host 127.0.0.1 --port 8000
