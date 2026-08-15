$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
$npmFallback = Join-Path ([Environment]::GetFolderPath('ProgramFiles')) 'nodejs\npm.cmd'
$npm = if ($npmCommand) { $npmCommand.Source } else { $npmFallback }

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Python virtual environment is missing. Run: python -m venv .venv'
}
if (-not (Test-Path -LiteralPath $npm)) {
    throw 'Node.js is missing. Install Node.js LTS first.'
}
$backend = Start-Process -FilePath $python `
    -ArgumentList '-m', 'uvicorn', 'backend.app:app', '--host', '127.0.0.1', '--port', '8000' `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru

try {
    & $npm --prefix (Join-Path $root 'frontend') run dev
}
finally {
    Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
}
