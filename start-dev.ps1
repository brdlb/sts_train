$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendPython = Join-Path $Root ".venv\Scripts\python.exe"
$FrontendDir = Join-Path $Root "frontend"

if (-not (Test-Path -LiteralPath $BackendPython)) {
    throw "Backend Python was not found at $BackendPython"
}

if (-not (Test-Path -LiteralPath $FrontendDir)) {
    throw "Frontend directory was not found at $FrontendDir"
}

$BackendCommand = @"
Set-Location -LiteralPath '$Root'
& '$BackendPython' -m uvicorn src.perudo.web.main:app --host 127.0.0.1 --port 5565
"@

$FrontendCommand = @"
Set-Location -LiteralPath '$FrontendDir'
npm run dev
"@

function Convert-ToEncodedCommand {
    param([string]$Command)

    return [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Command))
}

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-EncodedCommand", (Convert-ToEncodedCommand $BackendCommand)
)

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-EncodedCommand", (Convert-ToEncodedCommand $FrontendCommand)
)

Write-Host "Backend and frontend terminals started."
Write-Host "Backend:  http://127.0.0.1:5565"
Write-Host "Frontend: check the Vite URL in the frontend terminal, usually http://localhost:5173"
