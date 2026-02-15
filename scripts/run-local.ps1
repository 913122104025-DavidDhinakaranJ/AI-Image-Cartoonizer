$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $repoRoot "backend"
$frontendDir = Join-Path $repoRoot "frontend"
$pythonExe = Join-Path $backendDir ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Backend virtualenv not found. Run scripts/setup.ps1 first."
}

# Prevent stale dev servers from causing CORS/network confusion.
$oldBackend = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match "uvicorn app.main:app" }
if ($oldBackend) {
    $oldBackend | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
}

$oldFrontend = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -eq "node.exe" -and $_.CommandLine -match "vite" }
if ($oldFrontend) {
    $oldFrontend | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
}

$backendCmd = "Set-Location '$backendDir'; & '$pythonExe' -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
$frontendCmd = "Set-Location '$frontendDir'; npm run dev -- --host 0.0.0.0 --port 5173"

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd | Out-Null
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd | Out-Null

Write-Host "Backend:  http://localhost:8000"
Write-Host "Frontend: http://localhost:5173"
