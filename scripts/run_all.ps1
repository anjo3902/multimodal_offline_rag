# Run server and smoke tests (PowerShell)
# Usage: .\run_all.ps1

param(
    [string]$ProjectRoot = "E:\multimodal_rag_free"
)

Set-Location $ProjectRoot

# Kill existing python servers
Write-Host "Stopping any existing Python processes..." -ForegroundColor Yellow
taskkill /F /IM python.exe 2>$null | Out-Null
Start-Sleep -Seconds 1

# Ensure log directory
$logDir = Join-Path $ProjectRoot "logs"
if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$serverLog = Join-Path $logDir "server_out.log"
$serverErrLog = Join-Path $logDir "server_err.log"

# Choose python from venv if available
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) { $python = $venvPython } else { $python = "python" }

# Ensure PYTHONIOENCODING to utf-8 so unicode prints don't crash on Windows console
Set-Item -Path Env:PYTHONIOENCODING -Value "utf-8"

# Start server in background with redirected logs
Write-Host "Starting server with: $python run_server.py" -ForegroundColor Green
Start-Process -FilePath $python -ArgumentList "run_server.py" -WorkingDirectory $ProjectRoot -RedirectStandardOutput $serverLog -RedirectStandardError $serverErrLog -WindowStyle Hidden

# Wait for server to be ready
Write-Host "Waiting for server to start..." -ForegroundColor Yellow
$retries = 30
$started = $false
for ($i = 0; $i -lt $retries; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/stats/" -Method Get -TimeoutSec 2 -ErrorAction Stop
        $started = $true
        break
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

if (-not $started) {
    Write-Host "Server didn't respond after $retries attempts. Check $serverLog" -ForegroundColor Red
    exit 1
}

Write-Host "Server appears to be listening!" -ForegroundColor Green
# Open browser window for convenience
Start-Process "http://127.0.0.1:8000"

# Run automated tests
Write-Host "Running smoke tests..." -ForegroundColor Cyan
$scriptDir = Join-Path $ProjectRoot "scripts"
& "$scriptDir\run_tests.ps1"

Write-Host "Done. Review logs in: $serverLog" -ForegroundColor Green

# Leave server running. Use stop_server.ps1 to terminate.
Write-Host "Use scripts\stop_server.ps1 to stop the server." -ForegroundColor Cyan
