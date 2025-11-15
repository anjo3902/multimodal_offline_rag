# Kill any Python server processes (stop background server)
taskkill /F /IM python.exe 2>$null | Out-Null
Write-Host "Stopped Python processes (if any)" -ForegroundColor Yellow
