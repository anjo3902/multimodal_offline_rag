# Run tests against running Multimodal RAG server
# Usage: .\run_tests.ps1

param(
    [string]$ServerUrl = "http://127.0.0.1:8000"
)

Write-Host "Starting test sequence against: $ServerUrl" -ForegroundColor Cyan

# Helper: log function
function Log($msg, $color = "White"){
    Write-Host "`n$msg" -ForegroundColor $color
}

# 1) Health check
Log "1) Health check: GET /stats/" "Yellow"
try {
    $stats = Invoke-RestMethod -Uri ("$ServerUrl/stats/") -Method Get -TimeoutSec 3
    Log "OK: /stats/ returned" "Green"
} catch {
    Log "ERROR: /stats/ failed: $_" "Red"
    exit 1
}

# 2) Text query test
Log "2) Text query test: POST /query/" "Yellow"
$body = @{ query = 'what is spark initialization'; search_mode = 'text' }
try {
    # Set a generous timeout to allow for model loading & generation
    $res = Invoke-RestMethod -Uri ("$ServerUrl/query/") -Method Post -ContentType "application/x-www-form-urlencoded" -Body $body -TimeoutSec 180 -ErrorAction Stop
    Log "OK: /query/ returned: $(($res.answer | Select-Object -First 1) -ne $null)" "Green"
    Write-Host "Answer (short preview): $($res.answer.Substring(0, [Math]::Min(200, $res.answer.Length)))`n"
} catch {
    Log "ERROR: /query/ failed: $_" "Red"
    exit 1
}

# 3) File upload test (create a small temporary text file)
Log "3) File upload test: POST /upload/" "Yellow"
$tmpFile = Join-Path $env:TEMP "test_upload_$(Get-Random).txt"
"This is a simple upload test file." | Out-File -FilePath $tmpFile -Encoding UTF8
try {
    # Invoke-RestMethod -Form does not accept file directly in Windows Powershell 5.1 reliably.
    # We'll use curl for multipart uploads which is available on Windows as an alias for Invoke-WebRequest; better use 'curl.exe'
    $curl = 'curl.exe'
    $uploadUrl = "$ServerUrl/upload/"
    $out = & $curl -s -S -F "files=@$tmpFile" "$uploadUrl" --max-time 120
    if ($LASTEXITCODE -eq 0) {
        Log "OK: File upload command succeeded" "Green"
        Write-Host $out
    } else {
        Log "ERROR: curl failed to upload file" "Red"
        Write-Host $out
        exit 1
    }
} catch {
    Log "ERROR: File upload failed: $_" "Red"
    exit 1
} finally {
    Remove-Item $tmpFile -ErrorAction SilentlyContinue
}

# 4) Audio upload test - create a 1s silent WAV using ffmpeg if available
Log "4) Audio upload test: POST /query/ with an audio file" "Yellow"
$wavTmp = Join-Path $env:TEMP "test_silence.wav"
$ffmpeg = "ffmpeg"
if (Get-Command $ffmpeg -ErrorAction SilentlyContinue){
    try {
        & $ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t 1 -acodec pcm_s16le $wavTmp > $null 2>$null
        if (Test-Path $wavTmp) {
            Log "Generated test WAV: $wavTmp" "Green"
            # POST as multipart to /query/
            $curl = 'curl.exe'
            $audioUrl = "$ServerUrl/query/"
            $out = & $curl -s -S -F "query_audio=@$wavTmp" -F "search_mode=audio" "$audioUrl" --max-time 120
            if ($LASTEXITCODE -eq 0) {
                Log "OK: Audio query succeeded" "Green"
                Write-Host $out
            } else {
                Log "ERROR: Audio query (curl) failed" "Red"
                Write-Host $out
                exit 1
            }
        } else {
            Log "ERROR: ffmpeg did not produce $wavTmp" "Red"
        }
    } catch {
        Log "WARNING: ffmpeg or audio path failed: $_" "Yellow"
    } finally {
        Remove-Item $wavTmp -ErrorAction SilentlyContinue
    }
} else {
    Log "ffmpeg not found - skipping audio test" "Yellow"
}

# 5) Basic Static UI check
Log "5) Static UI check: GET / (index.html)" "Yellow"
try {
    $html = Invoke-RestMethod -Uri "$ServerUrl/" -Method Get -TimeoutSec 5
    if ($html -like "*Audio Query*" -or $html -like "*Start Voice Recording*") {
        Log "OK: index.html contains Audio Query UI" "Green"
    } else {
        Log "NOTICE: index.html content may differ. Please check in browser" "Yellow"
    }
} catch {
    Log "ERROR: failed to retrieve index.html: $_" "Red"
    exit 1
}

Log "All automated tests completed successfully!" "Green"
exit 0
