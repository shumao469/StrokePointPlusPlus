param(
    [int]$Port = 8851,
    [string]$Distro = 'Ubuntu-22.04',
    [switch]$EnableLocalUploads
)

$ErrorActionPreference = 'Stop'
$releaseRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Choose an unprivileged local port.' }
$occupied = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -eq $Port }
if ($occupied) { throw "Port $Port is occupied; no existing process was stopped." }
if ($releaseRoot -notmatch '^([A-Za-z]):\\') { throw 'This launcher expects a local Windows drive path.' }
$releaseDrive = $Matches[1].ToLowerInvariant()
$wslRoot = '/mnt/' + $releaseDrive + '/' + $releaseRoot.Substring(3).Replace('\','/')
$wslListening = & wsl.exe -d $Distro -- ss -H -ltn "sport = :$Port"
if ($LASTEXITCODE -ne 0) { throw 'Unable to verify the WSL listening port.' }
if ($wslListening) { throw "WSL port $Port is occupied; no existing service was stopped." }
$temporaryFs = (& wsl.exe -d $Distro -- stat -f -c '%T' /dev/shm).Trim()
if ($LASTEXITCODE -ne 0 -or $temporaryFs -ne 'tmpfs') { throw 'A verified RAM-backed /dev/shm is required for framework temporary upload buffering.' }
$python = "$wslRoot/.venv-app/bin/python"
& wsl.exe -d $Distro -- test -x $python
if ($LASTEXITCODE -ne 0) { throw 'Create the isolated .venv-app environment and install requirements-app.txt first.' }
$runtime = Join-Path $releaseRoot '.runtime-app'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$enabled = if ($EnableLocalUploads) { '1' } else { '0' }
$arguments = @('-d', $Distro, '--cd', $wslRoot, '--', 'env',
    'PYTHONDONTWRITEBYTECODE=1', 'TMPDIR=/dev/shm', "STROKEPOINT_ENABLE_LOCAL_UPLOAD=$enabled",
    $python, '-m', 'streamlit', 'run', 'apps/streamlit_app.py',
    '--server.address', '127.0.0.1', '--server.port', "$Port",
    '--server.headless', 'true', '--server.maxUploadSize', '8', '--server.fileWatcherType', 'none',
    '--browser.gatherUsageStats', 'false')
$process = Start-Process -FilePath 'wsl.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runtime 'stdout.log') `
    -RedirectStandardError (Join-Path $runtime 'stderr.log')
$localUrl = "http://127.0.0.1:$Port/"
$healthUrl = "http://127.0.0.1:$Port/_stcore/health"
$healthy = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -eq 200 -and $response.Content.Trim() -eq 'ok') { $healthy = $true; break }
    } catch { }
    Start-Sleep -Milliseconds 500
}
$receipt = [ordered]@{
    state = $(if ($healthy) { 'LOCAL_HEALTH_VERIFIED' } else { 'LOCAL_START_NEEDS_DIAGNOSIS' })
    url = $localUrl
    health_url = $healthUrl
    health_verified = $healthy
    windows_wsl_process_id = $process.Id
    started_at = (Get-Date).ToString('o')
    bind_address = '127.0.0.1'
    upload_feature_enabled = [bool]$EnableLocalUploads
    uploads_still_require_opt_in = $true
    framework_temp_directory = '/dev/shm'
    framework_temp_filesystem = $temporaryFs
    new_inference_or_training = $false
    public_deployment_created = $false
}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtime 'local_launch_receipt.json') -Encoding UTF8
$receipt | ConvertTo-Json
if (-not $healthy) { throw "Health was not verified; inspect only $runtime logs. No existing service was stopped." }
