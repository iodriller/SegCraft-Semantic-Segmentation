param(
  [ValidateSet("run", "doctor", "repair", "docker", "stop", "logs")]
  [string]$Action = "run",
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\scripts\install-utils.ps1
Initialize-Install -RepositoryRoot $PSScriptRoot -ProductName "SegCraft"
trap { Write-InstallFailure $_; Exit-InstallLock; exit 1 }
$UvVersion = "0.12.5"
$Url = "http://127.0.0.1:8000"
function Resolve-Uv {
  $command = Get-Command uv -ErrorAction SilentlyContinue
  foreach ($candidate in @($(if ($command) { $command.Source }), "$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
  }
  return $null
}
function Ensure-Uv {
  $uv = Resolve-Uv
  if ($uv) { return $uv }
  $installer = Join-Path $env:TEMP "segcraft-uv-$UvVersion.ps1"
  try {
    Save-InstallDownload -Url "https://astral.sh/uv/$UvVersion/install.ps1" -Destination $installer -Label "uv download"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer
  } finally { Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue }
  $uv = Resolve-Uv
  if (-not $uv) { throw "uv installed but could not be located." }
  return $uv
}
function Wait-Ready {
  for ($i = 0; $i -lt 120; $i++) {
    try { Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 2 | Out-Null; return $true } catch { Start-Sleep -Milliseconds 500 }
  }
  return $false
}
function Test-Ready { try { Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 2 | Out-Null; return $true } catch { return $false } }

if ($Action -in @("docker", "stop", "logs")) {
  $docker = Get-Command docker -ErrorAction SilentlyContinue
  $engineRunning = $false
  if ($docker) { docker info *> $null; $engineRunning = ($LASTEXITCODE -eq 0) }
  if ($Action -eq "stop" -and -not $engineRunning) { Write-Host "The native server runs in the foreground. Press Ctrl+C in its terminal to stop it."; exit 0 }
  if ($Action -eq "logs" -and -not $engineRunning) { Write-Host "The native server writes logs to its foreground terminal."; exit 0 }
  if (-not $docker) { throw "Docker is not installed." }
  if (-not $engineRunning) { throw "Docker is installed but its engine is not running." }
  if ($Action -eq "stop") { docker compose down; exit $LASTEXITCODE }
  if ($Action -eq "logs") { docker compose logs --follow; exit $LASTEXITCODE }
  Enter-InstallLock
  Assert-InstallFreeSpace -Path $PSScriptRoot -RequiredGB 3
  docker compose up --detach --build
  if ($LASTEXITCODE -ne 0) { throw "Docker Compose failed to start SegCraft." }
  if (-not (Wait-Ready)) { docker compose logs; throw "SegCraft did not become healthy at $Url." }
  Complete-Install
  Write-Host "SegCraft is ready at $Url" -ForegroundColor Green
  if (-not $NoBrowser) { Start-Process $Url }
  exit 0
}

$uv = Resolve-Uv
if ($Action -eq "doctor") {
  if (-not $uv) { throw "uv is missing. Run .\run.ps1 once." }
  & $uv run --frozen --no-sync segcraft doctor
  exit $LASTEXITCODE
}
Enter-InstallLock
Assert-InstallFreeSpace -Path $PSScriptRoot -RequiredGB 3
if (-not $uv) { $uv = Ensure-Uv }
$syncArgs = @("sync", "--frozen", "--extra", "web")
if ($Action -eq "repair") { $syncArgs += "--reinstall" }
Invoke-InstallRetry "dependency synchronization" {
  $output = & $uv @syncArgs 2>&1
  if ($LASTEXITCODE -ne 0) { throw "uv sync failed: $($output -join [Environment]::NewLine)" }
  $output | Write-Host
}
Complete-Install
if (Test-Ready) {
  Write-Host "SegCraft is already running at $Url" -ForegroundColor Green
  if (-not $NoBrowser) { Start-Process $Url }
  exit 0
}
$env:SEGCRAFT_OPEN_BROWSER = if ($NoBrowser) { "0" } else { "1" }
& $uv run --frozen --no-sync segcraft-web
exit $LASTEXITCODE
