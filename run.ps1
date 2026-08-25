param(
  [ValidateSet("run", "doctor", "repair", "docker", "stop", "logs")]
  [string]$Action = "run",
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$UvVersion = "0.12.5"
$Url = "http://127.0.0.1:8000"

function Invoke-Retry([string]$Label, [scriptblock]$Operation) {
  for ($attempt = 1; $attempt -le 3; $attempt++) {
    try { & $Operation; return } catch {
      if ($attempt -eq 3) { throw "$Label failed after 3 attempts: $($_.Exception.Message)" }
      Start-Sleep -Seconds ([math]::Pow(2, $attempt - 1))
    }
  }
}
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
    Invoke-Retry "uv download" { Invoke-WebRequest -UseBasicParsing -Uri "https://astral.sh/uv/$UvVersion/install.ps1" -OutFile $installer }
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
  docker compose up --detach --build
  if (-not (Wait-Ready)) { docker compose logs; throw "SegCraft did not become healthy at $Url." }
  Write-Host "SegCraft is ready at $Url" -ForegroundColor Green
  if (-not $NoBrowser) { Start-Process $Url }
  exit 0
}

$uv = if ($Action -eq "doctor") { Resolve-Uv } else { Ensure-Uv }
if (-not $uv) { throw "uv is missing. Run .\run.ps1 once." }
if ($Action -eq "doctor") { & $uv run --frozen --no-sync segcraft doctor; exit $LASTEXITCODE }

$syncArgs = @("sync", "--frozen", "--extra", "web")
if ($Action -eq "repair") { $syncArgs += "--reinstall" }
Invoke-Retry "dependency synchronization" { & $uv @syncArgs; if ($LASTEXITCODE -ne 0) { throw "uv sync exited with $LASTEXITCODE" } }
$env:SEGCRAFT_OPEN_BROWSER = if ($NoBrowser) { "0" } else { "1" }
& $uv run --frozen --no-sync segcraft-web
exit $LASTEXITCODE
