function Initialize-Install {
  param([string]$RepositoryRoot, [string]$ProductName)
  $script:InstallProduct = $ProductName
  $script:InstallState = Join-Path $RepositoryRoot ".setup"
  $script:InstallLog = Join-Path $script:InstallState "install.log"
  $script:InstallLock = $null
  New-Item -ItemType Directory -Force -Path $script:InstallState | Out-Null
  if ((Test-Path -LiteralPath $script:InstallLog) -and (Get-Item -LiteralPath $script:InstallLog).Length -gt 1MB) {
    Move-Item -LiteralPath $script:InstallLog -Destination "$script:InstallLog.1" -Force
  }
  Add-InstallLog "start pid=$PID platform=$([Environment]::OSVersion.Platform)"
}

function Add-InstallLog {
  param([string]$Message)
  if (-not $script:InstallLog) { return }
  $stamp = [DateTimeOffset]::Now.ToString("o")
  Add-Content -LiteralPath $script:InstallLog -Value "$stamp $Message" -Encoding UTF8
}

function Enter-InstallLock {
  param([int]$TimeoutSeconds = 60)
  if ($script:InstallLock) { return }
  $path = Join-Path $script:InstallState "install.lock"
  $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
  while ([DateTime]::UtcNow -lt $deadline) {
    try {
      $stream = [IO.File]::Open($path, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
      $stream.SetLength(0)
      $bytes = [Text.Encoding]::UTF8.GetBytes("pid=$PID`nstarted=$([DateTimeOffset]::Now.ToString('o'))`n")
      $stream.Write($bytes, 0, $bytes.Length)
      $stream.Flush()
      $script:InstallLock = $stream
      Add-InstallLog "lock acquired"
      return
    } catch [IO.IOException] {
      Start-Sleep -Milliseconds 250
    }
  }
  throw "Another $script:InstallProduct installation is still running. Wait for it to finish, then retry. Lock: $path"
}

function Exit-InstallLock {
  if (-not $script:InstallLock) { return }
  $script:InstallLock.Dispose()
  $script:InstallLock = $null
  Remove-Item -LiteralPath (Join-Path $script:InstallState "install.lock") -Force -ErrorAction SilentlyContinue
  Add-InstallLog "lock released"
}

function Write-InstallFailure {
  param($ErrorRecord)
  $message = if ($ErrorRecord.Exception) { $ErrorRecord.Exception.Message } else { [string]$ErrorRecord }
  Add-InstallLog "failure: $message"
  Write-Host "Setup log: $script:InstallLog" -ForegroundColor Yellow
}

function Test-InstallTransientError {
  param($ErrorRecord)
  $message = if ($ErrorRecord.Exception) { $ErrorRecord.Exception.ToString() } else { [string]$ErrorRecord }
  if ($message -match '(?i)checksum|hash mismatch|access denied|permission|unauthorized|forbidden|not found|404|unsupported|invalid argument|license') { return $false }
  return $message -match '(?i)timed? out|temporar|connection|name resolution|network|reset by peer|429|408|500|502|503|504|service unavailable|gateway'
}

function Invoke-InstallRetry {
  param([string]$Label, [scriptblock]$Operation, [int]$Attempts = 3)
  for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
    try {
      & $Operation
      return
    } catch {
      if ($attempt -ge $Attempts -or -not (Test-InstallTransientError $_)) { throw }
      $delay = [Math]::Min(4, [Math]::Pow(2, $attempt - 1))
      Add-InstallLog "$Label transient failure attempt=$attempt retry_in=${delay}s"
      Start-Sleep -Seconds $delay
    }
  }
}

function Save-InstallDownload {
  param([string]$Url, [string]$Destination, [string]$Label = "download")
  $parent = Split-Path -Parent $Destination
  if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
  $partial = "$Destination.partial"
  Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
  $methods = @()
  $methods += { Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $partial }
  if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) { $methods += { Start-BitsTransfer -Source $Url -Destination $partial } }
  if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
    $methods += { & curl.exe --fail --location --silent --show-error --output $partial $Url; if ($LASTEXITCODE -ne 0) { throw "curl download failed with exit $LASTEXITCODE" } }
  }
  $lastError = $null
  for ($attempt = 0; $attempt -lt 3; $attempt++) {
    $method = $methods[$attempt % $methods.Count]
    try {
      & $method
      if (-not (Test-Path -LiteralPath $partial) -or (Get-Item -LiteralPath $partial).Length -eq 0) { throw "$Label returned an empty file" }
      Move-Item -LiteralPath $partial -Destination $Destination -Force
      Add-InstallLog "$Label completed"
      return
    } catch {
      $lastError = $_
      Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
      if (-not (Test-InstallTransientError $_)) { throw }
      if ($attempt -lt 2) { Start-Sleep -Seconds ([Math]::Pow(2, $attempt)) }
    }
  }
  throw "$Label failed after 3 bounded attempts: $($lastError.Exception.Message)"
}

function Assert-InstallFreeSpace {
  param([string]$Path, [double]$RequiredGB)
  $full = [IO.Path]::GetFullPath($Path)
  $root = [IO.Path]::GetPathRoot($full)
  $drive = [IO.DriveInfo]::new($root)
  $available = $drive.AvailableFreeSpace / 1GB
  if ($available -lt $RequiredGB) { throw "Insufficient disk space for $script:InstallProduct. Required: ${RequiredGB} GB. Available: $([Math]::Round($available, 1)) GB on $root" }
  Add-InstallLog "disk available_gb=$([Math]::Round($available, 1)) required_gb=$RequiredGB"
}

function Complete-Install {
  Add-InstallLog "bootstrap complete"
  Exit-InstallLock
}
