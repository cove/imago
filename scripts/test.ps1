$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$scripts = @(
  (Join-Path $repoRoot "cast\scripts\test.ps1"),
  (Join-Path $repoRoot "photoalbums\scripts\test.ps1"),
  (Join-Path $repoRoot "vhs\scripts\test.ps1")
)

foreach ($script in $scripts) {
  Write-Host "==> Running $script"
  & $script @args
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}
