$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
Push-Location $repoRoot
try {
  & uv run pytest "vhs/test" @args
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
  & uv run python scripts/check_skylos.py vhs
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
