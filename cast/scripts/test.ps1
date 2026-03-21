$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $repoRoot
try {
  & uv run pytest "cast/tests" @args
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
  & uv run python scripts/check_skylos.py cast
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
