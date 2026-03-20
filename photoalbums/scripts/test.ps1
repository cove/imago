$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $repoRoot
try {
  & uv run pytest "photoalbums/tests" @args
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
  & uv run python scripts/check_skylos.py photoalbums
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
