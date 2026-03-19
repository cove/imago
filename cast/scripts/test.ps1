$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $repoRoot
try {
  & uv run pytest "cast/tests" @args
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
