$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot
try {
  & uv run pytest "cast/tests" "photoalbums/tests" "vhs/test" @args
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
