$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$py = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $py)) {
  $py = "python"
}

& $py -m pytest (Join-Path $repoRoot "vhs\\test") @args
exit $LASTEXITCODE
