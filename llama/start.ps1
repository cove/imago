$ErrorActionPreference = "Stop"

$envPath = Join-Path $PSScriptRoot ".env"
Get-Content $envPath | ForEach-Object {
    if ($_ -match '^\s*(#|$)') {
        return
    }

    $name, $value = $_ -split "=", 2
    Set-Item -Path "Env:$name" -Value $value
}

$serverBin = $env:LLAMA_SERVER_BIN
if (-not (Get-Command $serverBin -ErrorAction SilentlyContinue)) {
    $packageRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    $serverExe = Get-ChildItem $packageRoot -Recurse -Filter "$serverBin.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($serverExe) {
        $serverBin = $serverExe.FullName
    }
}

$startupError = $false
& $serverBin `
    --model (Join-Path $env:MODEL_DIR $env:MODEL_FILE) `
    --mmproj (Join-Path $env:MODEL_DIR $env:MMPROJ_FILE) `
    --alias $env:MODEL_ALIAS `
    --host $env:HOST `
    --port $env:PORT `
    --parallel $env:PARALLEL `
    --ctx-size $env:CTX_SIZE `
    --cache-ram $env:CACHE_RAM 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Write-Host $line
        if ($line -match "(?i)failed to open GGUF|error loading model|failed to load model|exiting due to model loading error") {
            $script:startupError = $true
        }
    }

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($startupError) {
    exit 1
}
