param(
    [Parameter(Mandatory = $false)]
    [string]$BackupPath = "$HOME\immich-db-backup-5-8-2026-1138.sql"
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path .env)) {
    throw "Missing .env. Copy .env.example to .env and fill in the runtime values first."
}

if (-not (Test-Path .env.restore)) {
    throw "Missing .env.restore. Copy .env.restore.example to .env.restore and fill in the temporary restore password first."
}

if (-not (Test-Path $BackupPath)) {
    throw "Backup file not found: $BackupPath"
}

$runtimeEnv = Get-Content .env | Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } | ConvertFrom-StringData
$runtimePassword = $runtimeEnv.DB_PASSWORD

if (-not $runtimePassword -or $runtimePassword -eq "change-me") {
    throw "Set a real DB_PASSWORD in .env before restoring."
}

Write-Host "Resetting the Immich database volume..."
docker compose down -v

Write-Host "Starting a fresh Postgres cluster with temporary restore credentials..."
docker compose --env-file .env.restore up -d database

Write-Host "Waiting for Postgres to accept connections..."
do {
    Start-Sleep -Seconds 2
    docker compose --env-file .env.restore exec -T database pg_isready -U immich_restore_admin -d postgres | Out-Null
} until ($LASTEXITCODE -eq 0)

Write-Host "Creating placeholder objects expected by the pg_dumpall backup..."
docker compose --env-file .env.restore exec -T database psql -U immich_restore_admin -d postgres -v ON_ERROR_STOP=1 -c "CREATE ROLE postgres SUPERUSER LOGIN;"
docker compose --env-file .env.restore exec -T database psql -U immich_restore_admin -d postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE immich OWNER postgres;"

Write-Host "Copying the cluster dump into the database container..."
docker cp $BackupPath immich_postgres:/tmp/immich-backup.sql

Write-Host "Restoring the cluster dump..."
docker compose --env-file .env.restore exec -T database psql -U immich_restore_admin -d postgres -v ON_ERROR_STOP=1 -f /tmp/immich-backup.sql

$escapedPassword = $runtimePassword.Replace("'", "''")
Write-Host "Resetting the restored postgres password to match .env..."
docker compose --env-file .env.restore exec -T database psql -U immich_restore_admin -d postgres -v ON_ERROR_STOP=1 -c "ALTER ROLE postgres WITH PASSWORD '$escapedPassword';"

Write-Host "Switching from restore credentials to normal runtime credentials..."
docker compose down

docker compose up -d

Write-Host "Restore complete."
