#!/usr/bin/env python3
"""
Trigger an Immich database backup via the API, copy it out of the container,
and save it gzip-9 compressed into this directory.

Puts Immich into maintenance mode (all background jobs paused) for the
duration of the backup, then restores prior state.

After saving, validates by spinning up a full throwaway Immich stack via
`docker compose -p <project>` using a generated compose file that pulls the
exact same image digests as production. Two-phase startup:

  Phase 1 — cold start with empty DB: Immich initialises the storage volume
             (creates .immich marker files, runs migrations).
  Phase 2 — restore backup, restart server, verify asset count via the
             Immich API matches the live server.

Tears the stack down (docker compose down -v) whether or not validation passes.

Usage:
    python backup.py [--out-dir PATH]
"""

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
TIMEOUT_MINUTES = 15
CONTAINER_BACKUP_DIR = "/data/backups"
SERVER_CONTAINER = "immich_server"

# All job queues except backupDatabase (which we need to run).
BACKGROUND_JOBS = [
    "thumbnailGeneration",
    "metadataExtraction",
    "videoConversion",
    "faceDetection",
    "facialRecognition",
    "smartSearch",
    "duplicateDetection",
    "backgroundTask",
    "storageTemplateMigration",
    "migration",
    "search",
    "sidecar",
    "library",
    "notifications",
]


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip()
    return env


def _container_image(container: str) -> str:
    return subprocess.run(
        ["docker", "inspect", "--format", "{{.Config.Image}}", container],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def job_command(base_url: str, headers: dict, job: str, command: str) -> None:
    resp = requests.put(
        f"{base_url}/api/jobs/{job}",
        headers=headers,
        json={"command": command},
        timeout=10,
    )
    resp.raise_for_status()


def enter_maintenance(base_url: str, headers: dict) -> list[str]:
    """Pause all background jobs; return the list that were not already paused."""
    resp = requests.get(f"{base_url}/api/jobs", headers=headers, timeout=10)
    resp.raise_for_status()
    statuses = resp.json()
    to_pause = [
        j for j in BACKGROUND_JOBS
        if not statuses.get(j, {}).get("queueStatus", {}).get("isPaused", False)
    ]
    print(f"Maintenance mode: pausing {len(to_pause)} job queue(s)...", flush=True)
    for job in to_pause:
        job_command(base_url, headers, job, "pause")
    return to_pause


def exit_maintenance(base_url: str, headers: dict, to_resume: list[str]) -> None:
    print(f"Resuming {len(to_resume)} job queue(s)...", flush=True)
    for job in to_resume:
        job_command(base_url, headers, job, "resume")


def trigger_and_wait(base_url: str, headers: dict) -> None:
    resp = requests.get(f"{base_url}/api/jobs", headers=headers, timeout=10)
    resp.raise_for_status()
    failed_before = resp.json()["backupDatabase"]["jobCounts"]["failed"]

    print("Triggering database backup...", flush=True)
    resp = requests.put(
        f"{base_url}/api/jobs/backupDatabase",
        headers=headers,
        json={"command": "start", "force": False},
        timeout=10,
    )
    resp.raise_for_status()

    time.sleep(3)  # let the job register before polling

    print(f"Polling backup job (timeout {TIMEOUT_MINUTES} min)...", flush=True)
    deadline = time.monotonic() + TIMEOUT_MINUTES * 60
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Backup did not finish within {TIMEOUT_MINUTES} minutes")
        time.sleep(5)
        resp = requests.get(f"{base_url}/api/jobs", headers=headers, timeout=10)
        resp.raise_for_status()
        counts = resp.json()["backupDatabase"]["jobCounts"]
        print(f"  waiting={counts['waiting']} active={counts['active']}", flush=True)
        if counts["active"] == 0 and counts["waiting"] == 0:
            if counts["failed"] > failed_before:
                raise RuntimeError(f"Backup job failed (failed count rose to {counts['failed']})")
            return


def latest_backup_in_container() -> str:
    result = subprocess.run(
        [
            "docker", "exec", SERVER_CONTAINER,
            "sh", "-c", f"ls -t {CONTAINER_BACKUP_DIR}/*.sql.gz 2>/dev/null | head -1",
        ],
        check=True, capture_output=True, text=True,
    )
    path = result.stdout.strip()
    if not path:
        raise FileNotFoundError(
            f"No *.sql.gz files found in {SERVER_CONTAINER}:{CONTAINER_BACKUP_DIR}"
        )
    return path


def recompress_to_gz9(gz_path: Path, out_path: Path) -> None:
    print("Recompressing to gzip -9...", flush=True)
    with gzip.open(gz_path, "rb") as src, gzip.open(out_path, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)


def live_asset_count(base_url: str, headers: dict) -> int:
    resp = requests.get(f"{base_url}/api/assets/statistics", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["videos"] + data["images"]


def _wait_immich_container(container: str, timeout_s: int = 300) -> None:
    """Poll the Immich ping endpoint via docker exec — no host port mapping needed."""
    print("  Waiting for Immich server to be healthy...", flush=True)
    deadline = time.monotonic() + timeout_s
    while True:
        if time.monotonic() > deadline:
            logs = subprocess.run(
                ["docker", "logs", container, "--tail", "40"],
                capture_output=True, text=True,
            )
            raise TimeoutError(
                f"Immich validation server not ready after {timeout_s}s.\n"
                f"Last logs:\n{logs.stdout}{logs.stderr}"
            )
        result = subprocess.run(
            ["docker", "exec", container,
             "curl", "-fsS", "-m", "2", "http://localhost:2283/api/server/ping"],
            capture_output=True,
        )
        if result.returncode == 0:
            return
        time.sleep(3)


def validate_backup(backup_path: Path, cfg: dict, base_url: str, headers: dict) -> None:
    """
    Spin up a throwaway Immich stack via docker compose -p <project> using a
    generated compose file with the exact same image digests as production.

    Two-phase startup:
      1. Cold boot with empty DB so Immich initialises the storage volume
         (.immich markers, migrations). Wait for healthy, then stop the server.
      2. Restore the backup. Restart the server and verify the Immich API
         reports the same asset count as the live server.

    Queries use docker exec curl inside the container — no host port binding.
    """
    api_key  = headers["x-api-key"]
    db_user  = cfg.get("DB_USERNAME", "postgres")
    db_name  = cfg.get("DB_DATABASE_NAME", "immich")
    stamp    = int(time.time())
    project  = f"immich-val-{stamp}"
    pg_ctr   = f"{project}-database-1"
    srv_ctr  = f"{project}-immich-server-1"

    pg_image     = _container_image("immich_postgres")
    redis_image  = _container_image("immich_redis")
    server_image = _container_image("immich_server")

    # Generated compose file — no container_name directives so docker compose
    # -p namespaces everything cleanly without clashing with production.
    val_compose = SCRIPT_DIR / f".docker-compose.validate-{stamp}.yml"
    val_compose.write_text(f"""\
services:
  database:
    image: "{pg_image}"
    environment:
      POSTGRES_PASSWORD: "validate_pw"
      POSTGRES_USER: "{db_user}"
      POSTGRES_DB: "{db_name}"
      POSTGRES_INITDB_ARGS: "--data-checksums"
    shm_size: 128mb
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: "{redis_image}"

  immich-server:
    image: "{server_image}"
    environment:
      DB_HOSTNAME: database
      DB_PORT: "5432"
      DB_USERNAME: "{db_user}"
      DB_PASSWORD: "validate_pw"
      DB_DATABASE_NAME: "{db_name}"
      REDIS_HOSTNAME: redis
    volumes:
      - library:/data
    depends_on:
      - database
      - redis

volumes:
  pgdata:
  library:
""", encoding="utf-8")

    compose = [
        "docker", "compose", "-p", project, "-f", str(val_compose),
    ]

    print(f"\nValidation: spinning up Immich stack (project={project})...", flush=True)
    try:
        # --- Phase 1: cold start so Immich initialises the storage volume ---
        print("  Phase 1: cold start (Immich initialises storage)...", flush=True)
        subprocess.run([*compose, "up", "-d"], check=True, capture_output=True)
        _wait_immich_container(srv_ctr)
        subprocess.run([*compose, "stop", "immich-server"], check=True, capture_output=True)
        print("  Storage initialised.", flush=True)

        # --- Restore backup ---
        print("  Restoring backup...", flush=True)
        t0 = time.monotonic()
        psql = subprocess.Popen(
            ["docker", "exec", "-i", pg_ctr,
             "psql", "-U", db_user, "-d", db_name, "-q", "-v", "ON_ERROR_STOP=1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        with gzip.open(backup_path, "rb") as gz:
            shutil.copyfileobj(gz, psql.stdin)
        psql.stdin.close()
        out, _ = psql.communicate()
        if psql.returncode != 0:
            raise RuntimeError(
                f"Restore failed (exit {psql.returncode}):\n"
                f"{out.decode('utf-8', errors='replace')[-3000:]}"
            )
        print(f"  Restore: {time.monotonic() - t0:.0f}s", flush=True)

        # --- Phase 2: restart server with restored data ---
        print("  Phase 2: restarting server with restored data...", flush=True)
        subprocess.run([*compose, "up", "-d", "immich-server"], check=True, capture_output=True)
        _wait_immich_container(srv_ctr)

        # --- Compare counts via Immich API (exec inside container) ---
        stat_out = subprocess.run(
            ["docker", "exec", srv_ctr,
             "curl", "-fsS",
             "-H", f"x-api-key: {api_key}",
             "http://localhost:2283/api/assets/statistics"],
            capture_output=True, text=True, check=True,
        ).stdout
        stat     = json.loads(stat_out)
        restored = stat["videos"] + stat["images"]
        expected = live_asset_count(base_url, headers)

        print(f"  validation stack: assets={restored}", flush=True)
        print(f"  live server:      assets={expected}", flush=True)

        if restored != expected:
            raise RuntimeError(
                f"Validation failed: restored count ({restored}) != live count ({expected})"
            )
        print("Validation passed.", flush=True)

    finally:
        print("  Tearing down validation stack...", flush=True)
        subprocess.run([*compose, "down", "-v"], capture_output=True)
        val_compose.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backup Immich DB via API, export as gzip-9 .sql.gz, then E2E validate."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=SCRIPT_DIR,
        help="Directory to write the backup file (default: immich/)",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_file = SCRIPT_DIR / ".env"
    if not env_file.exists():
        sys.exit("Missing .env — copy .env.example and fill it in first")

    cfg = load_env(env_file)
    api_key = cfg.get("IMMICH_API_KEY", "")
    if not api_key or api_key == "change-me":
        sys.exit("Set IMMICH_API_KEY in .env before running backup")

    base_url = (cfg.get("EXTERNAL_IMMICH_URL") or "http://localhost:2283").rstrip("/")
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    paused_jobs = enter_maintenance(base_url, headers)
    try:
        trigger_and_wait(base_url, headers)
    finally:
        exit_maintenance(base_url, headers, paused_jobs)

    print("Locating latest backup in container...", flush=True)
    container_path = latest_backup_in_container()
    gz_name = Path(container_path).name
    print(f"Found: {gz_name}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        print("Copying from container...", flush=True)
        subprocess.run(
            ["docker", "cp", f"{SERVER_CONTAINER}:{container_path}", str(tmp_path)],
            check=True,
        )

        gz_path = tmp_path / gz_name
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        out_file = out_dir / f"immich-db-backup-{stamp}.sql.gz"

        recompress_to_gz9(gz_path, out_file)

    size_mb = out_file.stat().st_size / 1_048_576
    print(f"Saved: {out_file.name}  ({size_mb:.1f} MB)", flush=True)

    validate_backup(out_file, cfg, base_url, headers)


if __name__ == "__main__":
    main()
