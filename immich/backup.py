#!/usr/bin/env python3
"""
Trigger an Immich database backup via the API, copy it out of the container,
and save it xz-compressed into this directory.

Puts Immich into maintenance mode (all background jobs paused) for the
duration of the backup, then restores prior state.

Usage:
    python backup.py [--out-dir PATH]
"""

import argparse
import gzip
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


def job_command(base_url: str, headers: dict, job: str, command: str) -> None:
    resp = requests.put(
        f"{base_url}/api/jobs/{job}",
        headers=headers,
        json={"command": command},
        timeout=10,
    )
    resp.raise_for_status()


def enter_maintenance(base_url: str, headers: dict) -> list[str]:
    """Pause all background jobs; return the list of jobs that were not already paused."""
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
        check=True,
        capture_output=True,
        text=True,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backup Immich DB via API and export as .sql.xz"
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
    print(f"Done: {out_file}  ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
