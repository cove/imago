"""Background job runner for long-running CLI processes."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
JOBS_DIR = REPO_ROOT / "mcp" / "jobs"
JOBS_STATE = JOBS_DIR / "jobs.json"


class JobRunner:
    def __init__(self) -> None:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._log_handles: dict[str, object] = {}
        self._load_state()

    def _load_state(self) -> None:
        if not JOBS_STATE.exists():
            return
        try:
            data = json.loads(JOBS_STATE.read_text(encoding="utf-8"))
            for job in data.values():
                if job["status"] in ("pending", "running"):
                    job["status"] = "interrupted"
            self._jobs = data
        except Exception:
            self._jobs = {}

    def _save_state(self) -> None:
        JOBS_STATE.write_text(json.dumps(self._jobs, indent=2), encoding="utf-8")

    def start(
        self,
        name: str,
        args: list[str],
        cwd: Optional[str] = None,
        env_extra: Optional[dict[str, str]] = None,
    ) -> str:
        """Start a subprocess job. Returns the job ID."""
        job_id = str(uuid.uuid4())[:8]
        log_path = JOBS_DIR / f"{job_id}.log"

        job: dict = {
            "id": job_id,
            "name": name,
            "command": " ".join(str(a) for a in args),
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "exit_code": None,
            "log_file": str(log_path),
        }

        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        log_file = open(log_path, "w", encoding="utf-8")  # noqa: WPS515

        proc = subprocess.Popen(
            args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=cwd or str(REPO_ROOT),
            env=env,
        )

        with self._lock:
            self._jobs[job_id] = job
            self._processes[job_id] = proc
            self._log_handles[job_id] = log_file
            self._save_state()

        def _wait() -> None:
            proc.wait()
            log_file.close()
            with self._lock:
                job["exit_code"] = proc.returncode
                job["status"] = "completed" if proc.returncode == 0 else "failed"
                job["ended_at"] = datetime.now(timezone.utc).isoformat()
                self._processes.pop(job_id, None)
                self._log_handles.pop(job_id, None)
                self._save_state()

        threading.Thread(target=_wait, daemon=True, name=f"job-{job_id}").start()
        return job_id

    def status(self, job_id: str) -> dict:
        """Get job status with recent log tail (last 30 lines)."""
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}
        result = dict(job)
        log_path = Path(job["log_file"])
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            result["recent_logs"] = "\n".join(lines[-30:])
            result["total_log_lines"] = len(lines)
        else:
            result["recent_logs"] = ""
            result["total_log_lines"] = 0
        return result

    def logs(self, job_id: str, last_n: int = 100) -> str:
        """Return the last N lines of a job's log output."""
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return f"Job {job_id} not found"
        log_path = Path(job["log_file"])
        if not log_path.exists():
            return "(no log output yet)"
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-last_n:])

    def cancel(self, job_id: str) -> dict:
        """Send a termination signal to a running job."""
        with self._lock:
            proc = self._processes.get(job_id)
            job = self._jobs.get(job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}
        if not proc:
            return {"error": f"Job {job_id} is not running (status: {job['status']})"}
        proc.terminate()
        return {"ok": True, "message": f"Sent terminate to job {job_id} ({job['name']})"}

    def list_jobs(self) -> list[dict]:
        """List all jobs, newest first, without log content."""
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.get("started_at", ""), reverse=True)
