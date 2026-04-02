"""Background job runner for long-running CLI processes."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from photoalbums.lib.ai_processing_locks import _cleanup_stale_processing_locks

REPO_ROOT = Path(__file__).resolve().parent
JOBS_DIR = REPO_ROOT / "mcp" / "jobs"
JOBS_STATE = JOBS_DIR / "jobs.json"


class JobRunner:
    def __init__(self) -> None:
        self._jobs_dir = JOBS_DIR
        self._jobs_state = JOBS_STATE
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._log_handles: dict[str, object] = {}
        self._load_state()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Return True if the process with given PID is still running."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _load_state(self) -> None:
        if self._jobs_state.exists():
            try:
                data = self._read_disk_jobs()
                interrupted_jobs: list[dict] = []
                for job in data.values():
                    if job["status"] in ("pending", "running"):
                        pid = job.get("pid")
                        if not pid or not self._pid_alive(pid):
                            job["status"] = "interrupted"
                            interrupted_jobs.append(job)
                self._jobs = data
                for job in interrupted_jobs:
                    self._cleanup_job_artifacts(job)
            except Exception:
                self._jobs = {}
        self._save_state()  # always write back: creates file if missing, persists interrupted status

    def _read_disk_jobs(self) -> dict[str, dict]:
        if not self._jobs_state.exists():
            return {}
        data = json.loads(self._jobs_state.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def _save_state(self) -> None:
        merged = self._read_disk_jobs()
        merged.update(self._jobs)
        self._jobs = merged
        tmp_path = self._jobs_state.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self._jobs, indent=2), encoding="utf-8")
        os.replace(tmp_path, self._jobs_state)

    @staticmethod
    def _job_cleanup_metadata(args: list[str]) -> dict[str, str] | None:
        if len(args) < 5:
            return None
        script_path = Path(str(args[1]))
        if script_path.name != "photoalbums.py":
            return None
        if [str(part) for part in args[2:4]] != ["ai", "index"]:
            return None
        try:
            photos_root = str(args[args.index("--photos-root") + 1]).strip()
        except (ValueError, IndexError):
            return None
        if not photos_root:
            return None
        return {
            "kind": "photoalbums_ai_locks",
            "photos_root": photos_root,
        }

    @staticmethod
    def _cleanup_job_artifacts(job: dict) -> None:
        cleanup = job.get("cleanup")
        if not isinstance(cleanup, dict):
            return
        if str(cleanup.get("kind") or "").strip() != "photoalbums_ai_locks":
            return
        photos_root = Path(str(cleanup.get("photos_root") or "").strip())
        if not str(photos_root):
            return
        _cleanup_stale_processing_locks(photos_root)

    def start(
        self,
        name: str,
        args: list[str],
        cwd: Optional[str] = None,
        env_extra: Optional[dict[str, str]] = None,
    ) -> str:
        """Start a subprocess job. Returns the job ID."""
        job_id = str(uuid.uuid4())[:8]
        log_path = self._jobs_dir / f"{job_id}.log"
        artifact_path = self._jobs_dir / f"{job_id}.artifacts.jsonl"

        job: dict = {
            "id": job_id,
            "name": name,
            "command": " ".join(str(a) for a in args),
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "exit_code": None,
            "log_file": str(log_path),
            "artifact_file": str(artifact_path),
        }
        cleanup = self._job_cleanup_metadata(args)
        if cleanup is not None:
            job["cleanup"] = cleanup

        env = os.environ.copy()
        env["IMAGO_JOB_ID"] = job_id
        env["IMAGO_JOB_ARTIFACTS"] = str(artifact_path)
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

        job["pid"] = proc.pid

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
        """Get job status metadata without log content."""
        with self._lock:
            self._jobs = {**self._read_disk_jobs(), **self._jobs}
            job = self._jobs.get(job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}
        return dict(job)

    def logs(self, job_id: str, last_n: int = 100) -> str:
        """Return the last N lines of a job's log output."""
        with self._lock:
            self._jobs = {**self._read_disk_jobs(), **self._jobs}
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
            self._jobs = {**self._read_disk_jobs(), **self._jobs}
            proc = self._processes.get(job_id)
            job = self._jobs.get(job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}
        if not proc:
            return {"error": f"Job {job_id} is not running (status: {job['status']})"}
        proc.terminate()
        return {
            "ok": True,
            "message": f"Sent terminate to job {job_id} ({job['name']})",
        }

    def list_jobs(self) -> list[dict]:
        """List all jobs, newest first, without log content."""
        with self._lock:
            self._jobs = {**self._read_disk_jobs(), **self._jobs}
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.get("started_at", ""), reverse=True)

    def shutdown(self, wait_timeout_seconds: float = 5.0) -> None:
        """Terminate running jobs and clean up any stale lock files they leave behind."""
        with self._lock:
            processes = dict(self._processes)
            jobs = {job_id: self._jobs.get(job_id) for job_id in processes}

        if not processes:
            return

        for proc in processes.values():
            with contextlib.suppress(Exception):
                proc.terminate()

        deadline = time.monotonic() + max(0.0, float(wait_timeout_seconds))
        for proc in processes.values():
            remaining = max(0.0, deadline - time.monotonic())
            with contextlib.suppress(Exception):
                proc.wait(timeout=remaining)

        ended_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for job_id, proc in processes.items():
                job = self._jobs.get(job_id)
                if isinstance(job, dict) and job.get("status") in ("pending", "running"):
                    job["status"] = "interrupted"
                    job["ended_at"] = job.get("ended_at") or ended_at
                    returncode = getattr(proc, "returncode", None)
                    if isinstance(returncode, int) and job.get("exit_code") is None:
                        job["exit_code"] = returncode
                self._processes.pop(job_id, None)
            self._save_state()

        for job in jobs.values():
            if isinstance(job, dict):
                self._cleanup_job_artifacts(job)
