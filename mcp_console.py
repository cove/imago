"""HTTP console for the MCP job runner.

Serves a browser UI at http://localhost:8091 for listing jobs,
tailing logs, viewing structured outputs, and cancelling running
processes.

Runs as a background thread inside the MCP server process so it shares
the same JobRunner instance and can cancel jobs directly.
"""

from __future__ import annotations

import json
import os
import signal
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

from mcp_console_ui import HTML
from photoalbums.lib.xmp_review import load_ai_xmp_review

if TYPE_CHECKING:
    from mcp_job_runner import JobRunner

_JOBS_STATE = Path(__file__).resolve().parent / "mcp" / "jobs" / "jobs.json"

DEFAULT_PORT = 8091


class _Handler(BaseHTTPRequestHandler):
    @staticmethod
    def _read_jobs() -> list[dict]:
        """Read all jobs from the shared jobs.json on disk."""
        if not _JOBS_STATE.exists():
            return []
        try:
            data = json.loads(_JOBS_STATE.read_text(encoding="utf-8"))
        except Exception:
            return []
        return sorted(data.values(), key=lambda job: job.get("started_at", ""), reverse=True)

    @staticmethod
    def _read_job(job_id: str) -> dict | None:
        if not _JOBS_STATE.exists():
            return None
        try:
            return json.loads(_JOBS_STATE.read_text(encoding="utf-8")).get(job_id)
        except Exception:
            return None

    @staticmethod
    def _read_artifacts(job: dict | None) -> list[dict]:
        if not isinstance(job, dict):
            return []
        artifact_file = str(job.get("artifact_file") or "").strip()
        if not artifact_file:
            return []
        path = Path(artifact_file)
        if not path.exists():
            return []
        rows: list[dict] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            text = str(line or "").strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._html()
            return

        if path == "/api/jobs":
            self._json(self._read_jobs())
            return

        if path == "/api/image":
            self._serve_image(query)
            return

        if path == "/api/xmp-review":
            self._xmp_review(query)
            return

        if path.startswith("/api/jobs/") and path.endswith("/logs"):
            self._job_logs(path, query)
            return

        if path.startswith("/api/jobs/") and path.endswith("/artifacts"):
            self._job_artifacts(path)
            return

        if path.startswith("/api/jobs/") and path.endswith("/stream"):
            job_id = path.split("/")[3]
            self._stream_logs(job_id)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path.split("/")[3]
            job = self._read_job(job_id)
            if not job:
                self._json({"error": f"Job {job_id} not found"})
                return
            if job.get("status") not in ("running", "pending"):
                self._json({"error": f"Job {job_id} is not running (status: {job['status']})"})
                return
            pid = job.get("pid")
            if not pid:
                self._json({"error": f"Job {job_id} has no PID recorded"})
                return
            try:
                os.kill(pid, signal.SIGTERM)
                self._json({"ok": True, "message": f"Sent SIGTERM to job {job_id} (pid {pid})"})
            except OSError as exc:
                self._json({"error": str(exc)})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _job_logs(self, path: str, query: dict[str, list[str]]) -> None:
        job_id = path.split("/")[3]
        job = self._read_job(job_id)
        if not job:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        last_n = 200
        values = query.get("last") or []
        if values:
            try:
                last_n = int(values[0])
            except ValueError:
                last_n = 200
        log_path = Path(job["log_file"])
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            logs = "\n".join(lines[-last_n:])
        else:
            logs = "(no output yet)"
        self._json({"logs": logs})

    def _job_artifacts(self, path: str) -> None:
        job_id = path.split("/")[3]
        job = self._read_job(job_id)
        if not job:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._json(self._read_artifacts(job))

    def _serve_image(self, query: dict[str, list[str]]) -> None:
        path_values = query.get("path") or []
        image_path = str(path_values[0] if path_values else "").strip()
        if not image_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing path parameter")
            return
        p = Path(image_path)
        if not p.exists() or not p.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }
        ct = content_types.get(p.suffix.lower())
        if not ct:
            self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        data = p.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _xmp_review(self, query: dict[str, list[str]]) -> None:
        sidecar_values = query.get("sidecar_path") or []
        sidecar_path = str(sidecar_values[0] if sidecar_values else "").strip()
        if not sidecar_path:
            self._json({"error": "Missing sidecar_path query parameter"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = load_ai_xmp_review(sidecar_path, include_raw_xml=True)
        except ValueError as exc:
            self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._json(payload)

    def _stream_logs(self, job_id: str) -> None:
        """Stream job log lines as SSE events until the job finishes."""
        job = self._read_job(job_id)
        if not job:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        log_path = Path(job["log_file"])
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        sent = 0
        last_heartbeat = time.monotonic()
        try:
            while True:
                if log_path.exists():
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    for line in lines[sent:]:
                        self.wfile.write(f"data: {line}\n\n".encode())
                        sent += 1
                    self.wfile.flush()
                    last_heartbeat = time.monotonic()
                status = (self._read_job(job_id) or {}).get("status", "")
                if status not in ("running", "pending"):
                    self.wfile.write(b"event: done\ndata: \n\n")
                    self.wfile.flush()
                    break
                if time.monotonic() - last_heartbeat >= 15:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    last_heartbeat = time.monotonic()
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _html(self) -> None:
        body = HTML.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        pass


def start_console(
    runner: "JobRunner | None" = None,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    """Start the job console HTTP server in a daemon thread.

    Reads job state from mcp/jobs/jobs.json so it works across processes.
    The runner parameter is accepted for backwards compatibility but unused.

    Returns the server object (call .shutdown() to stop it).
    """
    server = ThreadingHTTPServer((host, port), _Handler)
    Thread(target=server.serve_forever, daemon=True, name="mcp-console").start()
    return server
