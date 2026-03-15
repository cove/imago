"""HTTP console for the MCP job runner.

Serves a browser UI at http://localhost:8091 for listing jobs,
tailing logs, and cancelling running processes.

Runs as a background thread inside the MCP server process so it shares
the same JobRunner instance and can cancel jobs directly.
"""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_job_runner import JobRunner

DEFAULT_PORT = 8091

# ── HTML console ───────────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Imago Job Console</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #111; color: #ddd;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
  header { padding: 10px 16px; background: #1a1a1a; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
  header h1 { font-size: 15px; font-weight: 600; color: #eee; letter-spacing: .02em; }
  header .subtitle { font-size: 12px; color: #666; }
  .main { display: flex; flex: 1; overflow: hidden; }

  /* Job list */
  .job-list { width: 340px; flex-shrink: 0; border-right: 1px solid #2a2a2a;
              display: flex; flex-direction: column; overflow: hidden; }
  .job-list-header { padding: 8px 12px; font-size: 11px; font-weight: 600;
                     color: #666; text-transform: uppercase; letter-spacing: .08em;
                     border-bottom: 1px solid #222; background: #161616; flex-shrink: 0; }
  .job-list-body { overflow-y: auto; flex: 1; }
  .job-item { padding: 10px 12px; border-bottom: 1px solid #1e1e1e; cursor: pointer;
              display: flex; flex-direction: column; gap: 4px; transition: background .1s; }
  .job-item:hover { background: #1a1a1a; }
  .job-item.selected { background: #1c2333; border-left: 3px solid #4a7fc1; }
  .job-item .job-name { font-size: 12px; font-weight: 500; color: #ccc;
                        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .job-item .job-meta { font-size: 11px; color: #555; display: flex; gap: 8px; }
  .job-item .job-actions { margin-top: 4px; }
  .badge { display: inline-block; font-size: 10px; font-weight: 600; padding: 1px 6px;
           border-radius: 3px; text-transform: uppercase; letter-spacing: .04em; }
  .badge-running   { background: #0d3d1a; color: #3fb950; }
  .badge-completed { background: #1e1e1e; color: #666; }
  .badge-failed    { background: #3d0d0d; color: #f85149; }
  .badge-interrupted { background: #3d2a0d; color: #d29922; }
  .badge-cancelled { background: #3d2a0d; color: #d29922; }
  .badge-pending   { background: #0d2a3d; color: #58a6ff; }
  .btn-cancel { font-size: 10px; padding: 2px 8px; border-radius: 3px; border: 1px solid #555;
                background: transparent; color: #bbb; cursor: pointer; }
  .btn-cancel:hover { border-color: #f85149; color: #f85149; }
  .empty { padding: 24px 12px; text-align: center; font-size: 12px; color: #555; }

  /* Log panel */
  .log-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .log-header { padding: 8px 14px; border-bottom: 1px solid #222; background: #161616;
                display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
  .log-header .log-title { font-size: 12px; font-weight: 500; color: #ccc; flex: 1;
                            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .log-header .log-controls { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
  .log-header label { font-size: 11px; color: #666; display: flex; align-items: center; gap: 4px; cursor: pointer; }
  .log-header select { font-size: 11px; background: #222; color: #aaa; border: 1px solid #333;
                       border-radius: 3px; padding: 2px 4px; }
  .log-body { flex: 1; overflow-y: auto; padding: 10px 14px; }
  .log-body pre { font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
                  font-size: 11.5px; line-height: 1.55; white-space: pre-wrap;
                  word-break: break-all; color: #b0b0b0; }
  .log-placeholder { display: flex; align-items: center; justify-content: center;
                     height: 100%; font-size: 13px; color: #444; }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
         background: #3fb950; margin-right: 6px; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
</style>
</head>
<body>
<header>
  <h1>Imago Job Console</h1>
  <span class="subtitle">mcp/jobs &bull; auto-refreshes every 2s</span>
</header>
<div class="main">
  <div class="job-list">
    <div class="job-list-header">Jobs</div>
    <div class="job-list-body" id="job-list"></div>
  </div>
  <div class="log-panel">
    <div class="log-header">
      <span class="log-title" id="log-title">Select a job to view logs</span>
      <div class="log-controls">
        <label>
          Lines:
          <select id="log-lines">
            <option value="100">100</option>
            <option value="300">300</option>
            <option value="1000">1000</option>
            <option value="5000">5000</option>
          </select>
        </label>
        <label>
          <input type="checkbox" id="auto-scroll" checked> Auto-scroll
        </label>
      </div>
    </div>
    <div class="log-body" id="log-body">
      <div class="log-placeholder">No job selected</div>
    </div>
  </div>
</div>
<script>
  let selectedId = null;
  let jobs = [];

  function badge(status) {
    return `<span class="badge badge-${status}">${status}</span>`;
  }

  function duration(job) {
    if (!job.started_at) return '';
    const start = new Date(job.started_at);
    const end = job.ended_at ? new Date(job.ended_at) : new Date();
    const s = Math.round((end - start) / 1000);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s/60)}m ${s%60}s`;
    return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
  }

  function timeAgo(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  }

  async function fetchJobs() {
    try {
      const r = await fetch('/api/jobs');
      jobs = await r.json();
      renderJobList();
    } catch (e) { /* server may be starting */ }
  }

  function renderJobList() {
    const el = document.getElementById('job-list');
    if (!jobs.length) {
      el.innerHTML = '<div class="empty">No jobs yet</div>';
      return;
    }
    el.innerHTML = jobs.map(j => {
      const sel = j.id === selectedId ? ' selected' : '';
      const cancelBtn = j.status === 'running'
        ? `<div class="job-actions"><button class="btn-cancel" onclick="cancelJob('${j.id}', event)">Cancel</button></div>`
        : '';
      const live = j.status === 'running' ? '<span class="dot"></span>' : '';
      return `
        <div class="job-item${sel}" onclick="selectJob('${j.id}')">
          <div class="job-name">${live}${escHtml(j.name)}</div>
          <div class="job-meta">
            ${badge(j.status)}
            <span>${timeAgo(j.started_at)}</span>
            <span>${duration(j)}</span>
          </div>
          ${cancelBtn}
        </div>`;
    }).join('');
  }

  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function selectJob(id) {
    selectedId = id;
    renderJobList();
    fetchLogs();
  }

  async function fetchLogs() {
    if (!selectedId) return;
    const job = jobs.find(j => j.id === selectedId);
    if (!job) return;

    const n = document.getElementById('log-lines').value;
    try {
      const r = await fetch(`/api/jobs/${selectedId}/logs?last=${n}`);
      const data = await r.json();
      const body = document.getElementById('log-body');
      const title = document.getElementById('log-title');
      title.textContent = `${job.name}  [${job.id}]`;

      const autoScroll = document.getElementById('auto-scroll').checked;
      const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 40;

      body.innerHTML = `<pre>${escHtml(data.logs || '(no output yet)')}</pre>`;

      if (autoScroll && (job.status === 'running' || atBottom)) {
        body.scrollTop = body.scrollHeight;
      }
    } catch (e) {}
  }

  async function cancelJob(id, ev) {
    ev.stopPropagation();
    if (!confirm('Cancel this job?')) return;
    await fetch(`/api/jobs/${id}/cancel`, {method: 'POST'});
    await fetchJobs();
  }

  document.getElementById('log-lines').addEventListener('change', fetchLogs);

  // Tick: refresh job list and logs every 2s
  async function tick() {
    await fetchJobs();
    await fetchLogs();
  }
  tick();
  setInterval(tick, 2000);
</script>
</body>
</html>
"""


# ── HTTP handler ───────────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    runner: JobRunner  # injected at class creation time

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._html()
        elif path == "/api/jobs":
            self._json(self.runner.list_jobs())
        elif path.startswith("/api/jobs/") and path.endswith("/logs"):
            job_id = path.split("/")[3]
            # Parse ?last=N from query string
            last_n = 200
            if "?" in self.path:
                for part in self.path.split("?", 1)[1].split("&"):
                    if part.startswith("last="):
                        try:
                            last_n = int(part[5:])
                        except ValueError:
                            pass
            self._json({"logs": self.runner.logs(job_id, last_n)})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path.split("/")[3]
            self._json(self.runner.cancel(job_id))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _html(self) -> None:
        body = _HTML.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: object) -> None:
        body = json.dumps(data).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # suppress per-request logs


# ── Public API ─────────────────────────────────────────────────────────────────


def start_console(
    runner: JobRunner,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    """Start the job console HTTP server in a daemon thread.

    Shares the same JobRunner instance as the MCP server, so cancel works
    without any cross-process signalling.

    Returns the server object (call .shutdown() to stop it).
    """
    handler_class = type("Handler", (_Handler,), {"runner": runner})
    server = ThreadingHTTPServer((host, port), handler_class)
    Thread(target=server.serve_forever, daemon=True, name="mcp-console").start()
    print(f"Job console: http://localhost:{port}")
    return server
