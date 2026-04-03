"""Tests for MCP job runner persistence and console integration."""

import http.client
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mcp_job_runner
import mcp_console
from photoalbums.lib import xmp_sidecar

_FAKE_JOB = {
    "id": "abc12345",
    "name": "test_job",
    "command": "...",
    "status": "running",
    "started_at": "2026-03-16T00:00:00+00:00",
    "ended_at": None,
    "exit_code": None,
    "log_file": "abc12345.log",
    "artifact_file": "abc12345.artifacts.jsonl",
    "pid": 99999,
}


class TestJobRunnerPersistence(unittest.TestCase):
    def setUp(self):
        # ignore_cleanup_errors so open log file handles on Windows don't fail tearDown
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.jobs_dir = Path(self.tmp.name) / "jobs"
        self.jobs_dir.mkdir(parents=True)
        self.jobs_state = self.jobs_dir / "jobs.json"
        # Patch module-level paths so tests don't touch the real mcp/jobs/ directory
        self._orig_jobs_dir = mcp_job_runner.JOBS_DIR
        self._orig_jobs_state = mcp_job_runner.JOBS_STATE
        self._orig_console_state = mcp_console._JOBS_STATE
        mcp_job_runner.JOBS_DIR = self.jobs_dir
        mcp_job_runner.JOBS_STATE = self.jobs_state
        mcp_console._JOBS_STATE = self.jobs_state

    def tearDown(self):
        mcp_job_runner.JOBS_DIR = self._orig_jobs_dir
        mcp_job_runner.JOBS_STATE = self._orig_jobs_state
        mcp_console._JOBS_STATE = self._orig_console_state
        self.tmp.cleanup()

    def _write_fake_jobs(self, data):
        self.jobs_state.write_text(json.dumps(data), encoding="utf-8")

    def _read_disk_jobs(self):
        return json.loads(self.jobs_state.read_text(encoding="utf-8"))

    # ── Startup behaviour ──────────────────────────────────────────────────────

    def test_fresh_start_creates_jobs_json(self):
        """A runner starting with no pre-existing jobs.json should create it."""
        self.assertFalse(self.jobs_state.exists())
        mcp_job_runner.JobRunner()
        self.assertTrue(
            self.jobs_state.exists(),
            "jobs.json should be created on startup even when there are no jobs",
        )
        self.assertEqual(self._read_disk_jobs(), {})

    def test_load_state_marks_running_as_interrupted_on_disk(self):
        """On restart, running jobs must be written back to disk as interrupted."""
        self._write_fake_jobs({"abc12345": dict(_FAKE_JOB)})

        mcp_job_runner.JobRunner()  # simulate server restart

        data = self._read_disk_jobs()
        self.assertEqual(
            data["abc12345"]["status"],
            "interrupted",
            "_load_state() must write interrupted status back to disk so the console sees it",
        )

    def test_load_state_preserves_completed_jobs(self):
        """Completed/failed jobs should be left unchanged on restart."""
        job = dict(
            _FAKE_JOB,
            status="completed",
            ended_at="2026-03-16T00:01:00+00:00",
            exit_code=0,
        )
        self._write_fake_jobs({"abc12345": job})

        mcp_job_runner.JobRunner()

        data = self._read_disk_jobs()
        self.assertEqual(data["abc12345"]["status"], "completed")

    # ── start() persistence ────────────────────────────────────────────────────

    def test_start_writes_job_to_disk_immediately(self):
        """start() must write the new job to jobs.json before returning."""
        runner = mcp_job_runner.JobRunner()
        job_id = runner.start("test", [sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            self.assertTrue(self.jobs_state.exists())
            data = self._read_disk_jobs()
            self.assertIn(job_id, data)
            self.assertEqual(data[job_id]["status"], "running")
            self.assertIn("pid", data[job_id])
        finally:
            runner.cancel(job_id)

    def test_completed_job_written_to_disk(self):
        """When a job exits, its final status must be written to disk."""
        runner = mcp_job_runner.JobRunner()
        job_id = runner.start("quick", [sys.executable, "-c", "pass"])
        for _ in range(50):
            time.sleep(0.1)
            if runner.status(job_id).get("status") in ("completed", "failed"):
                break
        data = self._read_disk_jobs()
        self.assertIn(data[job_id]["status"], ("completed", "failed"))
        self.assertIsNotNone(data[job_id]["exit_code"])

    def test_status_does_not_include_log_content(self):
        """status() should return metadata only; logs come from logs()."""
        runner = mcp_job_runner.JobRunner()
        job_id = runner.start("quick", [sys.executable, "-c", "print('hello from job')"])
        for _ in range(50):
            time.sleep(0.1)
            if runner.status(job_id).get("status") in ("completed", "failed"):
                break

        status = runner.status(job_id)

        self.assertNotIn("recent_logs", status)
        self.assertNotIn("total_log_lines", status)
        self.assertIn("hello from job", runner.logs(job_id))

    def test_start_records_artifact_file_and_passes_env(self):
        """start() should persist an artifact path and expose it to the subprocess."""
        runner = mcp_job_runner.JobRunner()
        job_id = runner.start(
            "env",
            [sys.executable, "-c", "import os; print(os.environ['IMAGO_JOB_ARTIFACTS'])"],
        )
        for _ in range(50):
            time.sleep(0.1)
            if runner.status(job_id).get("status") in ("completed", "failed"):
                break
        data = self._read_disk_jobs()
        artifact_file = data[job_id]["artifact_file"]
        self.assertTrue(str(artifact_file).endswith(".artifacts.jsonl"))
        log_path = Path(data[job_id]["log_file"])
        self.assertIn(str(artifact_file), log_path.read_text(encoding="utf-8"))

    def test_start_launches_background_process_isolated_from_console(self):
        runner = mcp_job_runner.JobRunner()
        proc = mock.Mock()
        proc.pid = 4321
        proc.returncode = 0

        with mock.patch.object(mcp_job_runner.subprocess, "Popen", return_value=proc) as popen:
            job_id = runner.start("test", [sys.executable, "-c", "pass"])

        self.assertTrue(job_id)
        kwargs = popen.call_args.kwargs
        self.assertIs(kwargs["stdin"], mcp_job_runner.subprocess.DEVNULL)
        if mcp_job_runner.os.name == "nt":
            expected_flags = (
                getattr(mcp_job_runner.subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(mcp_job_runner.subprocess, "DETACHED_PROCESS", 0)
                | getattr(mcp_job_runner.subprocess, "CREATE_NO_WINDOW", 0)
            )
            self.assertEqual(kwargs["creationflags"], expected_flags)
        else:
            self.assertTrue(kwargs["start_new_session"])

    def test_save_state_merges_jobs_from_other_runner_instances(self):
        """Saving from one runner must not drop jobs started by another runner."""
        runner_a = mcp_job_runner.JobRunner()
        runner_b = mcp_job_runner.JobRunner()

        job_a = runner_a.start("job-a", [sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            data_after_a = self._read_disk_jobs()
            self.assertIn(job_a, data_after_a)

            runner_b._jobs["external"] = {
                "id": "external",
                "name": "external",
                "command": "...",
                "status": "running",
                "started_at": "2026-03-16T00:00:00+00:00",
                "ended_at": None,
                "exit_code": None,
                "log_file": str(self.jobs_dir / "external.log"),
                "artifact_file": str(self.jobs_dir / "external.artifacts.jsonl"),
                "pid": 12345,
            }
            runner_b._save_state()

            merged = self._read_disk_jobs()
            self.assertIn(job_a, merged)
            self.assertIn("external", merged)
        finally:
            runner_a.cancel(job_a)

    def test_shutdown_cleans_stale_photoalbums_lock_files(self):
        runner = mcp_job_runner.JobRunner()
        photos_root = self.jobs_dir / "photos"
        image_path = photos_root / "Album_A" / "Photo_01.jpg"
        image_path.parent.mkdir(parents=True)
        image_path.touch()

        lock_path = image_path.with_name(f"{image_path.name}.photoalbums-ai.lock")
        batch_lock_path = photos_root / ".photoalbums-ai.batch.lock"
        payload = json.dumps({"pid": 999999, "job_id": "job123"})
        lock_path.write_text(payload, encoding="utf-8")
        batch_lock_path.write_text(payload, encoding="utf-8")

        proc = mock.Mock()
        proc.pid = 999999
        proc.returncode = -15
        job = {
            "id": "job123",
            "name": "photoalbums_ai_index:cordell",
            "command": "...",
            "status": "running",
            "started_at": "2026-03-16T00:00:00+00:00",
            "ended_at": None,
            "exit_code": None,
            "log_file": str(self.jobs_dir / "job123.log"),
            "artifact_file": str(self.jobs_dir / "job123.artifacts.jsonl"),
            "pid": 999999,
            "cleanup": {
                "kind": "photoalbums_ai_locks",
                "photos_root": str(photos_root),
            },
        }
        runner._jobs["job123"] = job
        runner._processes["job123"] = proc
        runner._save_state()

        runner.shutdown()

        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()
        self.assertFalse(lock_path.exists())
        self.assertFalse(batch_lock_path.exists())
        saved = self._read_disk_jobs()
        self.assertEqual(saved["job123"]["status"], "interrupted")

    # ── Console visibility ─────────────────────────────────────────────────────

    def test_console_reads_jobs_written_by_runner(self):
        """_Handler._read_jobs() must see jobs written by JobRunner.start()."""
        runner = mcp_job_runner.JobRunner()
        job_id = runner.start("test", [sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            jobs = mcp_console._Handler._read_jobs()
            ids = [j["id"] for j in jobs]
            self.assertIn(job_id, ids, "Console should see the running job written by start()")
        finally:
            runner.cancel(job_id)

    def test_console_sees_interrupted_jobs_after_restart(self):
        """After a server restart, the console should show old jobs as interrupted.

        Uses a fake jobs.json to avoid timing races with the _wait() thread.
        """
        self._write_fake_jobs({"abc12345": dict(_FAKE_JOB)})

        # Simulate server restart
        mcp_job_runner.JobRunner()

        jobs = mcp_console._Handler._read_jobs()
        self.assertTrue(len(jobs) > 0, "Console should show jobs after restart")
        match = next((j for j in jobs if j["id"] == "abc12345"), None)
        self.assertIsNotNone(match, "Job should be visible in console")
        self.assertEqual(match["status"], "interrupted")


class TestConsoleHTTPStream(unittest.TestCase):
    """Integration tests for the /api/jobs/{id}/stream SSE endpoint."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.jobs_dir = Path(self.tmp.name) / "jobs"
        self.jobs_dir.mkdir(parents=True)
        self.jobs_state = self.jobs_dir / "jobs.json"

        self._orig_console_state = mcp_console._JOBS_STATE
        mcp_console._JOBS_STATE = self.jobs_state

        # Port 0 → OS picks a free port. daemon_threads so handlers don't block cleanup.
        self.server = mcp_console.start_console(host="127.0.0.1", port=0)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        mcp_console._JOBS_STATE = self._orig_console_state
        self.tmp.cleanup()

    def _write_fake_jobs(self, data):
        self.jobs_state.write_text(json.dumps(data), encoding="utf-8")

    def _request(self, path, timeout=5):
        """Send GET and return (status, content_type). Closes connection immediately."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, resp.getheader("Content-Type")
        finally:
            conn.close()

    def _request_json(self, path, timeout=5):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8")
            return resp.status, resp.getheader("Content-Type"), json.loads(body)
        finally:
            conn.close()

    # ── /stream route ──────────────────────────────────────────────────────────

    def test_stream_200_for_known_job(self):
        """Stream endpoint returns 200 + text/event-stream for a job that exists."""
        self._write_fake_jobs({"abc12345": dict(_FAKE_JOB)})
        status, ct = self._request("/api/jobs/abc12345/stream")
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", ct or "")

    def test_stream_404_for_unknown_job(self):
        """Stream endpoint returns 404 when job ID is not in jobs.json."""
        self._write_fake_jobs({})
        status, _ = self._request("/api/jobs/doesnotexist/stream")
        self.assertEqual(status, 404)

    def test_stream_404_when_jobs_json_missing(self):
        """Stream endpoint returns 404 when jobs.json doesn't exist yet."""
        # No jobs.json written — simulates fresh server with no jobs started
        status, _ = self._request("/api/jobs/abc12345/stream")
        self.assertEqual(status, 404)

    def test_stream_and_logs_routes_are_distinct(self):
        """/stream and /logs routes both work independently for the same job."""
        self._write_fake_jobs({"abc12345": dict(_FAKE_JOB)})
        stream_status, stream_ct = self._request("/api/jobs/abc12345/stream")
        logs_status, logs_ct = self._request("/api/jobs/abc12345/logs")
        self.assertEqual(stream_status, 200)
        self.assertIn("text/event-stream", stream_ct or "")
        self.assertEqual(logs_status, 200)
        self.assertIn("application/json", logs_ct or "")

    def test_artifacts_route_returns_job_artifacts(self):
        artifact_path = self.jobs_dir / "abc12345.artifacts.jsonl"
        artifact_path.write_text(
            json.dumps(
                {
                    "kind": "photoalbums_xmp",
                    "image_path": "C:\\photos\\Album_A\\Photo_01.jpg",
                    "sidecar_path": "C:\\photos\\Album_A\\Photo_01.xmp",
                    "label": "Photo_01.jpg",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        job = dict(_FAKE_JOB, artifact_file=str(artifact_path))
        self._write_fake_jobs({"abc12345": job})

        status, ct, payload = self._request_json("/api/jobs/abc12345/artifacts")

        self.assertEqual(status, 200)
        self.assertIn("application/json", ct or "")
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["sidecar_path"], "C:\\photos\\Album_A\\Photo_01.xmp")

    def test_xmp_review_route_returns_review_data(self):
        image_path = self.jobs_dir / "Photo_01.jpg"
        image_path.touch()
        sidecar_path = image_path.with_suffix(".xmp")
        xmp_sidecar.write_xmp_sidecar(
            sidecar_path,
            creator_tool="https://github.com/cove/imago",
            person_names=["Alice Example"],
            subjects=["bench"],
            description="Alice Example on a bench.",
            album_title="Family Book I",
            source_text="scan_001.tif",
            ocr_text="FAMILY BOOK",
            detections_payload={"objects": [{"label": "bench", "score": 0.8}]},
        )

        status, ct, payload = self._request_json(f"/api/xmp-review?sidecar_path={quote(str(sidecar_path))}")

        self.assertEqual(status, 200)
        self.assertIn("application/json", ct or "")
        self.assertEqual(payload["sidecar_path"], str(sidecar_path.resolve()))
        self.assertEqual(payload["person_names"], ["Alice Example"])
        self.assertIn("raw_xml", payload)


if __name__ == "__main__":
    unittest.main()
