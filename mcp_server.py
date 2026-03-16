"""Unified MCP server for imago projects (cast, photoalbums, vhs).

Run with:  python mcp_server.py
Or register in Claude Desktop / any MCP client.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_console import start_console
from mcp_job_runner import JobRunner
from photoalbums.common import PHOTO_ALBUMS_DIR

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")

CAST_STORE_DEFAULT = str(REPO_ROOT / "cast" / "data")
PHOTOS_ROOT_DEFAULT = str(PHOTO_ALBUMS_DIR)
VHS_DIR = str(REPO_ROOT / "vhs")
VHS_SCRIPT = str(REPO_ROOT / "vhs" / "vhs.py")
CAST_SCRIPT = str(REPO_ROOT / "cast.py")
PHOTOALBUMS_SCRIPT = str(REPO_ROOT / "photoalbums.py")
MANIFEST_DEFAULT = str(REPO_ROOT / "photoalbums" / "data" / "ai_index_manifest.jsonl")

CONSOLE_PORT = 8091
CONSOLE_HOST = "localhost"  # overridden at startup by --console-host

mcp = FastMCP("imago")
runner = JobRunner()


def _job_started(job_id: str) -> dict:
    """Wrap a job ID with monitoring instructions for MCP clients."""
    return {
        "job_id": job_id,
        "status": "started",
        "how_to_monitor": (
            f"Call job_logs('{job_id}') to read output. "
            f"Call job_status('{job_id}') to check completion. "
            f"Stream live output via SSE: GET http://{CONSOLE_HOST}:{CONSOLE_PORT}/api/jobs/{job_id}/stream"
        ),
    }


# ── Job management ─────────────────────────────────────────────────────────────


@mcp.tool()
def job_list() -> list[dict]:
    """List all background jobs and their status (newest first)."""
    return runner.list_jobs()


@mcp.tool()
def job_status(job_id: str) -> dict:
    """Get status and recent log tail for a job.

    Args:
        job_id: Job ID returned when the job was started.
    """
    return runner.status(job_id)


@mcp.tool()
def job_logs(job_id: str, last_n_lines: int = 100) -> str:
    """Get log output from a job.

    Args:
        job_id: Job ID.
        last_n_lines: Number of recent lines to return (default 100).
    """
    return runner.logs(job_id, last_n_lines)


@mcp.tool()
def job_cancel(job_id: str) -> dict:
    """Cancel a running job by sending a terminate signal.

    Args:
        job_id: Job ID to cancel.
    """
    return runner.cancel(job_id)


@mcp.resource("jobs://index")
def jobs_index_resource() -> str:
    """All known background jobs with status and log resource URIs.

    Discoverable in LM Studio under Resources. This is the entry point —
    read it to find job IDs, then read individual logs via job://<job_id>/logs.
    """
    jobs = runner.list_jobs()
    if not jobs:
        return "No jobs found."
    lines = ["# Imago Background Jobs", ""]
    for job in jobs:
        job_id = job["id"]
        lines += [
            f"## {job['name']}",
            f"- ID: `{job_id}`",
            f"- Status: {job['status']}",
            f"- Log resource URI: `job://{job_id}/logs`",
            f"- Web stream: http://{CONSOLE_HOST}:{CONSOLE_PORT}/api/jobs/{job_id}/stream",
            "",
        ]
    return "\n".join(lines)


@mcp.resource("job://{job_id}/logs")
def job_log_resource(job_id: str) -> str:
    """Full log output for a job, readable as an MCP resource.

    URI: job://<job_id>/logs
    """
    return runner.logs(job_id, last_n=10_000)


# ── Cast: read-only data tools ─────────────────────────────────────────────────


@mcp.tool()
def cast_list_people(store_dir: str = CAST_STORE_DEFAULT) -> list[dict]:
    """List all people in the Cast face identity store.

    Returns name, aliases, notes, and face count per person (no embeddings).

    Args:
        store_dir: Path to cast data directory (default: cast/data).
    """
    people_path = Path(store_dir) / "people.json"
    if not people_path.exists():
        return []
    data = json.loads(people_path.read_text(encoding="utf-8"))
    return [
        {
            "id": p.get("id", ""),
            "display_name": p.get("display_name", ""),
            "aliases": p.get("aliases", []),
            "notes": p.get("notes", ""),
            "face_count": len(p.get("face_ids", [])),
        }
        for p in data.get("people", [])
    ]


@mcp.tool()
def cast_list_reviews(
    status_filter: Optional[str] = None,
    limit: int = 50,
    store_dir: str = CAST_STORE_DEFAULT,
) -> list[dict]:
    """List face review queue items from the Cast store.

    Args:
        status_filter: Filter by status: 'pending', 'resolved', or 'ignored'. Omit for all.
        limit: Maximum number of items to return (default 50).
        store_dir: Path to cast data directory.
    """
    review_path = Path(store_dir) / "review_queue.jsonl"
    if not review_path.exists():
        return []
    items = []
    for line in review_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if status_filter and item.get("status") != status_filter:
            continue
        item.pop("embedding", None)
        items.append(item)
        if len(items) >= limit:
            break
    return items


# ── Cast: job-launching tools ──────────────────────────────────────────────────


@mcp.tool()
def cast_start_web(host: str = "0.0.0.0", port: int = 8093) -> dict:
    """Start the Cast face review web UI. Returns a job ID.

    The server runs until cancelled with job_cancel().

    Args:
        host: Bind host (default 0.0.0.0).
        port: Bind port (default 8093).
    """
    args = [PYTHON, CAST_SCRIPT, "web", "--host", host, "--port", str(port)]
    return _job_started(runner.start("cast_web_ui", args))


# ── Photoalbums: read-only data tools ─────────────────────────────────────────


@mcp.tool()
def photoalbums_manifest_summary(manifest_path: str = MANIFEST_DEFAULT) -> dict:
    """Summarise the AI index manifest: image counts grouped by processing state.

    Args:
        manifest_path: Path to ai_index_manifest.jsonl.
    """
    p = Path(manifest_path)
    if not p.exists():
        return {"error": "Manifest not found", "path": str(p)}
    states: dict[str, int] = {}
    total = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            state = rec.get("state", "unknown")
            states[state] = states.get(state, 0) + 1
            total += 1
        except json.JSONDecodeError:
            pass
    return {"total": total, "by_state": states}


# ── Photoalbums: job-launching tools ──────────────────────────────────────────


@mcp.tool()
def photoalbums_ai_index(
    photos_root: str = PHOTOS_ROOT_DEFAULT,
    cast_store: str = CAST_STORE_DEFAULT,
    caption_engine: str = "lmstudio",
    ocr_engine: str = "lmstudio",
    force: bool = False,
    disable_people: bool = False,
    disable_objects: bool = False,
    disable_ocr: bool = False,
    geocode_skip: bool = False,
    include_view: bool = False,
    max_images: int = 0,
    dry_run: bool = False,
    extra_args: Optional[list[str]] = None,
) -> dict:
    """Start a photoalbums AI indexing job (people → objects → OCR → captions → geocoding → XMP).

    This is a long-running operation. Returns a job ID immediately.
    Use job_status(job_id) to monitor progress and job_logs(job_id) for full output.

    Args:
        photos_root: Root directory of photos to process.
        cast_store: Cast data dir for people face-matching.
        caption_engine: Caption engine: 'qwen' or 'lmstudio'.
        ocr_engine: OCR engine: 'qwen' or 'lmstudio'.
        force: Ignore manifest and re-process all images.
        disable_people: Skip people identification step.
        disable_objects: Skip YOLO object detection step.
        disable_ocr: Skip OCR text extraction step.
        geocode_skip: Skip geocoding step.
        include_view: Also process files in *_View folders.
        max_images: Limit number of images to process (0 = unlimited).
        dry_run: Preview operations without writing any files.
        extra_args: Additional raw CLI arguments (e.g. ['--verbose']).
    """
    args = [
        PYTHON,
        PHOTOALBUMS_SCRIPT,
        "ai",
        "index",
        "--photos-root",
        photos_root,
        "--cast-store",
        cast_store,
        "--caption-engine",
        caption_engine,
        "--ocr-engine",
        ocr_engine,
    ]
    if force:
        args.append("--force")
    if disable_people:
        args.append("--disable-people")
    if disable_objects:
        args.append("--disable-objects")
    if disable_ocr:
        args.append("--disable-ocr")
    if geocode_skip:
        args.append("--geocode-skip-none")
    if include_view:
        args.append("--include-view")
    if max_images:
        args += ["--max-images", str(max_images)]
    if dry_run:
        args.append("--dry-run")
    if extra_args:
        args.extend(extra_args)

    name = f"photoalbums_ai_index:{Path(photos_root).name}"
    return _job_started(runner.start(name, args))


@mcp.tool()
def photoalbums_compress(photos_root: str = PHOTOS_ROOT_DEFAULT) -> dict:
    """Start a job to compress TIFF scans in-place. Returns a job ID.

    Args:
        photos_root: Root directory containing TIFF scans.
    """
    args = [PYTHON, PHOTOALBUMS_SCRIPT, "compress", "--photos-root", photos_root]
    return _job_started(
        runner.start(f"photoalbums_compress:{Path(photos_root).name}", args)
    )


@mcp.tool()
def photoalbums_stitch(
    photos_root: str = PHOTOS_ROOT_DEFAULT, validate_only: bool = False
) -> dict:
    """Start a job to stitch album page outputs. Returns a job ID.

    Args:
        photos_root: Root directory containing scans.
        validate_only: Only validate stitchability without writing outputs.
    """
    subcommand = "validate" if validate_only else "build"
    args = [
        PYTHON,
        PHOTOALBUMS_SCRIPT,
        "stitch",
        subcommand,
        "--photos-root",
        photos_root,
    ]
    return _job_started(
        runner.start(f"photoalbums_stitch_{subcommand}:{Path(photos_root).name}", args)
    )


# ── VHS: job-launching tools ───────────────────────────────────────────────────


@mcp.tool()
def vhs_start_tuner(host: str = "0.0.0.0", port: int = 8092) -> dict:
    """Start the VHS Tuner web UI. Returns a job ID.

    The server runs until cancelled with job_cancel().

    Args:
        host: Bind host (default 0.0.0.0).
        port: Bind port (default 8092).
    """
    args = [PYTHON, VHS_SCRIPT, "tuner", "--host", host, "--port", str(port)]
    return _job_started(runner.start("vhs_tuner_ui", args, cwd=VHS_DIR))


@mcp.tool()
def vhs_convert_avi(files: list[str]) -> dict:
    """Start a job to convert AVI capture files to lossless archive MKV. Returns a job ID.

    Args:
        files: List of AVI file paths to convert.
    """
    args = [PYTHON, VHS_SCRIPT, "convert", "avi"] + files
    label = ", ".join(Path(f).name for f in files[:3])
    return _job_started(runner.start(f"vhs_convert_avi:{label}", args, cwd=VHS_DIR))


@mcp.tool()
def vhs_convert_umatic(files: list[str]) -> dict:
    """Start a job to convert U-matic/ProRes MOV files to archive MKV. Returns a job ID.

    Args:
        files: List of MOV (or similar) file paths to convert.
    """
    args = [PYTHON, VHS_SCRIPT, "convert", "umatic"] + files
    label = ", ".join(Path(f).name for f in files[:3])
    return _job_started(runner.start(f"vhs_convert_umatic:{label}", args, cwd=VHS_DIR))


@mcp.tool()
def vhs_generate_proxies(frame_number: Optional[int] = None) -> dict:
    """Start a job to generate proxy MP4 files (half-resolution previews). Returns a job ID.

    Args:
        frame_number: Extract a specific frame number instead of full proxy generation.
    """
    args = [PYTHON, VHS_SCRIPT, "proxy"]
    if frame_number is not None:
        args += ["--frame-number", str(frame_number)]
    return _job_started(runner.start("vhs_generate_proxies", args, cwd=VHS_DIR))


@mcp.tool()
def vhs_metadata_build() -> dict:
    """Start a job to generate VHS archive metadata outputs and checksums. Returns a job ID."""
    args = [PYTHON, VHS_SCRIPT, "metadata", "build"]
    return _job_started(runner.start("vhs_metadata_build", args, cwd=VHS_DIR))


@mcp.tool()
def vhs_render(render_args: Optional[list[str]] = None) -> dict:
    """Start a full VHS delivery render pipeline job. Returns a job ID.

    Args:
        render_args: Additional arguments (e.g. ['--archive', 'MyArchive', '--title', 'Chapter1']).
    """
    args = [PYTHON, VHS_SCRIPT, "render"]
    if render_args:
        args.extend(render_args)
    return _job_started(runner.start("vhs_render", args, cwd=VHS_DIR))


@mcp.tool()
def vhs_generate_subtitles(
    archive: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """Start a job to generate subtitle sidecars via Whisper transcription. Returns a job ID.

    Args:
        archive: Archive name (metadata/<archive>).
        title: Chapter title.
    """
    args = [PYTHON, VHS_SCRIPT, "subtitles"]
    if archive:
        args += ["--archive", archive]
    if title:
        args += ["--title", title]
    return _job_started(runner.start("vhs_subtitles", args, cwd=VHS_DIR))


@mcp.tool()
def vhs_generate_comparison(
    archive: str,
    title: str,
    extra_args: Optional[list[str]] = None,
) -> dict:
    """Start a job to generate a side-by-side original vs. processed comparison video. Returns a job ID.

    Args:
        archive: Archive name.
        title: Chapter title.
        extra_args: Additional CLI arguments.
    """
    args = [PYTHON, VHS_SCRIPT, "compare", "--archive", archive, "--title", title]
    if extra_args:
        args.extend(extra_args)
    return _job_started(
        runner.start(f"vhs_compare:{archive}/{title}", args, cwd=VHS_DIR)
    )


@mcp.tool()
def vhs_verify_archive(
    manifest: Optional[str] = None,
    algorithm: str = "sha3",
) -> dict:
    """Start a job to verify archive checksums. Returns a job ID.

    Args:
        manifest: Path to manifest file (uses default if omitted).
        algorithm: Checksum algorithm: 'sha3' or 'blake3'.
    """
    args = [PYTHON, VHS_SCRIPT, "verify", "archive"]
    if manifest:
        args.append(manifest)
    args.append("--sha3" if algorithm == "sha3" else "--blake3")
    label = Path(manifest).name if manifest else "default"
    return _job_started(runner.start(f"vhs_verify_archive:{label}", args, cwd=VHS_DIR))


@mcp.tool()
def vhs_people_prefill(
    archive: str, chapter: str, cast_store: str = CAST_STORE_DEFAULT
) -> dict:
    """Start a job to prefill people metadata for a VHS chapter from the Cast store. Returns a job ID.

    Args:
        archive: Archive name (metadata/<archive>).
        chapter: Exact chapter title.
        cast_store: Path to cast data directory.
    """
    args = [
        PYTHON,
        VHS_SCRIPT,
        "people",
        "prefill-cast",
        "--archive",
        archive,
        "--chapter",
        chapter,
        "--cast-store",
        cast_store,
    ]
    return _job_started(
        runner.start(f"vhs_people_prefill:{archive}/{chapter}", args, cwd=VHS_DIR)
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Imago MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport: stdio (Claude Code), sse (legacy SSE at /sse), http (streamable-HTTP at /mcp)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8090, help="Bind port (default: 8090)"
    )
    parser.add_argument(
        "--console-host",
        default=None,
        help="Advertised hostname for job console URLs returned to clients (e.g. 192.168.4.26)",
    )
    args = parser.parse_args()

    CONSOLE_HOST = (
        args.console_host or args.host
    )  # noqa: F841 - read via module globals
    if args.transport != "stdio":
        start_console(runner, host=args.host, port=CONSOLE_PORT)
        print(f"Job console:     http://{CONSOLE_HOST}:{CONSOLE_PORT}", file=sys.stderr)

    if args.transport in ("sse", "http"):
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.settings.transport_security = None  # allow connections from any host

    if args.transport == "sse":
        print(
            f"MCP SSE server:  http://{args.host}:{args.port}/sse  (configure LM Studio with this URL)",
            file=sys.stderr,
        )
        mcp.run(transport="sse")
    elif args.transport == "http":
        print(
            f"MCP HTTP server: http://{args.host}:{args.port}/mcp  (configure LM Studio with this URL)",
            file=sys.stderr,
        )
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
