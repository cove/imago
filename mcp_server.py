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
from photoalbums.lib import album_sets
from photoalbums.lib.mcp_queries import (
    album_status as photoalbums_album_status_query,
    query_manifest_rows as photoalbums_query_manifest_rows,
    read_job_artifacts as photoalbums_read_job_artifacts,
    reprocess_audit as photoalbums_reprocess_audit_query,
)
from photoalbums.lib.xmp_review import (
    load_ai_xmp_review,
    resolve_ai_xmp_review_path,
)
from photoalbums.scanwatch import ScanWatchService

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")

CAST_STORE_DEFAULT = str(REPO_ROOT / "cast" / "data")
VHS_DIR = str(REPO_ROOT / "vhs")
VHS_SCRIPT = str(REPO_ROOT / "vhs" / "vhs.py")
CAST_SCRIPT = str(REPO_ROOT / "cast.py")
PHOTOALBUMS_SCRIPT = str(REPO_ROOT / "photoalbums.py")
PHOTOALBUMS_LMSTUDIO_BASE_URLS = (
    "http://192.168.4.72:1234",
    "http://192.168.4.21:1234",
)
PHOTOALBUMS_MULTI_WORKER_DEPRECATION = "workers > 1 is deprecated; prefer workers=1."

CONSOLE_PORT = 8091
CONSOLE_HOST = "localhost"  # overridden at startup by --console-host

mcp = FastMCP("imago")
runner = JobRunner()
scanwatch_services: dict[str, ScanWatchService] = {}


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


def _jobs_started(job_ids: list[str]) -> dict:
    if len(job_ids) == 1:
        return _job_started(job_ids[0])
    monitor = " ".join(
        f"Call job_status('{job_id}') / job_logs('{job_id}') for shard {idx + 1}." for idx, job_id in enumerate(job_ids)
    )
    payload = {
        "status": "started",
        "workers": len(job_ids),
        "child_job_ids": job_ids,
        "how_to_monitor": monitor,
    }
    payload["warning"] = PHOTOALBUMS_MULTI_WORKER_DEPRECATION
    return payload


def _resolve_ai_index_photo_path(photos_root: str, photo: Optional[str]) -> Optional[str]:
    if photo is None:
        return None
    if not isinstance(photo, str):
        raise ValueError("photo must be a single filename or path string")

    photo_value = photo.strip()
    if not photo_value:
        return None

    photo_path = Path(photo_value)
    if photo_path.is_absolute() or photo_path.parent != Path("."):
        return photo_value

    photos_root_path = Path(photos_root)
    if not photos_root_path.is_dir():
        raise ValueError(f"photos_root is not a directory: {photos_root}")

    target_name = photo_path.name.casefold()
    matches = sorted(
        path.resolve() for path in photos_root_path.rglob("*") if path.is_file() and path.name.casefold() == target_name
    )
    if not matches:
        raise ValueError(f"Photo filename '{photo_value}' was not found under photos_root '{photos_root}'.")
    if len(matches) > 1:
        joined = ", ".join(str(path) for path in matches[:10])
        if len(matches) > 10:
            joined += f", ... ({len(matches)} matches total)"
        raise ValueError(f"Photo filename '{photo_value}' is ambiguous under photos_root '{photos_root}': {joined}")
    return str(matches[0])


def _client_set_dict(album_set: album_sets.AlbumSet) -> dict[str, object]:
    return album_set.to_client_dict(
        default_archive_set=album_sets.default_archive_set_name(),
        default_scan_set=album_sets.default_scan_set_name(),
    )


def _archive_set(album_set: Optional[str] = None) -> album_sets.AlbumSet:
    return album_sets.resolve_archive_set(album_set)


def _scan_set(album_set: Optional[str] = None) -> album_sets.AlbumSet:
    return album_sets.resolve_scan_set(album_set)


def _scanwatch_service(album_set: Optional[str] = None) -> tuple[album_sets.AlbumSet, ScanWatchService]:
    set_config = _scan_set(album_set)
    service = scanwatch_services.get(set_config.name)
    if service is None:
        service = ScanWatchService(root=set_config.photos_root)
        scanwatch_services[set_config.name] = service
    return set_config, service


def _scanwatch_status_payload(album_set_name: str, status: dict[str, object]) -> dict[str, object]:
    return {
        "album_set": album_set_name,
        "running": status["running"],
        "event_count": status["event_count"],
        "pending_event_count": status["pending_event_count"],
        "needs_rescan_count": status["needs_rescan_count"],
        "archive_count": status["archive_count"],
    }


def _scanwatch_event_payload(album_set_name: str, event: dict[str, object]) -> dict[str, object]:
    archive = event.get("archive", {})
    return {
        "album_set": album_set_name,
        "id": event["id"],
        "status": event["status"],
        "target_name": event["target_name"],
        "page_num": event["page_num"],
        "note": event["note"],
        "archive_name": Path(str(event["archive_dir"])).name,
        "incoming_file": Path(str(event["incoming_path"])).name,
        "archive": {
            "archive_name": Path(str(archive.get("archive_dir", ""))).name if archive else "",
            "pending_event_ids": archive.get("pending_event_ids", []),
            "needs_rescan_pages": archive.get("needs_rescan_pages", []),
        },
    }


# ── Job management ─────────────────────────────────────────────────────────────


@mcp.tool()
def job_list() -> list[dict]:
    """List all background jobs and their status (newest first)."""
    return runner.list_jobs()


@mcp.tool()
def job_status(job_id: str) -> dict:
    """Get status metadata for a job.

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
def cast_list_people() -> list[dict]:
    """List all people in the Cast face identity store.

    Returns name, aliases, notes, and face count per person (no embeddings).
    """
    people_path = Path(CAST_STORE_DEFAULT) / "people.json"
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
) -> list[dict]:
    """List face review queue items from the Cast store.

    Args:
        status_filter: Filter by status: 'pending', 'resolved', or 'ignored'. Omit for all.
        limit: Maximum number of items to return (default 50).
    """
    review_path = Path(CAST_STORE_DEFAULT) / "review_queue.jsonl"
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
def photoalbums_list_sets(kind: Optional[str] = None) -> list[dict[str, object]]:
    """List configured album sets available to MCP clients.

    Args:
        kind: Optional filter: 'archive' or 'scanwatch'.

    Use the returned `album_set` value exactly when passing `album_set=...` to other tools.
    Do not pass the human-readable description.
    """
    return [_client_set_dict(album_set) for album_set in album_sets.list_album_sets(kind=kind)]


@mcp.tool()
def photoalbums_get_set(album_set: str) -> dict[str, object]:
    """Return client-safe metadata for one album set.

    Args:
        album_set: Exact short album-set value, for example 'cordell' or 'incoming_scans'.
    """
    return _client_set_dict(album_sets.get_album_set(album_set))


@mcp.resource("photoalbums://sets")
def photoalbums_sets_resource() -> str:
    """Markdown summary of configured album sets."""
    lines = ["# Photoalbums Album Sets", ""]
    for album_set in photoalbums_list_sets():
        lines += [
            f"## {album_set['album_set']}",
            f"- Kind: `{album_set['kind']}`",
            f"- Default: `{album_set['is_default']}`",
            f"- Skill: `{album_set['skill']}`",
            f"- Description: {album_set['description'] or '(none)'}",
            f'- Use in tool calls: `album_set="{album_set["album_set"]}"`',
            "",
        ]
    return "\n".join(lines)


@mcp.tool()
def photoalbums_manifest_summary(album_set: Optional[str] = None) -> dict:
    """Summarise AI index coverage: image counts by sidecar/processing state."""
    set_config = _archive_set(album_set)
    result = photoalbums_query_manifest_rows(
        photos_root=str(set_config.photos_root),
        limit=100000,
    )
    rows = result.get("rows") or []
    with_sidecar = sum(1 for r in rows if r.get("sidecar_present"))
    current = sum(1 for r in rows if r.get("sidecar_current"))
    processed = sum(1 for r in rows if r.get("processor_signature"))
    return {
        "album_set": set_config.name,
        "total_images": result.get("total_matches", 0),
        "with_sidecar": with_sidecar,
        "current_sidecars": current,
        "processed": processed,
    }


# ── Photoalbums: XMP review + job-launching tools ─────────────────────────────


@mcp.tool()
def photoalbums_manifest_query(
    album_set: Optional[str] = None,
    album: Optional[str] = None,
    file_name: Optional[str] = None,
    limit: int = 100,
) -> dict[str, object]:
    """Query photo status derived from XMP sidecars.

    Args:
        album: Filter to photos whose parent directory name contains this substring.
        file_name: Exact basename filter for a photo.
        limit: Maximum rows to return.
    """
    set_config = _archive_set(album_set)
    return photoalbums_query_manifest_rows(
        photos_root=str(set_config.photos_root),
        album=str(album or ""),
        file_name=str(file_name or ""),
        limit=limit,
    )


@mcp.tool()
def photoalbums_album_status(album: str, album_set: Optional[str] = None) -> dict[str, object]:
    """Return album coverage and cover-page readiness for a parent-directory filter.

    Args:
        album: Substring match against the parent directory name.
    """
    set_config = _archive_set(album_set)
    return photoalbums_album_status_query(
        photos_root=str(set_config.photos_root),
        album=album,
    )


@mcp.tool()
def photoalbums_job_artifacts(
    job_id: str,
    kind: Optional[str] = None,
    file_name: Optional[str] = None,
) -> dict[str, object]:
    """Read stored photoalbums job artifacts for a completed or running job.

    Args:
        job_id: Job ID returned by the MCP job runner.
        kind: Optional artifact kind filter.
        file_name: Optional exact basename filter against image_path/sidecar_path/label.
    """
    job = next((row for row in runner.list_jobs() if str(row.get("id")) == str(job_id)), None)
    if not isinstance(job, dict):
        raise ValueError(f"Job {job_id} not found")
    return photoalbums_read_job_artifacts(
        job=job,
        kind=str(kind or ""),
        file_name=str(file_name or ""),
    )


@mcp.tool()
def photoalbums_reprocess_audit(
    album_set: Optional[str] = None,
    album: Optional[str] = None,
    limit: int = 100,
) -> dict[str, object]:
    """List photo files that appear to need reprocessing based on file/XMP/manifest state.

    Args:
        album: Optional parent-directory substring filter.
        limit: Maximum rows to return.
    """
    set_config = _archive_set(album_set)
    return photoalbums_reprocess_audit_query(
        photos_root=str(set_config.photos_root),
        cast_store=str(set_config.cast_store or CAST_STORE_DEFAULT),
        album=str(album or ""),
        limit=limit,
    )


@mcp.tool()
def photoalbums_load_xmp(
    file_name: str,
    album_set: Optional[str] = None,
    include_raw_xml: bool = False,
) -> dict[str, object]:
    """Load a photoalbums AI XMP sidecar and return its stored fields for review.

    Args:
        file_name: Image filename/path or XMP filename/path. When an image is passed,
            the tool loads the matching .xmp sidecar.
        include_raw_xml: Include the raw XML sidecar text in the response.
    """
    set_config = _archive_set(album_set)
    sidecar_path, photo_path = resolve_ai_xmp_review_path(
        str(set_config.photos_root),
        file_name=file_name,
    )
    result = load_ai_xmp_review(sidecar_path, include_raw_xml=include_raw_xml)
    result["resolved_from"] = "photo" if photo_path is not None else "xmp_path"
    result["photo_path"] = str(photo_path) if photo_path is not None else None
    return result


@mcp.tool()
def photoalbums_ai_index(
    album_set: Optional[str] = None,
    photo: Optional[str] = None,
    reprocess_mode: str = "unprocessed",
    album: Optional[str] = None,
    max_images: int = 0,
    dry_run: bool = False,
    workers: int = 1,
    lmstudio_base_urls: Optional[list[str]] = None,
) -> dict:
    """Start a photoalbums AI indexing job (people → objects → OCR → captions → geocoding → XMP).

    This is a long-running operation. Returns job metadata immediately.
    For one worker, the response includes a single job_id. For multiple workers, it returns child_job_ids.
    Multi-worker runs are deprecated and kept only as temporary scaffolding.
    Use job_status(job_id) to monitor progress and job_logs(job_id) for full output.

    Before starting a large job, call photoalbums_reprocess_audit() to see reason_counts
    and choose the right reprocess_mode. Use dry_run=True to preview scope without writing.

    Args:
        photo: Optional filename (or full path) of a single photo to process.
            When omitted, the job processes all photos that match the other filters.
            A bare filename is searched under photos_root and always forces reprocessing.
        reprocess_mode: Controls which images are selected for processing.
            'unprocessed' (default): images with missing or stale sidecar — safe for normal
                runs, will not re-touch already-complete albums.
            'new_only': only images with no manifest entry (never indexed) — use this when
                resuming after a server restart to avoid re-doing completed work.
            'errors_only': only images whose sidecar contains a processing error
                (e.g. lmstudio_caption_error) — use to retry failed captions without
                reprocessing everything.
            'outdated': only images where the sidecar is older than the image file — use
                when source files have been updated.
            'cast_changed': only images that need people re-detection because the cast
                store has changed — targeted re-run after adding new people.
            'all': force reprocess every matching image regardless of state.
        album: Filter to photos whose parent directory name contains this substring (case-insensitive).
        max_images: Limit number of matching photos to process (0 = unlimited).
        dry_run: Preview what would be processed without writing any sidecar or manifest changes.
        workers: Deprecated when greater than 1. Start multiple shard jobs for the same request.
        lmstudio_base_urls: Optional per-worker LM Studio base URLs. When omitted, photoalbums
            jobs default to the configured GLM servers. Must be empty, a single URL, or one URL
            per worker.
    """
    valid_modes = {"unprocessed", "new_only", "errors_only", "outdated", "cast_changed", "all"}
    if reprocess_mode not in valid_modes:
        raise ValueError(f"reprocess_mode must be one of: {', '.join(sorted(valid_modes))}")
    workers = int(workers or 1)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    set_config = _archive_set(album_set)
    args = [
        PYTHON,
        PHOTOALBUMS_SCRIPT,
        "ai",
        "index",
        "--photos-root",
        str(set_config.photos_root),
        "--cast-store",
        str(set_config.cast_store or CAST_STORE_DEFAULT),
        "--reprocess-mode",
        reprocess_mode,
    ]
    if dry_run:
        args.append("--dry-run")
    resolved_photo = _resolve_ai_index_photo_path(str(set_config.photos_root), photo)
    if resolved_photo:
        if workers > 1:
            raise ValueError("workers > 1 is only supported when indexing multiple photos")
        args += ["--photo", resolved_photo]
    if album:
        args += ["--album", album]
    if max_images:
        args += ["--max-images", str(max_images)]

    urls = [str(url or "").strip() for url in list(lmstudio_base_urls or []) if str(url or "").strip()]
    if workers > 1 and not urls:
        urls = [str(url).strip() for url in PHOTOALBUMS_LMSTUDIO_BASE_URLS if str(url).strip()]
    if urls and len(urls) not in {1, workers}:
        raise ValueError("lmstudio_base_urls must be empty, a single URL, or exactly one URL per worker")

    name = f"photoalbums_ai_index:{set_config.name}"
    if workers == 1:
        worker_args = list(args)
        if urls:
            worker_args += ["--lmstudio-base-url", urls[0]]
        return _job_started(runner.start(name, worker_args))

    job_ids: list[str] = []
    for shard_index in range(workers):
        worker_args = list(args)
        worker_args += ["--shard-count", str(workers), "--shard-index", str(shard_index)]
        if urls:
            worker_args += ["--lmstudio-base-url", urls[shard_index % len(urls)]]
        job_ids.append(runner.start(f"{name}[{shard_index + 1}/{workers}]", worker_args))
    return _jobs_started(job_ids)


@mcp.tool()
def photoalbums_compress(album_set: Optional[str] = None) -> dict:
    """Start a job to compress TIFF scans in-place. Returns a job ID."""
    set_config = _archive_set(album_set)
    args = [PYTHON, PHOTOALBUMS_SCRIPT, "compress", "--photos-root", str(set_config.photos_root)]
    return _job_started(runner.start(f"photoalbums_compress:{set_config.name}", args))


@mcp.tool()
def photoalbums_stitch(validate_only: bool = False, album_set: Optional[str] = None) -> dict:
    """Start a job to stitch album page outputs. Returns a job ID.

    Args:
        validate_only: Only validate stitchability without writing outputs.
    """
    set_config = _archive_set(album_set)
    subcommand = "validate" if validate_only else "build"
    args = [
        PYTHON,
        PHOTOALBUMS_SCRIPT,
        "stitch",
        subcommand,
        "--photos-root",
        str(set_config.photos_root),
    ]
    return _job_started(runner.start(f"photoalbums_stitch_{subcommand}:{set_config.name}", args))


@mcp.tool()
def photoalbums_generate_ctm(
    album_id: str,
    page: Optional[int] = None,
    force: bool = False,
    album_set: Optional[str] = None,
) -> dict:
    """Start a job to generate archive XMP CTM metadata for a photo album page or album."""
    set_config = _archive_set(album_set)
    args = [
        PYTHON,
        PHOTOALBUMS_SCRIPT,
        "ctm",
        "generate",
        "--album-id",
        str(album_id),
        "--photos-root",
        str(set_config.photos_root),
    ]
    if page is not None:
        args += ["--page", str(int(page))]
    if force:
        args.append("--force")
    return _job_started(runner.start(f"photoalbums_ctm_generate:{album_id}", args))


@mcp.tool()
def photoalbums_review_ctm(
    album_id: str,
    page: int,
    album_set: Optional[str] = None,
) -> dict:
    """Return stored archive XMP CTM metadata for a specific photo album page."""
    set_config = _archive_set(album_set)
    args = [
        PYTHON,
        PHOTOALBUMS_SCRIPT,
        "ctm",
        "review",
        "--album-id",
        str(album_id),
        "--page",
        str(int(page)),
        "--photos-root",
        str(set_config.photos_root),
    ]
    job_id = runner.start(f"photoalbums_ctm_review:{album_id}:p{int(page)}", args)
    return _job_started(job_id)


# ── Photoalbums: scan watcher control ──────────────────────────────────────────


@mcp.tool()
def scanwatch_start(album_set: Optional[str] = None) -> dict[str, object]:
    """Start or restart the incoming scan watcher."""
    set_config, service = _scanwatch_service(album_set)
    return _scanwatch_status_payload(set_config.name, service.start())


@mcp.tool()
def scanwatch_stop(album_set: Optional[str] = None) -> dict[str, object]:
    """Stop the incoming scan watcher."""
    set_config, service = _scanwatch_service(album_set)
    return _scanwatch_status_payload(set_config.name, service.stop())


@mcp.tool()
def scanwatch_status(album_set: Optional[str] = None) -> dict[str, object]:
    """Return watcher state and reconstruction summary."""
    set_config, service = _scanwatch_service(album_set)
    return _scanwatch_status_payload(set_config.name, service.status())


@mcp.tool()
def scanwatch_refresh(album_set: Optional[str] = None) -> dict[str, object]:
    """Rebuild in-memory watcher state from the filesystem."""
    set_config, service = _scanwatch_service(album_set)
    return _scanwatch_status_payload(set_config.name, service.rebuild())


@mcp.tool()
def scanwatch_list_events(
    status: Optional[str] = None,
    limit: int = 100,
    album_set: Optional[str] = None,
) -> list[dict[str, object]]:
    """List scan events known to the watcher."""
    set_config, service = _scanwatch_service(album_set)
    return [
        _scanwatch_event_payload(set_config.name, event) for event in service.list_events(status=status, limit=limit)
    ]


@mcp.tool()
def scanwatch_get_event(event_id: str, album_set: Optional[str] = None) -> dict[str, object]:
    """Return one scan event with its archive context."""
    set_config, service = _scanwatch_service(album_set)
    return _scanwatch_event_payload(set_config.name, service.get_event_context(event_id))


@mcp.tool()
def scanwatch_list_rescans(limit: int = 100, album_set: Optional[str] = None) -> list[dict[str, object]]:
    """List pages that currently need another scan."""
    set_config, service = _scanwatch_service(album_set)
    items = service.list_rescans(limit=limit)
    return [
        {
            "album_set": set_config.name,
            "archive_name": Path(str(item["archive_dir"])).name,
            "page_num": item["page_num"],
            "scan_count": item["scan_count"],
            "files": [Path(str(path)).name for path in item["files"]],
        }
        for item in items
    ]


@mcp.tool()
def scanwatch_apply_decision(
    event_id: str,
    target_name: str,
    validate_stitch: bool = True,
    open_preview: bool = True,
    album_set: Optional[str] = None,
) -> dict[str, object]:
    """Rename, process, and optionally validate a scan event."""
    set_config, service = _scanwatch_service(album_set)
    result = service.apply_decision(
        event_id,
        target_name,
        validate_stitch=validate_stitch,
        open_preview=open_preview,
    )
    event = _scanwatch_event_payload(set_config.name, result["event"])
    archive = result.get("archive", {})
    event["archive"] = {
        "archive_name": Path(str(archive.get("archive_dir", ""))).name if archive else "",
        "pending_event_ids": archive.get("pending_event_ids", []),
        "needs_rescan_pages": archive.get("needs_rescan_pages", []),
    }
    return event


@mcp.resource("scanwatch://status")
def scanwatch_status_resource() -> str:
    """Text summary of current watcher state."""
    status = scanwatch_status()
    lines = [
        "# Scanwatch Status",
        "",
        f"- Album set: `{status['album_set']}`",
        f"- Running: `{status['running']}`",
        f"- Events: `{status['event_count']}`",
        f"- Pending: `{status['pending_event_count']}`",
        f"- Needs rescan: `{status['needs_rescan_count']}`",
    ]
    return "\n".join(lines)


@mcp.resource("scanwatch://events")
def scanwatch_events_resource() -> str:
    """Markdown list of known scan events."""
    events = scanwatch_list_events(limit=500)
    if not events:
        return "No scan events found."
    lines = ["# Scan Events", ""]
    for event in events:
        lines += [
            f"## {event['id']}",
            f"- Album set: `{event['album_set']}`",
            f"- Status: `{event['status']}`",
            f"- Archive: `{event['archive_name']}`",
            f"- Incoming: `{event['incoming_file']}`",
            f"- Target: `{event['target_name']}`",
            f"- Page: `{event['page_num']}`",
            f"- Note: `{event['note']}`",
            "",
        ]
    return "\n".join(lines)


@mcp.resource("scanwatch://event/{event_id}")
def scanwatch_event_resource(event_id: str) -> str:
    """Markdown summary for one scan event."""
    event = scanwatch_get_event(event_id)
    archive = event.get("archive", {})
    lines = [
        f"# Scan Event {event_id}",
        "",
        f"- Album set: `{event['album_set']}`",
        f"- Status: `{event['status']}`",
        f"- Archive: `{event['archive_name']}`",
        f"- Incoming: `{event['incoming_file']}`",
        f"- Target: `{event['target_name']}`",
        f"- Page: `{event['page_num']}`",
        f"- Note: `{event['note']}`",
        "",
        "## Archive Context",
        "",
        f"- Pending events: `{archive.get('pending_event_ids', [])}`",
        f"- Needs rescan pages: `{archive.get('needs_rescan_pages', [])}`",
    ]
    return "\n".join(lines)


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
    return _job_started(runner.start(f"vhs_compare:{archive}/{title}", args, cwd=VHS_DIR))


@mcp.tool()
def vhs_verify_archive(algorithm: str = "sha3") -> dict:
    """Start a job to verify archive checksums. Returns a job ID.

    Args:
        algorithm: Checksum algorithm: 'sha3' or 'blake3'.
    """
    args = [PYTHON, VHS_SCRIPT, "verify", "archive"]
    args.append("--sha3" if algorithm == "sha3" else "--blake3")
    return _job_started(runner.start("vhs_verify_archive", args, cwd=VHS_DIR))


@mcp.tool()
def vhs_get_render_settings(archive: str) -> dict:
    """Return the render_settings.json for a VHS archive, or an empty dict if none exists.

    Args:
        archive: Archive name (e.g. 'callahan_01_archive').
    """
    path = Path(VHS_DIR) / "metadata" / archive / "render_settings.json"
    if not path.exists():
        return {"archive": archive, "render_settings": None}
    return {"archive": archive, "render_settings": json.loads(path.read_text(encoding="utf-8"))}


@mcp.tool()
def vhs_get_chapters(archive: str) -> dict:
    """Return the chapters.tsv data for a VHS archive as a list of row dicts.

    Args:
        archive: Archive name (e.g. 'bennett_01_archive').
    """
    path = Path(VHS_DIR) / "metadata" / archive / "chapters.tsv"
    if not path.exists():
        return {"archive": archive, "chapters": None}
    import csv, io
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    return {"archive": archive, "chapters": list(reader)}


@mcp.tool()
def vhs_people_prefill(archive: str, chapter: str) -> dict:
    """Start a job to prefill people metadata for a VHS chapter from the Cast store. Returns a job ID.

    Args:
        archive: Archive name (metadata/<archive>).
        chapter: Exact chapter title.
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
        CAST_STORE_DEFAULT,
    ]
    return _job_started(runner.start(f"vhs_people_prefill:{archive}/{chapter}", args, cwd=VHS_DIR))


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
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8090, help="Bind port (default: 8090)")
    parser.add_argument(
        "--console-host",
        default=None,
        help="Advertised hostname for job console URLs returned to clients (e.g. 192.168.4.26)",
    )
    args = parser.parse_args()

    CONSOLE_HOST = args.console_host or args.host  # noqa: F841 - read via module globals
    try:
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
    finally:
        runner.shutdown()
