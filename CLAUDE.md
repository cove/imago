# AGENTS.md

Purpose: repository-wide operating rules for AI coding agents working on this project.

## Core Policy

- Prefer forward-only changes.
- Do not add backward compatibility by default.
- Do not add fall backs.
- When schema/config formats change, migrate all project data forward in the same change.
- If backward compatibility is requested, implement it only when explicitly asked.

## Data and Schema Migrations

- Treat `metadata/*/render_settings.json` and related metadata as migratable assets.
- When renaming keys or structures:
  - update readers/writers in code,
  - run migration across existing metadata files,
  - remove old keys/paths unless explicitly told to keep them.
- Keep a single canonical schema in code and on disk.

## UI and Terminology

- Prefer precise naming that matches behavior.
- Current preferred term: `Gamma Correction` (not `Brightness`).
- Keep user-facing labels aligned with metadata/renderer semantics.

## Preview Render Behavior

- Preview should apply only the current wizard stage effects by default:
  - Review step: bad-frame repair only.
  - Gamma Correction step: gamma correction only.
  - Summary step: combined output.

## Engineering Defaults

- Make focused, minimal changes.
- Avoid introducing extra abstraction unless it reduces real maintenance cost.
- Update docs/help text when behavior or naming changes.
- Validate changed Python modules with `python -m py_compile` when possible.

## Python Environment

- For Python commands in this repo, do not rely on PATH-resolved `python`.
- Use the repo virtualenv interpreter: `C:\Users\covec\Videos\imago\.venv\Scripts\python.exe`.
- From the repo root, prefer `.\.venv\Scripts\python.exe -m ...` for test, lint, and validation commands.
- Validate changed Python modules with `.\.venv\Scripts\python.exe -m py_compile` when possible.

## Git Hygiene

- Do not revert unrelated local changes.
- Commit only files related to the requested task unless asked otherwise.
- Use clear commit messages that describe behavior change.

## If Unsure

- Ask for a decision only when truly ambiguous.
- Otherwise choose the simplest forward-moving implementation consistent with these rules.

---

## MCP Server

The repo exposes a unified MCP server (`mcp_server.py`) registered via `.mcp.json`.
It surfaces all three projects as callable tools and provides a background job runner
for long-running operations.

### Starting the Server

The MCP server starts automatically when Claude Code opens this project (via `.mcp.json`).
The job console starts with it on port 8091.

To start manually:
```bash
.venv/Scripts/python.exe mcp_server.py
```

### Starting Jobs

Jobs are started by asking Claude to call an MCP tool. Examples:

> "Start an AI index job on D:/Albums"
> "Convert capture.avi to archive MKV"
> "Show me all running jobs"
> "Tail the logs for job a1b2c3d4"

Claude calls the appropriate tool (e.g. `photoalbums_ai_index`, `vhs_convert_avi`),
gets back a job ID, and you can monitor progress in the console at `http://0.0.0.0:8091`.

### Job Console

Open `http://localhost:8091` in a browser to see:
- All jobs with status, duration, and a live pulsing indicator for running jobs
- Log tail for any selected job (auto-refreshes every 2s)
- Cancel button for running jobs

### Registration

`.mcp.json` at the repo root registers the server with Claude Code automatically.
Restart Claude Code (or approve the server prompt) to activate it.

For Claude Desktop or other MCP clients, add to your config:

```json
{
  "mcpServers": {
    "imago": {
      "command": "C:\\Users\\covec\\Videos\\imago\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\covec\\Videos\\imago\\mcp_server.py"],
      "cwd": "C:\\Users\\covec\\Videos\\imago"
    }
  }
}
```

### Job Runner Pattern

Long-running tools return a **job ID** immediately. Use the job tools to track progress:

```
job_id = photoalbums_ai_index(photos_root="D:/Albums")
job_status(job_id)          # status + last 30 log lines
job_logs(job_id, 200)       # full log tail
job_cancel(job_id)          # terminate the process
job_list()                  # all jobs, newest first
```

Job logs are persisted to `mcp/jobs/<job_id>.log`. State survives server restarts
(jobs interrupted mid-run are marked `"interrupted"`).

---

### Cast Tools

| Tool | Type | Description |
|---|---|---|
| `cast_list_people` | immediate | List all people (name, aliases, notes, face count) |
| `cast_list_reviews` | immediate | List review queue items; filter by `status_filter` (`pending`/`resolved`/`ignored`) |
| `cast_start_web` | job | Start the Cast review web UI on port 8093 |

**Cast web UI** (`cast_start_web`): runs until cancelled with `job_cancel()`.
Open `http://localhost:8093` to review and assign face identities.

---

### Photoalbums Tools

| Tool | Type | Description |
|---|---|---|
| `photoalbums_manifest_summary` | immediate | Image counts grouped by processing state |
| `photoalbums_ai_index` | job | Run full AI pipeline (people → objects → OCR → captions → geocoding → XMP) |
| `photoalbums_compress` | job | Compress TIFF scans in-place |
| `photoalbums_stitch` | job | Build (or validate) stitched album page outputs |

**`photoalbums_ai_index` key parameters:**

| Parameter | Default | Notes |
|---|---|---|
| `photos_root` | *(required)* | Root directory to scan |
| `caption_engine` | `"qwen"` | `"qwen"` or `"lmstudio"` |
| `ocr_engine` | `"qwen"` | `"qwen"` or `"lmstudio"` |
| `force` | `false` | Re-process all images, ignoring manifest |
| `disable_people` | `false` | Skip Cast face-matching step |
| `disable_objects` | `false` | Skip YOLO object detection |
| `disable_ocr` | `false` | Skip OCR extraction |
| `geocode_skip` | `false` | Skip Nominatim geocoding |
| `max_images` | `0` | Limit images processed (0 = unlimited) |
| `dry_run` | `false` | Preview without writing files |
| `extra_args` | `null` | Raw CLI args, e.g. `["--verbose"]` |

---

### VHS Tools

| Tool | Type | Description |
|---|---|---|
| `vhs_start_tuner` | job | Start the VHS Tuner web UI on port 8092 |
| `vhs_convert_avi` | job | Convert AVI capture files to lossless archive MKV |
| `vhs_convert_umatic` | job | Convert U-matic/ProRes MOV files to archive MKV |
| `vhs_generate_proxies` | job | Generate half-resolution proxy MP4s |
| `vhs_metadata_build` | job | Generate archive metadata outputs and checksums |
| `vhs_render` | job | Run full delivery render pipeline |
| `vhs_generate_subtitles` | job | Generate subtitle sidecars via Whisper |
| `vhs_generate_comparison` | job | Render side-by-side original vs. processed video |
| `vhs_verify_archive` | job | Verify archive checksum manifest (`sha3` or `blake3`) |
| `vhs_people_prefill` | job | Prefill chapter people metadata from Cast store |

**VHS Tuner** (`vhs_start_tuner`): runs until cancelled with `job_cancel()`.
Open `http://localhost:8092` for the interactive frame-review and gamma-correction UI.

**Typical VHS pipeline sequence:**
```
vhs_convert_avi(files=["capture.avi"])          # → job_id
vhs_metadata_build()                            # → job_id
vhs_generate_proxies()                          # → job_id
vhs_start_tuner()                               # → job_id  (open browser, tune)
vhs_render(render_args=["--archive", "foo"])    # → job_id
vhs_generate_subtitles(archive="foo")           # → job_id
vhs_verify_archive()                            # → job_id
```
