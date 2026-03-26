---
name: photoalbums-scan-control
description: >-
  Control incoming photo album scan workflows through the scanwatch MCP server. Use when working on
  incoming_scan.tif handling, scan renaming, page retry after stitch failure, or album-specific scan
  naming rules that should live in skill text instead of Python code.
---

# Photoalbums Scan Control

## Overview

Use this skill to drive the scanwatch MCP server for incoming scans. The agent should inspect the
current event, choose the destination filename from the album-specific naming reference, apply the
decision, and handle retry after stitch failure without hard-coding naming rules in Python.

## Workflow

1. Call `scanwatch_status` to confirm the watcher is running.
2. Call `scanwatch_list_events` to find pending incoming scans.
3. Call `scanwatch_get_event` to inspect the archive context before naming.
4. Read `references/naming.md` for the current album naming rules.
5. Choose the target filename and call `scanwatch_apply_decision`.
6. If stitch validation fails, call `scanwatch_list_rescans` and keep scanning the same page until it validates.

## Naming Rules

- Do not hard-code naming conventions in Python.
- Use `references/naming.md` as the current source of truth for the album's filename pattern.
- If the album naming convention changes, update the reference file and the prompt, not the watcher.

## Failure Handling

- If stitch validation fails, report the page number, archive directory, and scan count.
- Do not advance to the next page until validation succeeds.
- Keep the failure actionable: the operator needs to scan another copy of the same page.

## MCP Tools

Use the `scanwatch_*` tools exposed by the unified `mcp_server.py`:

- `scanwatch_start`
- `scanwatch_stop`
- `scanwatch_status`
- `scanwatch_refresh`
- `scanwatch_list_events`
- `scanwatch_get_event`
- `scanwatch_list_rescans`
- `scanwatch_apply_decision`
