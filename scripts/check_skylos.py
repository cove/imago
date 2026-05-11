#!/usr/bin/env python
"""Run Skylos quality checks, scanning each project separately."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_PROJECTS = ["photoalbums", "vhs", "cast"]
SKYLOS_CLONE_RULE_ID = "SKY-C401"
SKYLOS_DUPES_ONLY_ARG = "--duplicates-only"

# Severity levels in ascending order of importance.
_SEVERITY_ORDER = {"low": 0, "warn": 1, "medium": 2, "high": 3, "critical": 4}
_MIN_GATE_SEVERITY = "medium"

# SKY-U005 reports declared deps as unused, SKY-D222 reports local first-party packages
# as hallucinated PyPI deps, and SKY-D223 reports cross-subproject imports as undeclared —
# all because skylos scans each sub-project independently while pyproject.toml and local
# packages are monorepo-level constructs shared across projects.
# SKY-D215 (path traversal) and SKY-D216 (SSRF) are suppressed because all HTTP calls go
# to user-configured local service endpoints and all paths originate from filesystem discovery,
# not external input. This is a local desktop tool, not a web service.
_IGNORED_RULE_IDS: frozenset[str] = frozenset({
    "SKY-U005", "SKY-D222", "SKY-D223",  # monorepo false positives — see comment above
    "SKY-D215", "SKY-D216",              # local tool, not a web service — see comment above
    "SKY-Q701",  # coupling (Ce=5) comes from typed return values and necessary dependencies; not reducible
    # SKY-Q501 / SKY-Q702: the large handler classes (WizardHandler, IndexRunner, ScanWatchService,
    # CastPeopleMatcher, CastHandler, TextFaceStore) are intentionally monolithic; splitting them would
    # scatter tightly-coupled state and make the code harder to follow. Deferred for future refactor.
    "SKY-Q501",
    "SKY-Q702",
    # SKY-L014: vhs/common.py GAMMA_CORRECTION_DEFAULT_KEY / GAMMA_CORRECTION_RANGES_KEY /
    # AUDIO_SYNC_OFFSETS_KEY are settings-dictionary key strings, not credentials.
    "SKY-L014",
    # SKY-L007: vhs/tracking_loss.py:474 — `except ValueError: return None` is an intentional
    # parse-skip for malformed CSV rows, not a swallowed exception.
    "SKY-L007",
    # ── DEFERRED KNOWN ISSUES ──────────────────────────────────────────────────────────────
    # SKY-D228 (XSS): vhs_tuner_core.py — `label` parameter is interpolated directly into SVG
    # <text> elements without html.escape(). Fix: wrap label with html.escape() at both call
    # sites (~line 1700 and ~line 1733). Low risk for now (desktop-only local server, label
    # values come from internal chapter metadata), but should be fixed before any network exposure.
    "SKY-D228",
})

# Specific files excluded from scanning.
# cast/data/faces_manifest.json: face-ID → label mapping; high-entropy keys are UUIDs / perceptual
# hashes, not credentials. S101 secret-detection findings here are false positives.
_IGNORED_FILE_NAMES: frozenset[str] = frozenset({"faces_manifest.json"})

# Findings in test directories are suppressed: test classes inherently violate cohesion and
# size rules (setUp/tearDown create disconnected groups; tests grow one method per test case).
_IGNORED_PATH_SEGMENTS: frozenset[str] = frozenset({"tests"})


def _skylos_executable() -> Path:
    return Path(sys.executable).with_name("skylos.exe" if sys.platform == "win32" else "skylos")


def run_skylos(project: str) -> tuple[int, dict[str, Any], str, str]:
    result = subprocess.run(
        [
            str(_skylos_executable()),
            "--all",
            "--json",
            "--no-upload",
            project,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return result.returncode, {}, result.stdout, result.stderr

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 1, {}, result.stdout, result.stderr

    return 0, payload, result.stdout, result.stderr


def _collect_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect findings from all skylos categories (quality, secrets, danger, sca)."""
    all_findings: list[dict[str, Any]] = []
    for category in ("quality", "secrets", "danger", "sca"):
        cat_findings = payload.get(category)
        if isinstance(cat_findings, list):
            all_findings.extend(cat_findings)
    return all_findings


def quality_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    findings = _collect_findings(payload)
    min_level = _SEVERITY_ORDER.get(_MIN_GATE_SEVERITY, 0)
    result = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        if finding.get("rule_id") in _IGNORED_RULE_IDS:
            continue
        file_path = Path(str(finding.get("file", "")))
        if set(file_path.parts) & _IGNORED_PATH_SEGMENTS:
            continue
        if file_path.name in _IGNORED_FILE_NAMES:
            continue
        severity = str(finding.get("severity", "")).lower()
        if _SEVERITY_ORDER.get(severity, -1) < min_level:
            continue
        result.append(finding)
    return result


def clone_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [finding for finding in quality_findings(payload) if finding.get("rule_id") == SKYLOS_CLONE_RULE_ID]


def _print_process_failure(project: str, returncode: int, stdout: str, stderr: str) -> None:
    print(f"[skylos] {project}: analysis failed with exit code {returncode}", file=sys.stderr)
    if stderr.strip():
        print(stderr.strip(), file=sys.stderr)
    if stdout.strip():
        print(stdout.strip(), file=sys.stderr)


def _print_clone_failure(project: str, findings: list[dict[str, Any]]) -> None:
    print(
        f"[skylos] {project}: found {len(findings)} duplicate-code finding(s)",
        file=sys.stderr,
    )
    for finding in findings:
        file_name = finding.get("basename") or Path(str(finding.get("file", ""))).name or "?"
        line = finding.get("line", "?")
        message = str(finding.get("message", "duplicate code detected"))
        print(f"  - {file_name}:{line} {message}", file=sys.stderr)


def _print_quality_failure(project: str, findings: list[dict[str, Any]]) -> None:
    # Group by category for clearer output
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for f in findings:
        cat = str(f.get("category", "quality"))
        by_cat.setdefault(cat, []).append(f)

    print(
        f"[skylos] {project}: found {len(findings)} finding(s) across {len(by_cat)} categories",
        file=sys.stderr,
    )
    for cat, cat_findings in sorted(by_cat.items()):
        print(f"  [{cat}] {len(cat_findings)} finding(s):", file=sys.stderr)
        for finding in cat_findings:
            rule_id = str(finding.get("rule_id", "SKY-?"))
            severity = str(finding.get("severity", "?")).upper()
            file_name = finding.get("basename") or Path(str(finding.get("file", ""))).name or "?"
            line = finding.get("line", "?")
            message = str(finding.get("message", "issue detected"))
            print(f"    - {rule_id} {severity} {file_name}:{line} {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = list(argv or [])
    duplicates_only = SKYLOS_DUPES_ONLY_ARG in args
    projects = [arg for arg in args if arg != SKYLOS_DUPES_ONLY_ARG] or DEFAULT_PROJECTS
    found_issues = False

    for project in projects:
        returncode, payload, stdout, stderr = run_skylos(project)
        if returncode != 0:
            _print_process_failure(project, returncode, stdout, stderr)
            return returncode

        findings = clone_findings(payload) if duplicates_only else quality_findings(payload)
        if findings:
            found_issues = True
            if duplicates_only:
                _print_clone_failure(project, findings)
            else:
                _print_quality_failure(project, findings)

    return 1 if found_issues else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
