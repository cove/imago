#!/usr/bin/env python
"""Fail on Skylos duplicate-code findings, scanning each project separately."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_PROJECTS = ["photoalbums", "vhs", "cast"]
SKYLOS_CLONE_RULE_ID = "SKY-C401"


def _skylos_executable() -> Path:
    return Path(sys.executable).with_name("skylos.exe" if sys.platform == "win32" else "skylos")


def run_skylos(project: str) -> tuple[int, dict[str, Any], str, str]:
    result = subprocess.run(
        [
            str(_skylos_executable()),
            "--quality",
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


def clone_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    findings = payload.get("quality", [])
    if not isinstance(findings, list):
        return []
    return [
        finding for finding in findings if isinstance(finding, dict) and finding.get("rule_id") == SKYLOS_CLONE_RULE_ID
    ]


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
        file_name = finding.get("basename") or Path(str(finding.get("file", ""))).name
        line = finding.get("line", "?")
        message = str(finding.get("message", "duplicate code detected"))
        print(f"  - {file_name}:{line} {message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    projects = argv or DEFAULT_PROJECTS
    found_duplicates = False

    for project in projects:
        returncode, payload, stdout, stderr = run_skylos(project)
        if returncode != 0:
            _print_process_failure(project, returncode, stdout, stderr)
            return returncode

        findings = clone_findings(payload)
        if findings:
            found_duplicates = True
            _print_clone_failure(project, findings)

    return 1 if found_duplicates else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
