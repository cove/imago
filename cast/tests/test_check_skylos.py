from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_skylos.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_skylos", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(payload: dict, *, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["skylos"],
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr=stderr,
    )


def test_check_skylos_runs_each_project_separately(monkeypatch):
    module = _load_module()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _result({"quality": []})

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.main([]) == 0
    assert [cmd[-1] for cmd in calls] == ["photoalbums", "vhs", "cast"]
    assert all("--json" in cmd for cmd in calls)
    assert all("--quality" in cmd for cmd in calls)
    assert all(len(cmd) == 5 for cmd in calls)


def test_check_skylos_fails_only_on_clone_findings(monkeypatch, capsys):
    module = _load_module()
    payloads = {
        "photoalbums": {"quality": [{"rule_id": "SKY-U005", "message": "unused dep"}]},
        "vhs": {
            "quality": [
                {
                    "rule_id": "SKY-C401",
                    "basename": "render.py",
                    "line": 42,
                    "message": "Clone group detected",
                }
            ]
        },
        "cast": {"quality": []},
    }

    def fake_run(cmd, **kwargs):
        return _result(payloads[cmd[-1]])

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.main([]) == 1

    err = capsys.readouterr().err
    assert "[skylos] vhs: found 1 duplicate-code finding(s)" in err
    assert "render.py:42 Clone group detected" in err
    assert "photoalbums" not in err
