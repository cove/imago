set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
python := "uv run python"
ruff := "uv run ruff"

[default]
default:
  @just --list

sync:
  uv sync --locked

bootstrap:
  {{python}} scripts/bootstrap_runtime.py

test:
  .\scripts\test.ps1 -q

test-cast:
  .\cast\scripts\test.ps1 -q

test-photoalbums:
  .\photoalbums\scripts\test.ps1 -q

test-vhs:
  .\vhs\scripts\test.ps1 -q

format:
  {{ruff}} format .

lint:
  {{python}} scripts/check_ruff.py

typecheck:
  {{python}} scripts/check_pyright.py

dupes:
  {{python}} scripts/check_skylos.py

complexity:
  {{python}} scripts/check_radon.py

quality:
  {{python}} scripts/check_ruff.py
  {{python}} scripts/check_pyright.py
  {{python}} scripts/check_skylos.py

cast-init:
  {{python}} cast.py init

cast-web:
  {{python}} cast.py web

vhs-tuner:
  {{python}} vhs/vhs.py tuner

mcp-http:
  {{python}} mcp_server.py --transport http --host 0.0.0.0 --port 8090 --console-host 192.168.4.26

photoalbums-ai:
  {{python}} photoalbums.py ai
