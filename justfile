set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

[default]
default:
  @just --list

sync:
  uv sync --locked

bootstrap:
  uv run python scripts/bootstrap_runtime.py

test:
  .\scripts\test.ps1 -q

test-cast:
  .\cast\scripts\test.ps1 -q

test-photoalbums:
  .\photoalbums\scripts\test.ps1 -q

test-vhs:
  .\vhs\scripts\test.ps1 -q

format:
  uv run ruff format .

lint:
  uv run python scripts/check_ruff.py

typecheck:
  uv run python scripts/check_pyright.py

dupes:
  uv run python scripts/check_skylos.py

complexity:
  uv run python scripts/check_radon.py

quality:
  uv run python scripts/check_ruff.py
  uv run python scripts/check_pyright.py
  uv run python scripts/check_skylos.py

cast-init:
  uv run python cast.py init

cast-web:
  uv run python cast.py web

vhs-tuner:
  uv run python vhs.py tuner

mcp-http:
  uv run python mcp_server.py --transport http --host 0.0.0.0 --port 8090 --console-host 192.168.4.26

photoalbums-ai:
  uv run python photoalbums.py ai
