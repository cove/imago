set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
python := "uv run python"
ruff := "uv run ruff"
photoalbums_root := `uv run python -c "from photoalbums.common import get_photo_albums_dir; print(get_photo_albums_dir())"`

[default]
default:
  @just --list

sync:
  uv sync --locked

bootstrap:
  {{python}} scripts/bootstrap_runtime.py

test:
  {{python}} -m pytest "cast/tests" "photoalbums/tests" "vhs/test" "tests"

codecoverage:
  {{python}} -m pytest "cast/tests" "photoalbums/tests" "vhs/test" "tests" --cov=cast --cov=photoalbums --cov=vhs --cov=tests --cov-report=term-missing

evals:
   {{python}} -m pytest photoalbums/tests/test_rule_extraction_eval.py -v -m integration -s

format *paths:
  {{ruff}} format {{if paths != "" { paths } else { "." }}}

lint:
  {{python}} scripts/check_ruff.py

typecheck:
  {{python}} scripts/check_basedpyright.py

dupes:
  {{python}} scripts/check_skylos.py --duplicates-only

skylos:
  {{python}} scripts/check_skylos.py

deadcode:
  {{python}} scripts/check_vulture.py

complexity:
  {{ruff}} check --select C90 photoalbums vhs cast

quality: lint typecheck dupes skylos deadcode complexity

verification:
  uv run qodo

security:
  uv run dryrun

cast-init:
  {{python}} -m cast init

cast-web:
  {{python}} -m cast web

vhs-tuner:
  {{python}} vhs/vhs.py tuner

vhs-render *args:
  {{python}} vhs/vhs.py render {{args}}

vhs-subtitles archive="":
  {{python}} vhs/vhs.py subtitles {{if archive != "" { "--archive " + archive } else { "" }}}

vhs-metadata-build:
  {{python}} vhs/vhs.py metadata build

vhs-metadata-embed file="":
  {{python}} vhs/vhs.py metadata embed {{if file != "" { '"' + file + '"' } else { "" }}}

vhs-verify-archive archive="":
  {{python}} vhs/vhs.py verify archive {{if archive != "" { "--archive " + archive } else { "" }}}

mcp-http:
  {{python}} -m imago_mcp.server --transport http --host 0.0.0.0 --port 8090 --console-host 192.168.4.26

photoalbums-map:
  {{python}} -m photoalbums metadata map "{{photoalbums_root}}" --port 8095

photoalbums-list-render-pipeline-steps:
  {{python}} -m photoalbums process --photos-root "." --list-steps

photoalbums-render-pipeline *args:
  {{python}} -m photoalbums process --photos-root "{{photoalbums_root}}" {{args}}

photoalbums-list-scan-pipeline-steps:
  {{python}} -m photoalbums watch --list-steps

photoalbums-scan-pipeline album="":
  {{python}} -m photoalbums watch {{if album != "" { "--album " + '"' + album + '"' } else { "" }}}

photoalbums-bennett-scan-pipeline:
  {{python}} -m photoalbums.bennett watch
