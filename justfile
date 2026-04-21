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
  {{python}} -m pytest "cast/tests" "photoalbums/tests" "vhs/test" 

evals:
   {{python}} -m pytest photoalbums/tests/test_rule_extraction_eval.py -v -m integration -s

format:
  {{ruff}} format .

lint:
  {{python}} scripts/check_ruff.py

typecheck:
  {{python}} scripts/check_pyright.py

dupes:
  {{python}} scripts/check_skylos.py

deadcode:
  {{python}} scripts/check_vulture.py

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
  {{python}} mcp_server.py --transport http --host 0.0.0.0 --port 8090 --console-host 192.168.4.26

photoalbums-map:
  {{python}} photoalbums.py metadata map "{{photoalbums_root}}" --port 8095

photoalbums-list-steps:
  {{python}} photoalbums.py process --photos-root "." --list-steps

photoalbums-render *args:
  {{python}} photoalbums.py process --photos-root "{{photoalbums_root}}" {{args}}

photoalbums-render-validate:
  {{python}} photoalbums.py render validate

photoalbums-watch:
  {{python}} photoalbums.py watch
  
