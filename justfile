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
  {{python}} photoalbums.py metadata map "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" --port 8095

photoalbums-ai:
  {{python}} photoalbums.py ai

photoalbums-ai-gps:
  {{python}} photoalbums.py ai gps

photoalbums-render:
  {{python}} photoalbums.py render

photoalbums-render-validate:
  {{python}} photoalbums.py render validate

photoalbums-detect-regions *args:
  {{python}} photoalbums.py detect-view-regions --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-crop-regions album="" *args:
  {{python}} photoalbums.py crop-regions {{if album != "" { '"' + album + '"' } else { "" }}} --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-repair-crop-source album="" *args:
  {{python}} photoalbums.py repair-crop-source {{if album != "" { '"' + album + '"' } else { "" }}} --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-ctm-generate *args:
  {{python}} photoalbums.py ctm generate --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-ctm-review *args:
  {{python}} photoalbums.py ctm review --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-ctm-apply *args:
  {{python}} photoalbums.py ctm-apply --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-render-pipeline *args:
  {{python}} photoalbums.py render-pipeline --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums"

photoalbums-watch:
  {{python}} photoalbums.py watch
  
