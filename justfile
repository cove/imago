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

evals-compare:
  {{python}} -m pytest photoalbums/tests/test_rule_extraction_eval.py -v -m integration -s -k london-shared-location-date-direct

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

[working-directory: "promptfoo_local"]
promptfoo-eval eval_filter="shared_location_date":
  $env:PROMPTFOO_EVAL_FILTER='{{eval_filter}}'; npm run eval

[working-directory: "promptfoo_local"]
promptfoo-view:
   $env:PROMPTFOO_VIEW_HOST="0.0.0.0"; npm run view

vhs-metadata-embed file="":
  {{python}} vhs/vhs.py metadata embed {{if file != "" { '"' + file + '"' } else { "" }}}

vhs-verify-archive archive="":
  {{python}} vhs/vhs.py verify archive {{if archive != "" { "--archive " + archive } else { "" }}}

mcp-http:
  {{python}} mcp_server.py --transport http --host 0.0.0.0 --port 8090 --console-host 192.168.4.26

photoalbums-map:
  {{python}} photoalbums.py metadata map "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" --port 8095

photoalbums-process *args:
  {{python}} photoalbums.py process --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-steps:
  {{python}} photoalbums.py process --photos-root "." --list-steps

photoalbums-ai *args:
  {{python}} photoalbums.py process --step ai-index --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-ai-gps *args:
  {{python}} photoalbums.py process --step ai-index --gps-only --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-render *args:
  {{python}} photoalbums.py process --step render --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-render-validate:
  {{python}} photoalbums.py render validate

photoalbums-render-pipeline *args:
  {{python}} photoalbums.py process --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-detect-regions *args:
  {{python}} photoalbums.py process --step detect-regions --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-crop-regions *args:
  {{python}} photoalbums.py process --step crop-regions --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-ctm-review *args:
  {{python}} photoalbums.py ctm review --photos-root "C:\Users\covec\OneDrive\Cordell, Leslie & Audrey\Photo Albums" {{args}}

photoalbums-watch:
  {{python}} photoalbums.py watch
  
