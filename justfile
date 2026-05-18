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

llama-gemma4:
  /bin/zsh -lc 'set -o pipefail; llama-server -m "$HOME/.lmstudio/models/lmstudio-community/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf" --mmproj "$HOME/.lmstudio/models/lmstudio-community/gemma-4-31B-it-GGUF/mmproj-gemma-4-31B-it-BF16.gguf" --alias mlx-community/gemma-4-e2b-it-4bit --host 0.0.0.0 --port 8080 --parallel 1 --ctx-size 8192 --cache-ram 0 2>&1 | awk '\''/prompt eval time =/ || /^[[:space:]]*eval time =/ || tolower($0) ~ /error|failed|fatal/ { print }'\'''

cast-init:
  {{python}} -m cast init

cast-web:
  {{python}} -m cast web

# Pull named face bounding boxes from Immich and write them into local XMP sidecars.
# Requires IMMICH_URL and IMMICH_API_KEY env vars (or pass --immich-url / --api-key).
# Pass --dry-run to preview changes without writing.  Pass -v for verbose output.
cast-immich-sync *args:
  {{python}} -m cast immich-sync --photos-root "{{photoalbums_root}}" {{args}}

cast-immich-cast-import *args:
  {{python}} -m cast immich-cast-import --photos-root "{{photoalbums_root}}" {{args}}

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

# Scan pipeline using previous LM Studio (http://192.168.4.72:1234/v1)
photoalbums-scan-pipeline-prev album="":
  $env:LMSTUDIO_BASE_URL = "http://192.168.4.72:1234/v1"; {{python}} -m photoalbums watch {{if album != "" { "--album " + '"' + album + '"' } else { "" }}}

photoalbums-bennett-scan-pipeline:
  {{python}} -m photoalbums.bennett watch

photoalbums-checksums:
  {{python}} -m photoalbums.sha3_tree_hashes "{{photoalbums_root}}"

photoalbums-verify-checksums:
  {{python}} -m photoalbums.sha3_tree_hashes "{{photoalbums_root}}" --verify
