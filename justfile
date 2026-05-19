set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]
python := "uv run python"
ruff := "uv run ruff"
photoalbums_root := `uv run python -c "from photoalbums.common import get_photo_albums_dir; print(get_photo_albums_dir())"`
llama_install := if os() == "windows" {
  "if (Get-Command llama-server -ErrorAction SilentlyContinue) { exit 0 }; winget list --id ggml.llamacpp --exact | Out-Null; if ($LASTEXITCODE -eq 0) { exit 0 }; winget install --id ggml.llamacpp --exact --accept-package-agreements --accept-source-agreements"
} else if os() == "macos" {
  "brew install llama.cpp"
} else {
  "echo \"llama-gemma4 only supports Windows and macOS\" && exit 1"
}
llama_start := if os() == "windows" {
  "pwsh -File llama/start.ps1"
} else if os() == "macos" {
  "bash llama/start.sh"
} else {
  "echo \"llama-gemma4 only supports Windows and macOS\" && exit 1"
}

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

llama-install:
  {{llama_install}}

llama-gemma4: llama-install
  {{llama_start}}

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

# Trigger Immich DB backup via API, pull it out of the container, compress with xz.
# Puts Immich into maintenance mode (all background jobs paused) for the duration.
immich-backup *args:
  just --justfile immich/justfile backup {{args}}
