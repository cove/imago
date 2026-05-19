#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
set -a
source ./.env
set +a

log_file="${TMPDIR:-/tmp}/llama-startup.$$.log"
set +e
"${LLAMA_SERVER_BIN}" \
  --model "${MODEL_DIR}/${MODEL_FILE}" \
  --mmproj "${MODEL_DIR}/${MMPROJ_FILE}" \
  --alias "${MODEL_ALIAS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --parallel "${PARALLEL}" \
  --ctx-size "${CTX_SIZE}" \
  --cache-ram "${CACHE_RAM}" 2>&1 | tee "${log_file}"
status=${PIPESTATUS[0]}
set -e

if [ "${status}" -ne 0 ]; then
  exit "${status}"
fi

if grep -Eiq "failed to open GGUF|error loading model|failed to load model|exiting due to model loading error" "${log_file}"; then
  exit 1
fi
