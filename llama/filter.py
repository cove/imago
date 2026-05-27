#!/usr/bin/env python3
"""
Pipe llama-server --verbose output through this script.
  - W/E level lines  → printed in yellow/red
  - Completion JSON  → pretty-printed with color (large fields stripped)
  - Everything else  → suppressed
Exits 1 if a fatal startup error pattern was detected.
"""
import sys
import json
import re

# ANSI — disabled when stdout is not a tty (e.g. redirected to a file)
if sys.stdout.isatty():
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    YELLOW  = "\033[33m"
    GREEN   = "\033[32m"
    CYAN    = "\033[36m"
    MAGENTA = "\033[35m"
else:
    RESET = BOLD = DIM = RED = YELLOW = GREEN = CYAN = MAGENTA = ""

FATAL_PAT = re.compile(
    r"failed to open GGUF|error loading model|failed to load model|exiting due to model loading error",
    re.IGNORECASE,
)

# Large / unreadable fields to drop before display
STRIP_FROM_VERBOSE = {"generation_settings", "prompt"}


def colorize_json(text: str) -> str:
    out = []
    for line in text.splitlines():
        # Key
        line = re.sub(
            r'^(\s*)"([^"]+)":',
            lambda m: f'{m.group(1)}{CYAN}"{m.group(2)}"{RESET}:',
            line,
        )
        # String value
        line = re.sub(
            r'(:\s*)"(.*)"(,?)$',
            lambda m: f'{m.group(1)}{GREEN}"{m.group(2)}"{RESET}{m.group(3)}',
            line,
        )
        # Number value
        line = re.sub(
            r'(:\s*)(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(,?)$',
            lambda m: f'{m.group(1)}{YELLOW}{m.group(2)}{RESET}{m.group(3)}',
            line,
        )
        # Boolean / null
        line = re.sub(
            r'(:\s*)(true|false|null)(,?)$',
            lambda m: f'{m.group(1)}{MAGENTA}{m.group(2)}{RESET}{m.group(3)}',
            line,
        )
        out.append(line)
    return "\n".join(out)


def get_level(line: str) -> str:
    """Return D/I/W/E if line has a llama log prefix, else empty string."""
    parts = line.split(" ", 2)
    if len(parts) >= 2 and len(parts[1]) == 1 and parts[1] in "DIWE":
        return parts[1]
    return ""


def extract_json(line: str) -> dict | None:
    idx = line.find("{")
    if idx < 0:
        return None
    try:
        return json.loads(line[idx:])
    except json.JSONDecodeError:
        return None


def print_response(data: dict) -> None:
    content = None
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        pass

    usage   = data.get("usage")
    verbose = data.get("__verbose") or {}
    timings = verbose.get("timings")

    if not (content or usage or timings):
        return

    print(f"{DIM}{'─' * 60}{RESET}", flush=True)

    if content:
        try:
            obj = json.loads(content)
            print(colorize_json(json.dumps(obj, indent=2)), flush=True)
        except (json.JSONDecodeError, ValueError):
            print(content, flush=True)

    if usage:
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        cached_str = f"  cached={cached}" if cached else ""
        print(
            f"{DIM}prompt={usage.get('prompt_tokens')}  "
            f"completion={usage.get('completion_tokens')}  "
            f"total={usage.get('total_tokens')}{cached_str}{RESET}",
            flush=True,
        )

    if timings:
        n   = timings.get("predicted_n", 0)
        tps = round(timings.get("predicted_per_second", 0), 1)
        ms  = round(timings.get("predicted_ms", 0))
        print(f"{DIM}{n} tokens @ {tps}/s  ({ms} ms){RESET}", flush=True)


had_fatal = False

for raw in sys.stdin:
    line = raw.rstrip("\n")

    if FATAL_PAT.search(line):
        had_fatal = True

    level = get_level(line)

    if level == "E":
        print(f"{RED}{line}{RESET}", flush=True)
        continue
    if level == "W":
        print(f"{YELLOW}{line}{RESET}", flush=True)
        continue
    if level in ("I", "D"):
        data = extract_json(line)
        if data and "choices" in data:
            print_response(data)
        # all other I/D lines: suppress
        continue

    # No log prefix — show if it looks like a warning or error
    if re.search(r"\b(warn|error)\b", line, re.IGNORECASE):
        print(f"{YELLOW}{line}{RESET}", flush=True)

sys.exit(1 if had_fatal else 0)
