# AGENTS.md

Purpose: repository-wide operating rules for AI coding agents working on this project.

## Core Policy

Use `just quality` and `just test` to validate code changes.
Prefer reconstructing state from ground truth rather than storing it in a database.
Prefer fewer files with clear internal structure over many small files. A single module with logical sections is better than splitting into multiple files that only have one caller. Only create a new module when it has a genuinely distinct responsibility and will be imported from multiple places.
Always bubble up underlying errors when reporting failures. Do not interpret or discard low-level errors.
Do not add docstrings, comments, or type annotations to code you did not change. Only add comments where the logic is not self-evident.
Do not add error handling, fallbacks, or validation for scenarios that cannot happen. Trust internal code and framework guarantees. Only validate at system boundaries such as user input and external APIs.
Do not create helpers, utilities, or abstractions for one-time operations. Do not design for hypothetical future requirements. Keep the implementation as simple as the task allows.

### Rule 1 — Think Before Coding

No silent assumptions. State what you're assuming. Surface tradeoffs. Ask before guessing. Push back when a simpler approach exists.

### Rule 2 — Simplicity First

Minimum code that solves the problem. No speculative features. No abstractions for single-use code. If a senior engineer would call it overcomplicated — simplify.

### Rule 3 — Surgical Changes

Touch only what you must. Don't "improve" adjacent code, comments, or formatting. Don't refactor what isn't broken. Match existing style.

### Rule 4 — Goal-Driven Execution

Define success criteria. Loop until verified. Don't tell Claude what steps to follow, tell it what success looks like and let it iterate.

### Rule 5 — Use the model only for judgment calls

Use AI for: classification, drafting, summarization, extraction from unstructured text.
Do NOT use AI for: routing, retries, status-code handling, deterministic transforms.
If a status code already answers the question, plain code answers the question.

### Rule 6 — Token budgets are not advisory

Per-task budget: 4,000 tokens.
Per-session budget: 30,000 tokens.
If a task is approaching budget, summarize and start fresh. Do not push through.
Surfacing the breach > silently overrunning.

### Rule 7 — Surface conflicts, don't average them

If two existing patterns in the codebase contradict, don't blend them.
Pick one (the more recent / more tested), explain why, and flag the other for cleanup.
"Average" code that satisfies both rules is the worst code.

### Rule 8 — Read before you write

Before adding code in a file, read the file's exports, the immediate caller, and any obvious shared utilities.
If you don't understand why existing code is structured the way it is, ask before adding to it.
"Looks orthogonal to me" is the most dangerous phrase in this codebase.

### Rule 9 — Tests verify intent, not just behavior

Every test must encode WHY the behavior matters, not just WHAT it does.
If you can't write a test that would fail when business logic changes, the function is wrong.

### Rule 10 — Checkpoint after every significant step

After completing each step in a multi-step task: summarize what was done, what's verified, what's left.
Don't continue from a state you can't describe back to me.
If you lose track, stop and restate.

### Rule 11 — Match the codebase's conventions, even if you disagree

If the codebase uses snake_case and you'd prefer camelCase: snake_case.
If the codebase uses class-based components and you'd prefer hooks: class-based.
Disagreement is a separate conversation. Inside the codebase, conformance > taste.
If you genuinely think the convention is harmful, surface it. Don't fork it silently.

### Rule 12 — Fail loud

If you can't be sure something worked, say so explicitly.
"Migration completed" is wrong if 30 records were skipped silently.
"Tests pass" is wrong if you skipped any.
"Feature works" is wrong if you didn't verify the edge case I asked about.
Default to surfacing uncertainty, not hiding it.

## Python Environment

For Python commands in this repo, do not rely on PATH-resolved `python`.
To run ad-hoc Python scripts, use `uv run python -c`.
Use `uv sync` from the repo root to create or update the project environment.

## Powershell on Windows

Use `pwsh` in the PATH instead of `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`

## Git Hygiene

Do not revert unrelated local changes.
Commit only files related to the requested task unless asked otherwise.
Use clear commit messages that describe the behavior change.
Git hooks are configured to run tests. Do not use `--no-verify` to skip tests; fix the errors instead.
