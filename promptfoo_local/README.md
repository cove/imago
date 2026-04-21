# Promptfoo Local

Separate local `promptfoo` workspace for replaying photoalbums eval cases against LM Studio without modifying the production pipeline.

## Install

```powershell
cd C:\Users\covec\Videos\imago\promptfoo_local
npm install
```

## Run

Run all compatible eval JSON files under `photoalbums/evals`:

```powershell
npm run eval
```

Run only a specific eval by id or filename substring:

```powershell
$env:PROMPTFOO_EVAL_FILTER = "shared_location_date"
npm run eval
```

Open the local web UI after an eval run:

```powershell
npm run view
```

## How It Works

- `tests.py` loads eval JSON files from `photoalbums/evals`
- `provider.py` calls LM Studio directly, but imports helper code from `photoalbums.lib`
- prompts live in `promptfoo_local/prompts`
- assertions live in `promptfoo_local/assert.py`

## Prompt Variants

- `prompts/production.txt` replays the production prompt exactly
- `prompts/shared_middle_strict.txt` adds stricter instructions before the production prompt

Add more prompt files and list them in `promptfooconfig.yaml`.

## Provider Variants

Provider settings live in `promptfooconfig.yaml`.

Current examples:

- `gemma31b-greedy`
- `gemma31b-tp09-tk40`

Add more provider entries to compare models and generation settings side by side in the UI.
