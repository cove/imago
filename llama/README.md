# llama.cpp + Gemma 4

This folder keeps the native `llama.cpp` startup config for the Gemma 4 GGUF deployment together.

## First-time setup

1. Install `llama.cpp`:
   - Windows: `winget install --id ggml.llamacpp --exact`
   - macOS: `brew install llama.cpp`
2. Copy `.env.example` to `.env`.
3. Set `MODEL_DIR` to the folder containing:
   - `gemma-4-31B-it-Q4_K_M.gguf`
   - `mmproj-gemma-4-31B-it-BF16.gguf`
4. Start the server:
   - Project entrypoint: `just llama-gemma4`
   - Windows: `.\start.ps1`
   - macOS: `bash ./start.sh`
5. Use the OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1`.

The default alias stays `mlx-community/gemma-4-e2b-it-4bit` because the photo pipeline already requests that model name in `photoalbums/ai_models.toml`.
