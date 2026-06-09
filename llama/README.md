# llama.cpp + Gemma 4

This folder keeps the native `llama.cpp` startup config for the Gemma 4 GGUF deployment together.

## First-time setup

1. Install `llama.cpp`:
   - Windows: `winget install --id ggml.llamacpp --exact`
   - macOS: `brew install llama.cpp`
2. Copy `.env.example` to `.env`.
3. Set `MODEL_DIR` to the folder containing:
   - `gemma-4-31B-it-QAT-Q4_0.gguf`
   - `mmproj-gemma-4-31B-it-QAT-BF16.gguf`
4. Start the server:
   - Project entrypoint: `just llama-gemma4`
   - Windows: `.\start.ps1`
   - macOS: `bash ./start.sh`
5. Use the OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1`.

The alias is set to `google/gemma-4-31b-qat` to match the model name the photo pipeline requests in `photoalbums/ai_models.toml` (`pc`). llama.cpp serves the single loaded GGUF regardless of the request's `model` field, so the alias is primarily for a consistent `/v1/models` id.
