# Architecture

This document explains how SCRIBE works internally and why each tool was chosen. Read this before modifying the pipeline.

## Pipeline

```
input audio file (mp3/wav/m4a/flac/ogg)
   │
   ▼
[1] CLI entry  (src/scribe/cli.py)
   │   typer-based; parses flags, loads Config
   │
   ▼
[2] Pipeline orchestration  (src/scribe/pipeline.py)
   │   - validates extension
   │   - skips if output exists (unless --force)
   │   - calls transcribe → summarize → render → writes file
   │
   ▼
[3] Transcription  (src/scribe/transcribe.py)
   │   ┌─ ffprobe                  → duration
   │   ├─ whisperx.load_audio()   → numpy array
   │   ├─ faster-whisper large-v3  → segments + detected language
   │   ├─ wav2vec2 forced alignment → word-level timestamps
   │   └─ pyannote v4 community-1   → speaker segments
   │           │
   │           └─ whisperx.assign_word_speakers → merged segments with .speaker
   │
   ▼
[4] Summarization  (src/scribe/summarize.py)
   │   joins all segment.text → POST to local Ollama /api/generate
   │   with language-specific prompt template (da / en)
   │
   ▼
[5] Rendering  (src/scribe/render.py)
   │   YAML frontmatter + "## Summary" + "## Transcript"
   │   speaker change → new line with bold label; continuation → just timestamp
   │
   ▼
output/<stem>.md
```

## Module reference

| Module | Responsibility |
|---|---|
| `cli.py` | Typer commands `transcribe` and `batch`. No business logic. |
| `config.py` | Loads `.env`, exposes a frozen `Config` dataclass. |
| `pipeline.py` | Orchestrates one file end-to-end; handles file discovery for batch mode. |
| `transcribe.py` | WhisperX wrapper. Returns `TranscriptResult(language, duration, segments[])`. |
| `summarize.py` | Single HTTP call to Ollama with a language-specific prompt. |
| `render.py` | Produces the final Markdown string. Pure function — no I/O. |

## Key design decisions

These are deliberate and documented so future maintainers don't relitigate them.

### Python, not Swift/Rust
All credible local ASR + diarization libraries are Python-native (WhisperX, faster-whisper, pyannote.audio). A Swift port would require re-implementing or wrapping these, which is more work than it's worth for a single-user CLI.

### `uv` for dependency management
Recommended by both WhisperX and pyannote 4. Markedly faster than pip and handles the PyTorch/torchaudio version pinning that breaks pip resolvers. The repo includes `uv.toml` with `system-certs = true` because corporate networks with MITM TLS will otherwise fail to download from PyPI.

### WhisperX over plain Whisper, whisper.cpp, or mlx-whisper
- **WhisperX** bundles ASR + word-level alignment + diarization integration in one library. Quality matches large-v3.
- **whisper.cpp** is faster on CPU but provides no diarization and only segment-level timestamps.
- **mlx-whisper** is faster on Apple Silicon (MLX/Metal) but has a smaller ecosystem and no integrated diarization — we'd need to wire pyannote in manually. Worth revisiting if speed becomes a blocker.

### pyannote v4 `speaker-diarization-community-1`
Current state-of-the-art open diarizer (DER 12.9% on AMI). The older `pyannote/speaker-diarization-3.1` is deprecated. The "community-1" model is free but gated — you must accept terms on HuggingFace once.

### Ollama + Gemma 4 for summarization
- **Ollama** is the de-facto local LLM runtime on macOS, with native MLX support for Apple Silicon.
- **Gemma 4 12B (MLX)** chosen over Gemma 2, Llama 3, Qwen 3:
  - 256K context (Gemma 2 only has 8K — would truncate long interviews)
  - MMLU-Pro 69.4% vs Gemma 2's 45% — much better structured reasoning
  - Explicit 140+ language training → safer bet for Danish than Llama/Qwen
  - MLX backend → 2–3× faster than GGUF on Apple Silicon

### CPU with int8, not MPS
WhisperX/faster-whisper on macOS work most reliably on CPU with int8 quantization. The MPS (Metal) backend in PyTorch is unstable for the operators these models use. CPU performance is acceptable (~1.7–2× realtime on M-series).

### Markdown output, not JSON/SRT/VTT
Markdown is human-readable, diff-friendly, and easy to paste into research notes, Notion, Obsidian etc. A future `--format json` flag would be a small addition if someone needs structured output for downstream tooling.

## External dependencies and gotchas

| Dependency | Notes |
|---|---|
| `ffmpeg` | Required at runtime. Used by both `ffprobe` (duration) and Whisper's audio loader. |
| `ollama` | Must be running as a service (`brew services start ollama`) and the configured model must be pulled. |
| `pyannote.audio` v4 | Requires HuggingFace token + manual license acceptance on the model page. Cannot be automated. |
| `torchcodec` | Pulled in transitively by pyannote. On macOS prints an rpath warning at import time but works fine (uses fallback). Harmless. |
| HuggingFace cache | All models live under `~/.cache/huggingface/hub/`. Safe to delete to free space; will re-download next run. |

## Testing strategy

There are no automated tests yet. The shape of a useful test suite:

- **Unit**: `render.py` is pure and easy to snapshot-test.
- **Integration**: ship a 30-second public-domain sample audio file and verify the pipeline runs end-to-end. Skip the model-quality assertions — those are not stable across model versions.
- **Smoke**: `uv run scribe --help` should exit 0; `Config.from_env()` should not raise.

## Where to extend

| Want to | Modify |
|---|---|
| Add a new audio format | `SUPPORTED_EXTENSIONS` in `pipeline.py` |
| Change the summary prompt | `_PROMPTS` dict in `summarize.py` |
| Change the output structure | `render.py` |
| Use a different ASR backend | Replace `transcribe()` body, keep the same return type (`TranscriptResult`) |
| Add a `resummarize` command | New typer command in `cli.py` that parses existing MD frontmatter+transcript and re-runs `summarize()` |
| Add a watch-folder mode | New typer command that polls `input/` and runs `process_file()` on new files |
