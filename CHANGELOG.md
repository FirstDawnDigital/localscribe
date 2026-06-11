# Changelog

All notable changes to SCRIBE are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Comprehensive documentation set under `docs/`: USAGE, ARCHITECTURE, TROUBLESHOOTING, PRIVACY.
- `examples/sample-output.md` — committed sample of the generated Markdown format.
- `LICENSE` (MIT).

### Changed
- Default summarization model switched from `gemma2:9b` to `gemma4:12b-mlx`. Gemma 4 brings a 256K context window (vs 8K), explicit 140+ language training including Danish, and MLX-native execution for ~2× speedup on Apple Silicon over the equivalent GGUF.
- `pyproject.toml` Python range relaxed to `>=3.11,<3.14` to support the system Python 3.13 that ships in the standard macOS Python.org installer.
- README rewritten for clarity; performance numbers updated from the pessimistic 0.3–0.5× realtime estimate to the measured ~1.7–2× realtime on M-series chips.

### Fixed
- `uv.toml` with `system-certs = true` so `uv sync` works on corporate networks with MITM TLS certificates (issuer chain in the macOS keychain).
- WhisperX 3.8.6 renamed the `DiarizationPipeline` constructor's `use_auth_token` kwarg to `token`. `src/scribe/transcribe.py` updated to match — previously crashed with `TypeError: DiarizationPipeline.__init__() got an unexpected keyword argument 'use_auth_token'` when diarization was enabled.

## [0.1.0] - Initial scaffolding

### Added
- Project skeleton: `pyproject.toml`, `uv.toml`, `.python-version`, `.env.example`, `.gitignore`.
- `src/scribe/` package with modules: `cli`, `config`, `pipeline`, `transcribe`, `summarize`, `render`.
- Two CLI commands: `scribe transcribe FILE` and `scribe batch`.
- WhisperX large-v3 transcription on CPU with int8 quantization.
- pyannote.audio v4 speaker diarization (`speaker-diarization-community-1`).
- Ollama-based summarization with Danish/English prompt templates.
- Markdown output with YAML frontmatter, language-aware section headings, and speaker-labeled timestamped transcript.
