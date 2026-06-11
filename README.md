# SCRIBE

Local interview transcription for Apple Silicon Macs. Drop an MP3/WAV/M4A file into `input/`, get back a Markdown transcript with speaker labels, timestamps and a locally generated summary. Auto-detects Danish and English. **Nothing leaves your machine** apart from the one-time model downloads.

## Why

- **GDPR/privacy**: interview audio with named participants is sensitive personal data; cloud transcription (Otter, PLAUD, Teams Copilot) means data residency questions and processor agreements. SCRIBE avoids all of that.
- **No vendor lock-in**: free and open-source stack, no subscription, no rate limits.
- **Works with any source**: any audio file you can drag onto your Mac — not just meetings recorded inside a specific app.

See [docs/PRIVACY.md](docs/PRIVACY.md) for exactly what does and does not leave the machine.

## Quickstart

Requires macOS on Apple Silicon (M1 or later), ~12 GB free disk space, and an internet connection for first-time model downloads.

```bash
# 1. System dependencies
brew install ffmpeg ollama uv
brew services start ollama
ollama pull gemma4:12b-mlx

# 2. Project setup
git clone https://github.com/FirstDawnDigital/localscribe.git
cd localscribe
uv sync
cp .env.example .env

# 3. HuggingFace token (one-time, for speaker diarization)
#    a) Sign up at https://huggingface.co
#    b) Accept license at https://huggingface.co/pyannote/speaker-diarization-community-1
#    c) Create read token at https://huggingface.co/settings/tokens
#    d) Paste it into .env on the HF_TOKEN= line

# 4. Drop an audio file in input/ and run
uv run scribe transcribe "input/my-interview.mp3"
```

Output appears as `output/my-interview.md`.

## What you get

```markdown
---
source_file: "my-interview.mp3"
duration: 00:32:14
language: da
transcribed_at: 2026-06-11T10:23:00+00:00
whisper_model: large-v3
summary_model: gemma4:12b-mlx
---

## Resumé
Et koncist resumé på dansk med nøglepointer og temaer.

## Transskription

**[00:00:00] SPEAKER_00:** Tak fordi du tog dig tid…
**[00:00:12] SPEAKER_01:** Det var så lidt.
```

A real example is committed at [examples/sample-output.md](examples/sample-output.md).

## Performance (measured on M-series)

| Audio length | Wall-clock | Realtime ratio |
|---|---|---|
| 23 min English podcast | ~13 min | ~1.8× realtime |
| 60 min interview (est.) | ~35 min | ~1.7× realtime |

Numbers include WhisperX large-v3 transcription, pyannote diarization, and Gemma 4 summarization on a single Mac. First run is ~3 GB slower because of model downloads.

## Documentation

| Document | For |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | Daily use: every command, every flag, all `.env` variables, common workflows |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the pipeline works and why each tool was chosen |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Known errors and how to fix them |
| [docs/PRIVACY.md](docs/PRIVACY.md) | What leaves the machine, and what doesn't |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

## License

MIT — see [LICENSE](LICENSE). The bundled dependencies have their own licenses; pyannote in particular requires you to accept Gated Model terms on HuggingFace before use.
