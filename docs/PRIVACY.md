# Privacy & Data Handling

SCRIBE was built specifically to keep interview audio off third-party servers. This document spells out exactly what does and does not leave your machine.

## TL;DR

After first-time setup, **no audio, transcripts, or summaries leave your computer**. Everything runs locally on the Apple Silicon chip.

## What goes over the network

| When | What | To where | Why |
|---|---|---|---|
| Once, during `uv sync` | Python package downloads | PyPI (`files.pythonhosted.org`) | Standard dependency install |
| Once, on first transcription | WhisperX `large-v3` weights (~3 GB) | HuggingFace (`huggingface.co`) | ASR model |
| Once, on first diarization | pyannote `speaker-diarization-community-1` weights (~50 MB) | HuggingFace (`huggingface.co`) | Diarization model |
| Once, on `ollama pull gemma4:12b-mlx` | LLM weights (~7 GB) | `ollama.com` registry | Summarization model |

**That's the complete list.** Audio files and transcripts are never uploaded anywhere. There is no telemetry, no analytics, no crash reporting.

You can verify this:

```bash
# After first setup, disconnect from the network and run:
uv run scribe transcribe input/foo.mp3
# It will work.
```

## What stays on your machine

| Data | Where |
|---|---|
| Input audio | `input/` (or wherever you placed it) |
| Transcripts | `output/` |
| Model weights | `~/.cache/huggingface/hub/` and Ollama's storage (`~/.ollama/models/`) |
| Your HF token | `.env` in the project root (gitignored) |

No data is sent to OpenAI, Anthropic, Google, Microsoft, or any other cloud service.

## Comparison to cloud transcription services

| Tool | Where audio goes | GDPR posture |
|---|---|---|
| **SCRIBE** | Stays on your Mac | Data never leaves your control — straightforward |
| PLAUD | Uploaded to PLAUD servers → OpenAI Whisper API | Requires DPA with PLAUD + sub-processor agreement with OpenAI |
| Microsoft Teams transcription | Microsoft 365 cloud | Covered by your M365 DPA, but data crosses tenant boundary |
| Otter.ai | Uploaded to Otter servers | US-hosted; explicit consent required from EU interviewees |
| OpenAI Whisper API | Uploaded to OpenAI | Requires DPA; data retention varies by API tier |

For interviews containing personal data (and most interviews do), SCRIBE is the lowest-friction path to a defensible GDPR position because the processing happens entirely within your existing legitimate use of your own device.

## Things you still need to handle yourself

SCRIBE addresses the technical processing layer. It does not handle:

- **Consent**: you still need informed consent from interviewees to record and transcribe them.
- **Storage**: where you keep the audio and Markdown after generation is your call (encrypted disk recommended).
- **Sharing**: if you email or upload the transcript later, that's a separate disclosure decision.
- **Retention**: decide and document how long you keep recordings vs. transcripts.
- **Pseudonymization**: speaker labels are `SPEAKER_00`, `SPEAKER_01`, etc. You add real names manually. Consider keeping a separate key file if you need pseudonymization for compliance.

## Disabling the HuggingFace token

If you don't need speaker diarization (e.g. solo recordings or monologues), you can run SCRIBE without ever signing up for HuggingFace:

```bash
uv run scribe transcribe --no-diarize input/foo.mp3
```

In this mode the only HuggingFace traffic is the one-time WhisperX model download, which doesn't require a token.

## Audit trail

The output Markdown's YAML frontmatter records what processed the file:

```yaml
whisper_model: large-v3
summary_model: gemma4:12b-mlx
transcribed_at: 2026-06-11T10:23:00+00:00
```

This is enough to reconstruct, months later, exactly which model versions produced a given transcript — useful for compliance documentation.
