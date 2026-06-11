# Usage

This document covers everything beyond the README quickstart: every command, every flag, configuration, and common workflows.

## CLI commands

SCRIBE exposes two commands. Run `uv run scribe --help` or `uv run scribe transcribe --help` for the auto-generated reference.

### `scribe transcribe FILE`

Transcribe one audio file.

```bash
uv run scribe transcribe "input/interview.mp3"
```

Options:

| Flag | Default | Purpose |
|---|---|---|
| `-l, --language` | auto-detect | Force a language code (`da`, `en`, etc.). Useful when auto-detect fails on short or noisy clips. |
| `--no-summary` | off | Skip the LLM summary step. Faster, no Ollama required. |
| `--no-diarize` | off | Skip speaker diarization. No HF token needed. |
| `-f, --force` | off | Overwrite existing output Markdown. |
| `-m, --model` | `large-v3` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3`. |
| `-o, --output-dir` | `output/` | Where to write the Markdown file. |

### `scribe batch`

Transcribe every supported audio file in a directory.

```bash
uv run scribe batch                  # processes input/
uv run scribe batch --dir ./recordings
```

Same flags as `transcribe`, except the positional argument is replaced by `-d/--dir`. Files where the output already exists are skipped unless `--force` is passed.

## Supported input formats

`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`. Anything `ffmpeg` can decode would work in principle; extend `SUPPORTED_EXTENSIONS` in [src/scribe/pipeline.py](src/scribe/pipeline.py) if you need more.

## Output format

One Markdown file per audio file, named `<stem>.md` in the output directory. Structure:

```markdown
---
source_file: "<original filename>"
duration: HH:MM:SS
language: <detected ISO 639-1>
transcribed_at: <ISO 8601 timestamp>
whisper_model: <model used>
summary_model: <model used, omitted if --no-summary>
---

# <filename without extension>

## Summary   (or "Resumé" for Danish)
<paragraph + bullets>

## Transcript   (or "Transskription" for Danish)

**[HH:MM:SS] SPEAKER_00:** first utterance from a speaker
**[HH:MM:SS]** continuation from the same speaker
**[HH:MM:SS] SPEAKER_01:** when speaker changes
```

Speaker labels are anonymous (`SPEAKER_00`, `SPEAKER_01`, …). You map them to real names manually after the fact — SCRIBE does not do voice recognition.

## Configuration (`.env`)

All settings can be overridden in `.env` (copy from `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | _(unset)_ | HuggingFace read token. Required for diarization. Get one at <https://huggingface.co/settings/tokens>. |
| `OLLAMA_MODEL` | `gemma4:12b-mlx` | Model used for summarization. Must already be pulled via `ollama pull`. |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint. Change only if running on another host. |
| `WHISPER_MODEL` | `large-v3` | Whisper model size. `medium` is ~3× faster with a small quality cost on English; non-English suffers more. |
| `WHISPER_DEVICE` | `cpu` | Apple Silicon Macs use CPU with int8 via faster-whisper. Do not set to `mps` — WhisperX does not support MPS reliably. |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantization for faster-whisper. `int8` is the right choice for CPU. |

## Common workflows

### Bulk transcribe a folder of recordings

```bash
cp ~/Library/Group\ Containers/group.com.apple.VoiceMemos.shared/Recordings/*.m4a input/
uv run scribe batch
```

### Re-summarize without re-transcribing

There is no direct flag for this; re-running with `--force` re-does everything. If the summary is the only thing you want to change, delete the `## Summary` section from the existing Markdown and run a separate one-off Ollama prompt against the transcript. (A `scribe resummarize` command could be added — open an issue.)

### Faster turnaround at slight quality cost

```bash
uv run scribe transcribe input/foo.mp3 --model medium
```

`medium` is ~3× faster than `large-v3` and remains usable for clean English. For Danish, prefer `large-v3`.

### Run without internet (after first model download)

Once the models are cached under `~/.cache/huggingface/` and Ollama has the model, SCRIBE works fully offline. Confirm with:

```bash
ls ~/.cache/huggingface/hub/
ollama list
```

### Process audio that's not in `input/`

```bash
uv run scribe transcribe ~/Downloads/recording.wav --output-dir ~/Documents/transcripts
```

## Tips

- **Speaker count.** pyannote auto-detects the number of speakers. If you know the exact count and want better results, you can tweak `DiarizationPipeline(...)` in [src/scribe/transcribe.py](src/scribe/transcribe.py) to pass `num_speakers=N`.
- **Long files (>1 h).** WhisperX handles them, but expect proportional wall-clock time. There's no benefit to splitting manually — diarization needs the full file to keep speaker IDs consistent.
- **Noisy recordings.** No built-in denoising. Pre-process with `ffmpeg -i in.mp3 -af "afftdn=nf=-25" out.wav` if needed, or look into [Demucs](https://github.com/facebookresearch/demucs).
- **Don't commit `.env`.** It contains your HF token. The repo's `.gitignore` already excludes it.
