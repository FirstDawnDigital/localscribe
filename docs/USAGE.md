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
| `-a, --audio-preset` | `auto` | Audio preprocessing chain. See [Audio presets](#audio-presets). |

### `scribe batch`

Transcribe every supported audio file in a directory.

```bash
uv run scribe batch                  # processes input/
uv run scribe batch --dir ./recordings
```

Same flags as `transcribe`, except the positional argument is replaced by `-d/--dir`. Files where the output already exists are skipped unless `--force` is passed.

## Audio presets

Real-world recordings (meetings, interviews on phones, conference table mics) are
often 15–18 dB below broadcast loudness and have uneven gain between speakers.
This confuses Whisper's language detection and pyannote's VAD/diarization.
SCRIBE can preprocess audio with `ffmpeg` before transcription:

| Preset | Filter chain | When to use |
|---|---|---|
| `none` | (raw input) | Files already mastered for broadcast / podcast. |
| `clean` | highpass + loudnorm | Default safe choice for most recordings. |
| `meeting` | highpass + dynaudnorm + loudnorm | Far-field mics, multiple speakers, uneven levels. |
| `auto` *(default)* | probes the file, picks `clean` or `meeting` | Recommended — set and forget. |

The chosen preset and key probe metrics (mean volume, LUFS-I, LRA, silence ratio)
are written into the output Markdown's YAML frontmatter so you can audit what was
done.

In the `2026-06-12` test battery, switching from `none` to `meeting` on a
Danish conference recording fixed a language misdetection (Norwegian → Danish),
recovered a sentence Whisper had reduced to four words, and split a previously
collapsed two-speaker interview into two distinct diarized voices. Lowering VAD
thresholds or adding FFT denoise made results *worse* (Whisper hallucinated
YouTube captions like "Tak fordi du så med"), so the presets stay conservative.

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

### Use a Danish-fine-tuned Whisper (situational)

OpenAI's `large-v3` has weak Danish coverage and is prone to YouTube-caption
hallucinations on near-silence ("Tak fordi du så med", "Danske tekster af
Nicolai Winther"). For **close-mic** Danish recordings (lavalier, headset,
single-speaker dictation), a community fine-tune can be a good swap:

```bash
uv run scribe transcribe input/møde.m4a -m pluttodk/hviske-tiske -l da
```

**Caveat — measured 2026-06-13 on a far-field conference clip:**
hviske-v2 was 2× faster but produced clearly worse output than
`large-v3` + decoder hardening: more spelling errors, full hallucinations
on the noisier middle segment, and one segment flipped to English despite
`-l da`. hviske-v2 is trained on close-mic read/conversational Danish
(CoRal, Common Voice) and degrades sharply on meeting/far-field audio.
**Default to `large-v3` for far-field; only switch to hviske-v2 if your
input is clean close-mic.**

Recommended Danish models (all CTranslate2-compatible with faster-whisper):

| Model | Size | Notes |
|---|---|---|
| `pluttodk/hviske-tiske` | ~3.1 GB | hviske-v2 family, OpenRAIL license (commercial OK). Close-mic only. |
| `syvai/faster-hviske-v3-conversation` | ~3.1 GB | Fine-tuned on conversational Danish, same family. Not A/B-tested yet. |

Or set `WHISPER_MODEL=pluttodk/hviske-tiske` in `.env` to make it the
default. Always pass `-l da` — these models are Danish-only.

**Avoid:** hviske-v5 and later — relicensed to CC-BY-NC-4.0 (non-commercial).

### Speech separation for far-field meetings (`--separator pixit`)

For overlapping/far-field meeting audio where the default pipeline drops
content (e.g. Whisper produces 3 segments for 5 minutes), try PixIT
speech separation — the only pretrained separator trained on real
distant-mic meeting audio (AMI-SDM):

```bash
uv run scribe transcribe input/møde.m4a -l da -s pixit
```

This replaces pyannote diarization with PixIT's joint
diarization+separation: PixIT produces one cleaned audio stream per
speaker, each gets transcribed independently, and the per-stream
transcripts are merged.

**First-time setup:**

1. Accept user conditions while logged in on BOTH of these gated
   repositories (PixIT depends on both, and the inner one is gated
   separately):
   - <https://huggingface.co/pyannote/speech-separation-ami-1.0>
   - <https://huggingface.co/pyannote/separation-ami-1.0>
2. Make sure your `HF_TOKEN` (already required for diarization) is from
   the same account.
3. Install the optional dependency group:
   ```bash
   uv pip install -e ".[separate]"
   ```
   This pulls `asteroid` (the ToTaToNet separation network) and
   `speechbrain` (ECAPA speaker embeddings).
4. First run will download a few hundred MB of model weights into the
   HF cache.

**Caveats (from our own testing — see `docs/BACKLOG.md` § B3):**
- **Slow on CPU.** ~12 min for a 5-min clip on an M1 (vs ~8 min for the
  default pipeline), because Whisper runs once per separated stream.
- **Max ~3 concurrent speakers.** PixIT was trained for typical AMI
  scenes; very crowded recordings will lose voices.
- **English-trained separator.** Separation is largely language-agnostic
  in practice (it works on waveforms, not text), but this hasn't been
  rigorously validated for Danish.
- **Leakage removal is mandatory** and SCRIBE applies the masking step
  automatically — without it the published WER actually gets *worse*.
- **Phantom-stream hallucinations.** PixIT's diarizer can over-produce
  speakers, and the resulting near-silent stream is loud enough for
  Whisper to invent subtitle credits ("Danske tekster af …",
  "Subtitles by …"). SCRIBE defends against this with three filters:
  a 1.5 s minimum active-duration gate, a −42 dBFS RMS gate on the
  cleaned stream, and a regex post-filter that drops segments matching
  the canonical Whisper subtitle-credit patterns. On our 5-min test
  clip the phantom 3rd stream gets through both energy gates but the
  regex catches its output and the rendered transcript is clean. Still
  inspect every PixIT transcript before trusting it — new attractor
  strings can show up and need to be added to
  `scribe.separate.HALLUCINATION_REGEXES`.

### Score a transcript against a manual reference (`scribe eval`)

To compare runs objectively (e.g. did switching from `large-v3` to
`pluttodk/hviske-tiske` actually help?), produce a hand-curated reference for
one or two representative recordings and score future runs against it with
[meeteval](https://github.com/fgnt/meeteval)'s tcpWER.

Install the optional eval extra once:

```bash
uv pip install 'meeteval>=0.4.0'    # or: uv sync --extra eval (when supported)
```

Workflow:

```bash
# 1. Transcribe normally
uv run scribe transcribe input/sample.m4a

# 2. Bootstrap a reference from the (likely flawed) hypothesis
uv run scribe eval bootstrap -i output/sample.md -o references/sample.ref.stm

# 3. Open references/sample.ref.stm in an editor and fix the words,
#    speakers, and rough start/end times against the actual audio.
#    The STM format is one line per segment:
#      <session> 1 <SPEAKER_XX> <start_s> <end_s> the actual words
#    Spaces and the apostrophe in your filename are sanitized.

# 4. Re-transcribe after any change (model swap, preset change, ...)
uv run scribe transcribe input/sample.m4a --force -m pluttodk/hviske-tiske -l da

# 5. Score
uv run scribe eval score -r references/sample.ref.stm -i output/sample.md
```

`tcpwer` reports word error rate (substitutions + insertions + deletions) plus
diarization errors (missed_speaker = a true speaker we never produced;
falarm_speaker = a cluster we invented). The default 5 s collar is forgiving
for segment-level timestamps; lower it if you have word-level alignment.

**Important:** SCRIBE only emits segment-level timestamps, so meeteval is told
to spread per-word times evenly across each segment's character span. WERs
will move primarily with text accuracy and speaker attribution, not exact
timing.

## Tips

- **Speaker count.** pyannote auto-detects the number of speakers. If you know the exact count and want better results, you can tweak `DiarizationPipeline(...)` in [src/scribe/transcribe.py](src/scribe/transcribe.py) to pass `num_speakers=N`.
- **Long files (>1 h).** WhisperX handles them, but expect proportional wall-clock time. There's no benefit to splitting manually — diarization needs the full file to keep speaker IDs consistent.
- **Noisy recordings.** No built-in denoising. Pre-process with `ffmpeg -i in.mp3 -af "afftdn=nf=-25" out.wav` if needed, or look into [Demucs](https://github.com/facebookresearch/demucs).
- **Don't commit `.env`.** It contains your HF token. The repo's `.gitignore` already excludes it.
