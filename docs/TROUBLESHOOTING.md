# Troubleshooting

Known issues we've actually hit, with the fix.

## `bash: pip: command not found`

You ran `pip install something`. SCRIBE uses `uv`, not pip. Don't install Python packages with pip — they wouldn't land in the SCRIBE venv anyway.

If you need to install a new dependency, add it to `pyproject.toml` and run:

```bash
uv sync
```

To run any Python tool inside SCRIBE's venv:

```bash
uv run <command>
# e.g. uv run python -c "import whisperx; print(whisperx.__version__)"
```

---

## `NameResolutionError: huggingface.co` or `client error (Connect)` during `uv sync` or first run

The host couldn't be reached. Two common causes on a work laptop:

1. **VPN dropped or DNS hiccup.** Reconnect and retry.
2. **Corporate MITM TLS certificates.** `uv sync` will fail with `invalid peer certificate: UnknownIssuer`. The repo's `uv.toml` already sets `system-certs = true` so uv uses the macOS keychain. If you still see this, your keychain doesn't trust the MITM root — talk to IT and have them install it. As a temporary workaround:

   ```bash
   uv sync --native-tls
   ```

---

## `HF_TOKEN is required for speaker diarization`

You ran without setting up the HuggingFace token. Two options:

- **Set it up** (5 minutes, one-time): see the [README quickstart, step 3](../README.md#quickstart).
- **Skip diarization** for this run:

  ```bash
  uv run scribe transcribe --no-diarize input/foo.mp3
  ```

The diarization model is gated — even with a valid token you must also accept the license at <https://huggingface.co/pyannote/speaker-diarization-community-1> while logged in. The "Access repository" button must show as confirmed.

---

## `TypeError: DiarizationPipeline.__init__() got an unexpected keyword argument 'use_auth_token'`

You're on a newer WhisperX than the code was written against. WhisperX 3.8.x renamed the kwarg from `use_auth_token` to `token`. Fix in `src/scribe/transcribe.py`:

```python
diarize_pipeline = whisperx.diarize.DiarizationPipeline(
    token=cfg.hf_token,        # was: use_auth_token=cfg.hf_token
    device=cfg.whisper_device,
)
```

Already fixed in this repo as of [CHANGELOG entry](../CHANGELOG.md). If you see it again after a future `uv sync`, the API may have changed again — inspect `.venv/lib/python*/site-packages/whisperx/diarize.py` for the current `DiarizationPipeline.__init__` signature.

---

## `torchcodec is not installed correctly … LC_RPATH's found` warning

This warning prints at import time on every run. It's harmless. pyannote.audio v4 ships with `torchcodec` for audio decoding but its dylibs are linked against ffmpeg versions that don't match the Homebrew ffmpeg path; pyannote falls back to a different decoder that works fine.

You can ignore it. To silence it cosmetically:

```bash
uv run scribe transcribe input/foo.mp3 2>&1 | grep -v -E "(torchcodec|libavutil|LC_RPATH|libtorchcodec)"
```

---

## Ollama: `connection refused` to `localhost:11434`

The Ollama service isn't running.

```bash
brew services start ollama
# verify
curl -s http://localhost:11434/api/tags | head
```

---

## Ollama: `model not found, try pulling it first`

The configured model isn't on disk. Either change `OLLAMA_MODEL` in `.env` or pull the model:

```bash
ollama pull gemma4:12b-mlx
# or whichever is in your .env
ollama list   # see what's installed
```

---

## First run hangs at "Transcribing …" for several minutes

It's downloading the WhisperX large-v3 model (~3 GB) on first use. Check progress with:

```bash
du -sh ~/.cache/huggingface/hub/
```

If the cache size keeps growing, it's working. Output appears after the download completes.

---

## Output looks like gibberish or wrong language

- **Force the language**: `--language da` or `--language en`. Auto-detect can fail on very short clips, music intros, or speech with strong accents.
- **Check the audio**: play the file in QuickTime to make sure it's not silent or corrupted.
- **Whisper hallucinations on silence**: extended silent stretches sometimes produce phantom sentences like "Subtitles by …" or random foreign-language text. This is a known Whisper pathology. Trim leading/trailing silence with `ffmpeg`:

  ```bash
  ffmpeg -i input.mp3 -af silenceremove=start_periods=1:start_silence=0.5:start_threshold=-50dB output.mp3
  ```

---

## Diarization assigns everything to one speaker

- The audio may genuinely be a monologue.
- For very short clips (<30 s) pyannote may not have enough signal to separate speakers.
- Overlapping speech and similar voices both reduce diarization accuracy. Expected DER on AMI is ~13%; field recordings will be worse.

---

## `output/<file>.md` says it "exists" and won't regenerate

The pipeline skips files where output already exists. Use `--force`:

```bash
uv run scribe transcribe --force input/foo.mp3
```

---

## Running out of memory / Mac slows to a crawl

Gemma 4 12B uses ~7 GB of RAM during summarization. On 8 GB Macs this competes with everything else. Mitigations:

- Use the smaller summarizer: `OLLAMA_MODEL=gemma4:e4b` in `.env` (~5 GB).
- Or skip summary on long files: `--no-summary`.

---

## When to file an issue

Include:
- The exact command you ran
- The full error output (not just the last line)
- macOS version, chip (M1/M2/M3/M4), RAM
- `uv run scribe --help` works? `ollama list` output?
- File a sample (or describe the audio: duration, language, single/multi-speaker)
