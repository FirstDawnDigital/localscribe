"""PixIT speaker separation pre-stage for SCRIBE.

Wraps `pyannote/speech-separation-ami-1.0` — the only pretrained speech
separation pipeline trained on real far-field meeting audio (AMI-SDM). On
the published benchmark this lowers cpWER from 38.8 % to 32.8 % with the
full leakage-removal pipeline; without leakage removal the same pipeline
makes WER *worse* (50.1 %), so the masking step is mandatory.

Workflow:

    1. Load pipeline (gated HF model; requires accepting terms once).
    2. Run on the (preprocessed) audio → diarization + per-speaker streams.
    3. Apply leakage removal: zero out each stream outside its speaker's
       diarized regions, with a small collar pad to avoid clipping edges.
    4. Write each stream to a temp 16 kHz wav, return paths + speaker labels.

The caller then runs ASR on each stream independently and merges segments
by start time, tagging each with the PixIT speaker label.

Optional dependency: `pip install scribe[separation]` (no extra packages —
pyannote.audio is already a hard dep — but kept as a marker for the future
in case PixIT-specific deps are needed).
"""
from __future__ import annotations

import logging
import re
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

PIXIT_MODEL = "pyannote/speech-separation-ami-1.0"
PIXIT_SAMPLE_RATE = 16_000

# How many seconds of audio on either side of a speaker's diarized region
# the leakage-removal mask should keep open. 0.5 s matches the recommendation
# from the PixIT paper and keeps Whisper from clipping word edges.
LEAKAGE_COLLAR_SECONDS = 0.5

# Drop separated streams whose total non-silent duration is below this
# threshold — these are usually a phantom 3rd speaker that PixIT invented
# and that would only contribute hallucinations to the merged transcript.
MIN_STREAM_ACTIVE_SECONDS = 1.5

# Drop streams whose post-mask RMS (over the *active* samples only) is below
# this dBFS threshold. PixIT's diarizer can claim a phantom speaker is active
# for plenty of seconds even when the underlying audio is leakage-level
# residue. Empirically on our 5-min far-field clip the real speakers land
# around -30 to -22 dBFS RMS, while the phantom 3rd stream sits at <-45 dBFS
# and produces Whisper subtitle-credit hallucinations.
MIN_STREAM_RMS_DBFS = -42.0

# Post-ASR filter: if a segment matches any of these regex patterns it is
# dropped. These are the canonical Whisper "subtitle credits" hallucination
# attractors — they're not in any real recording, only in YouTube/Netflix
# subtitle training data, and Whisper falls back to them on near-silent
# stream segments. Case-insensitive.
HALLUCINATION_REGEXES = (
    r"\bdanske tekster\b",
    r"\btekstet af\b",
    r"\bsubtitles? by\b",
    r"\bsubtitles? provided by\b",
    r"\btranscribed by\b",
    r"\bscandinavian text service\b",
    r"\bsdi media\b",
    r"\bnordisk undertekst\b",
    r"\bamara\.org\b",
)


@dataclass
class SeparatedStream:
    """One separated audio stream + the speaker label PixIT assigned it."""

    speaker: str
    wav_path: Path
    active_seconds: float


def _patch_speechbrain_for_pyannote() -> None:
    """pyannote.audio 4.0.4 calls speechbrain's `EncoderClassifier.from_hparams`
    with a `token=` kwarg, which is forwarded as `**kwargs` to
    `Pretrained.__init__`. speechbrain >=1.0 removed that kwarg, raising
    `TypeError: Pretrained.__init__() got an unexpected keyword argument
    'token'`. Patch it to silently drop the kwarg. No-op if speechbrain isn't
    installed or has already been patched."""
    try:
        from speechbrain.inference.interfaces import Pretrained  # type: ignore
    except ImportError:
        return
    orig_init = Pretrained.__init__
    if getattr(orig_init, "_pyannote_token_shim", False):
        return

    def _patched(self, *args, **kwargs):
        kwargs.pop("token", None)
        kwargs.pop("use_auth_token", None)
        return orig_init(self, *args, **kwargs)

    _patched._pyannote_token_shim = True  # type: ignore[attr-defined]
    Pretrained.__init__ = _patched  # type: ignore[method-assign]


def _load_pipeline(cfg: Config):
    """Load the PixIT pipeline once. Raises a clear error if gating not done."""
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pyannote.audio not installed — should be part of the base deps. "
            "Run `uv sync`."
        ) from e

    if not cfg.hf_token:
        raise RuntimeError(
            "HF_TOKEN is required to load the PixIT model. Set it in .env. "
            "Also accept the model terms at "
            f"https://huggingface.co/{PIXIT_MODEL}"
        )

    _patch_speechbrain_for_pyannote()

    log.info("Loading separation pipeline: %s", PIXIT_MODEL)
    try:
        pipeline = Pipeline.from_pretrained(PIXIT_MODEL, token=cfg.hf_token)
    except TypeError as e:
        # Older pyannote.audio (<3.2) used `use_auth_token` instead of `token`.
        # We only retry if the error is specifically about that kwarg —
        # otherwise we'd mask real bugs from inner code.
        if "unexpected keyword argument 'token'" in str(e):
            pipeline = Pipeline.from_pretrained(
                PIXIT_MODEL, use_auth_token=cfg.hf_token
            )
        else:
            raise
    except ImportError as e:
        raise RuntimeError(
            f"PixIT requires an extra dependency that is missing: {e}\n"
            f"Install with: `uv pip install asteroid speechbrain` (or "
            f"`uv pip install -e .[separate]`)."
        ) from e
    except Exception as e:
        msg = str(e).lower()
        if "403" in msg or "gated" in msg or "restricted" in msg or "unauthorized" in msg:
            raise RuntimeError(
                f"Failed to load {PIXIT_MODEL}: access denied. Accept the "
                f"model terms for both:\n"
                f"  - https://huggingface.co/{PIXIT_MODEL}\n"
                f"  - https://huggingface.co/pyannote/separation-ami-1.0\n"
                f"using the HF account whose token is in HF_TOKEN, then retry."
            ) from e
        raise RuntimeError(f"Failed to load {PIXIT_MODEL}: {e}") from e

    return pipeline


def _diarization_to_speaker_mask(
    diarization,
    speaker: str,
    num_samples: int,
    sample_rate: int = PIXIT_SAMPLE_RATE,
    collar_seconds: float = LEAKAGE_COLLAR_SECONDS,
) -> np.ndarray:
    """Build a 1-D float32 mask over the separated stream that is 1.0 where
    `speaker` is diarized as active (plus a small collar), 0.0 elsewhere.

    This is the manual leakage-removal step. Multiplying the separated source
    by this mask before ASR removes cross-talk from other speakers that PixIT
    couldn't fully isolate — without this step the WER actually gets worse
    than the unseparated baseline.
    """
    mask = np.zeros(num_samples, dtype=np.float32)
    collar = int(collar_seconds * sample_rate)
    for segment, _, spk in diarization.itertracks(yield_label=True):
        if spk != speaker:
            continue
        start = max(0, int(segment.start * sample_rate) - collar)
        end = min(num_samples, int(segment.end * sample_rate) + collar)
        if end > start:
            mask[start:end] = 1.0
    return mask


def _write_wav_mono_16k(samples: np.ndarray, path: Path) -> None:
    """Write a mono float32 array to a 16-bit PCM wav file at 16 kHz."""
    # Clip to [-1, 1] and convert to int16
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(PIXIT_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _load_audio_for_pipeline(audio_path: Path):
    """Pre-load audio into a {'waveform', 'sample_rate'} dict.

    pyannote.audio 4.x uses torchcodec for audio decoding, but torchcodec's
    macOS wheels currently link against ffmpeg dylibs via @rpath in a way
    that breaks on Homebrew installs (`Library not loaded: @rpath/libavutil.*`).
    The documented workaround in pyannote's own warning is to preload the
    audio with another backend and pass it as a dict — which we do via
    torchaudio (which works fine).
    """
    import torchaudio  # type: ignore
    waveform, sample_rate = torchaudio.load(str(audio_path))
    return {"waveform": waveform, "sample_rate": sample_rate}


def separate(audio_path: Path, cfg: Config, work_dir: Path) -> list[SeparatedStream]:
    """Run PixIT on `audio_path` and return one cleaned stream per speaker.

    `work_dir` should be an existing directory (typically a tempdir from the
    caller) where the per-speaker wav files will be written.
    """
    pipeline = _load_pipeline(cfg)

    log.info("Running PixIT separation on %s", audio_path.name)
    audio_dict = _load_audio_for_pipeline(audio_path)
    result = pipeline(audio_dict)

    # The pipeline returns either (diarization, sources) or
    # (diarization, sources, embeddings) depending on call flags.
    if isinstance(result, tuple) and len(result) >= 2:
        diarization, sources = result[0], result[1]
    else:
        raise RuntimeError(
            f"Unexpected return type from {PIXIT_MODEL}: {type(result).__name__}. "
            "Expected (diarization, sources) tuple."
        )

    # `sources` is a pyannote SlidingWindowFeature; `.data` is (num_samples, num_speakers).
    data = np.asarray(sources.data)
    if data.ndim != 2:
        raise RuntimeError(
            f"Unexpected sources shape {data.shape}; expected (num_samples, num_speakers)."
        )
    num_samples, num_speakers = data.shape
    labels = list(diarization.labels())
    if len(labels) != num_speakers:
        log.warning(
            "PixIT label count (%d) != source count (%d) — taking the min and "
            "trusting positional ordering.",
            len(labels), num_speakers,
        )
    n = min(len(labels), num_speakers)

    streams: list[SeparatedStream] = []
    for i in range(n):
        speaker = labels[i]
        raw = data[:, i].astype(np.float32)

        # Leakage removal: mute audio outside this speaker's diarized regions.
        mask = _diarization_to_speaker_mask(diarization, speaker, num_samples)
        cleaned = raw * mask
        active_samples = int(mask.sum())
        active_seconds = active_samples / PIXIT_SAMPLE_RATE

        if active_seconds < MIN_STREAM_ACTIVE_SECONDS:
            log.info(
                "Dropping near-silent stream for %s (active=%.2fs)",
                speaker, active_seconds,
            )
            continue

        # Energy check on the active portion only — phantom streams pass
        # the duration filter because the diarizer thinks they have speech,
        # but the actual audio left after leakage removal is residue.
        active_samples_vec = cleaned[mask > 0]
        if active_samples_vec.size > 0:
            rms = float(np.sqrt(np.mean(active_samples_vec.astype(np.float64) ** 2)))
            rms_dbfs = 20 * np.log10(rms + 1e-10)
        else:
            rms_dbfs = -120.0
        if rms_dbfs < MIN_STREAM_RMS_DBFS:
            log.info(
                "Dropping low-energy stream for %s (active=%.2fs, rms=%.1f dBFS)",
                speaker, active_seconds, rms_dbfs,
            )
            continue

        out_path = work_dir / f"pixit_{speaker}.wav"
        _write_wav_mono_16k(cleaned, out_path)
        log.info(
            "PixIT stream %s -> %s (%.1fs active of %.1fs, rms=%.1f dBFS)",
            speaker, out_path.name, active_seconds,
            num_samples / PIXIT_SAMPLE_RATE, rms_dbfs,
        )
        streams.append(SeparatedStream(speaker=speaker, wav_path=out_path, active_seconds=active_seconds))

    if not streams:
        raise RuntimeError("PixIT produced no usable streams.")
    return streams


_HALLUCINATION_PATTERN = re.compile(
    "|".join(HALLUCINATION_REGEXES), flags=re.IGNORECASE
)


def is_subtitle_hallucination(text: str) -> bool:
    """Return True if `text` matches a known Whisper subtitle-credit
    hallucination pattern. Used to post-filter PixIT stream output before
    merging — phantom streams that pass the energy gate can still produce
    fragments like 'Danske tekster af …' which never appear in real audio.
    """
    return bool(_HALLUCINATION_PATTERN.search(text))
