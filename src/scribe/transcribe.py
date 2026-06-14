"""WhisperX-based transcription with diarization."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class TranscriptResult:
    language: str
    duration: float
    segments: list[Segment]


# Decoder defaults hardened against Whisper's signature failure modes on noisy/
# far-field audio: repetition loops, YouTube-caption hallucinations on silence,
# and self-reinforcing context drift. Tunable via WhisperX's asr_options.
HARDENED_ASR_OPTIONS: dict[str, Any] = {
    # Stop the previous transcript from biasing the next window — primary cause of
    # "Ja. Ja. Ja." loops once one bad token slips in.
    "condition_on_previous_text": False,
    # Lower than WhisperX default 2.4 so collapsed-entropy segments (the signature
    # of a repetition loop) get rejected instead of emitted.
    "compression_ratio_threshold": 2.0,
    # Slightly more aggressive silence rejection than the 0.6 default; pairs with
    # condition_on_previous_text=False to kill subtitle-credit hallucinations on
    # near-silent tails.
    "no_speech_threshold": 0.5,
    # Newer faster-whisper option: when the model is "transcribing" silence,
    # drop the segment entirely. Safe no-op on faster-whisper builds that
    # don't recognise the key.
    "hallucination_silence_threshold": 0.5,
}


def probe_duration(audio_path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def transcribe(
    audio_path: Path,
    cfg: Config,
    language: str | None = None,
    diarize: bool = True,
    vad_options: dict[str, float] | None = None,
) -> TranscriptResult:
    """Run full WhisperX pipeline: transcribe → align → diarize → assign speakers."""
    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    duration = probe_duration(audio_path)

    # 1. Transcribe
    load_kwargs: dict[str, Any] = dict(
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
        language=language,
        asr_options=HARDENED_ASR_OPTIONS,
    )
    if vad_options:
        load_kwargs["vad_options"] = vad_options
    try:
        asr_model = whisperx.load_model(cfg.whisper_model, **load_kwargs)
    except TypeError:
        # Older WhisperX builds (or faster-whisper backends) may reject one of
        # the hardened options (typically hallucination_silence_threshold).
        # Retry with a minimal safe subset.
        safe_opts = {
            k: v for k, v in HARDENED_ASR_OPTIONS.items()
            if k != "hallucination_silence_threshold"
        }
        load_kwargs["asr_options"] = safe_opts
        asr_model = whisperx.load_model(cfg.whisper_model, **load_kwargs)
    asr_result: dict[str, Any] = asr_model.transcribe(audio, batch_size=8)
    detected_lang: str = asr_result["language"]

    # 2. Word-level alignment
    try:
        align_model, align_meta = whisperx.load_align_model(
            language_code=detected_lang,
            device=cfg.whisper_device,
        )
        aligned = whisperx.align(
            asr_result["segments"],
            align_model,
            align_meta,
            audio,
            cfg.whisper_device,
            return_char_alignments=False,
        )
    except Exception:
        # If no alignment model exists for the language, fall back to raw segments.
        aligned = asr_result

    # 3. Diarization (optional; requires HF token)
    if diarize:
        if not cfg.hf_token:
            raise RuntimeError(
                "HF_TOKEN is required for speaker diarization. "
                "Set it in .env and accept the model license at "
                "https://huggingface.co/pyannote/speaker-diarization-community-1 "
                "— or run with --no-diarize to skip."
            )
        diarize_pipeline = whisperx.diarize.DiarizationPipeline(
            token=cfg.hf_token,
            device=cfg.whisper_device,
        )
        diarize_segments = diarize_pipeline(audio)
        merged = whisperx.assign_word_speakers(diarize_segments, aligned)
        raw_segments = merged.get("segments", [])
        default_speaker = "SPEAKER_00"
    else:
        raw_segments = aligned.get("segments", [])
        default_speaker = "SPEAKER"

    segments: list[Segment] = []
    for seg in raw_segments:
        segments.append(
            Segment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                speaker=str(seg.get("speaker", default_speaker)),
                text=str(seg.get("text", "")).strip(),
            )
        )

    return TranscriptResult(language=detected_lang, duration=duration, segments=segments)
