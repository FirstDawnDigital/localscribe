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
) -> TranscriptResult:
    """Run full WhisperX pipeline: transcribe → align → diarize → assign speakers."""
    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    duration = probe_duration(audio_path)

    # 1. Transcribe
    asr_model = whisperx.load_model(
        cfg.whisper_model,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
        language=language,
    )
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
