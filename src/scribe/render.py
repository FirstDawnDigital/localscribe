"""Render transcription results to Markdown."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .transcribe import TranscriptResult


def _fmt_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_duration(seconds: float) -> str:
    return _fmt_timestamp(seconds)


def render(
    source_file: Path,
    result: TranscriptResult,
    summary: str | None,
    whisper_model: str,
    ollama_model: str | None,
) -> str:
    """Build the full Markdown document."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    frontmatter = [
        "---",
        f'source_file: "{source_file.name}"',
        f"duration: {_fmt_duration(result.duration)}",
        f"language: {result.language}",
        f"transcribed_at: {now}",
        f"whisper_model: {whisper_model}",
    ]
    if ollama_model:
        frontmatter.append(f"summary_model: {ollama_model}")
    frontmatter.append("---")

    parts: list[str] = ["\n".join(frontmatter), ""]

    parts.append(f"# {source_file.stem}")
    parts.append("")

    if summary:
        heading = "Resumé" if result.language == "da" else "Summary"
        parts.append(f"## {heading}")
        parts.append("")
        parts.append(summary)
        parts.append("")

    transcript_heading = "Transskription" if result.language == "da" else "Transcript"
    parts.append(f"## {transcript_heading}")
    parts.append("")

    current_speaker: str | None = None
    for seg in result.segments:
        if not seg.text:
            continue
        ts = _fmt_timestamp(seg.start)
        if seg.speaker != current_speaker:
            parts.append("")
            parts.append(f"**[{ts}] {seg.speaker}:** {seg.text}")
            current_speaker = seg.speaker
        else:
            parts.append(f"**[{ts}]** {seg.text}")

    parts.append("")
    return "\n".join(parts)
