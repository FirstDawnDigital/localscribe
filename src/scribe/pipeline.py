"""End-to-end pipeline: audio file → Markdown transcript."""
from __future__ import annotations

import tempfile
from pathlib import Path

from rich.console import Console

from .audio_preprocess import PRESETS, apply_preset
from .audio_probe import probe
from .config import Config, OUTPUT_DIR
from .render import render
from .summarize import summarize
from .transcribe import Segment, TranscriptResult, transcribe

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
SUPPORTED_SEPARATORS = {"none", "pixit"}

console = Console()


def _auto_preset(metrics_path: Path) -> str:
    """Pick a preset from probe metrics. Conservative: 'clean' by default,
    'meeting' when loudness range is wide (uneven speakers) or signal is low."""
    m = probe(metrics_path)
    lufs = m.lufs_i if m.lufs_i is not None else -23.0
    lra = m.lufs_lra if m.lufs_lra is not None else 0.0
    mean_db = m.mean_volume_db if m.mean_volume_db is not None else -20.0
    if lra >= 10.0 or mean_db <= -25.0 or lufs <= -25.0:
        return "meeting"
    return "clean"


def process_file(
    audio_path: Path,
    cfg: Config,
    *,
    language: str | None = None,
    do_summary: bool = True,
    diarize: bool = True,
    force: bool = False,
    output_dir: Path = OUTPUT_DIR,
    audio_preset: str = "auto",
    separator: str = "none",
) -> Path:
    """Transcribe one audio file and write a Markdown result. Returns the output path.

    audio_preset:
      - 'none'    : feed the raw file to Whisper unchanged.
      - 'clean'   : highpass + loudnorm (safe on any source).
      - 'meeting' : also dynaudnorm — recommended for far-field / uneven recordings.
      - 'auto'    : probe the file and pick clean or meeting.

    separator:
      - 'none'  : single-stream pipeline (pyannote diarization after ASR).
      - 'pixit' : run pyannote/speech-separation-ami-1.0 first, transcribe
                  each separated speaker stream independently, merge results.
                  Replaces pyannote diarization (PixIT does both jointly).
    """
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if separator not in SUPPORTED_SEPARATORS:
        raise ValueError(f"Unknown separator {separator!r}; choose from {SUPPORTED_SEPARATORS}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{audio_path.stem}.md"

    if output_path.exists() and not force:
        console.print(f"[yellow]Skipping[/yellow] {audio_path.name} (output exists, use --force)")
        return output_path

    # Resolve preset (auto runs a quick probe). Probe is also captured for the frontmatter.
    if audio_preset == "auto":
        chosen = _auto_preset(audio_path)
        console.print(f"[cyan]Audio preset[/cyan] auto → [bold]{chosen}[/bold]")
    else:
        if audio_preset not in PRESETS:
            raise ValueError(f"Unknown audio preset {audio_preset!r}")
        chosen = audio_preset

    audio_metrics_dict: dict[str, float | None] | None = None
    if chosen != "none":
        m = probe(audio_path)
        audio_metrics_dict = {
            "mean_volume_db": m.mean_volume_db,
            "lufs_i": m.lufs_i,
            "lufs_lra": m.lufs_lra,
            "silence_ratio": m.silence_ratio,
        }

    console.print(f"[bold cyan]Transcribing[/bold cyan] {audio_path.name}")

    with tempfile.TemporaryDirectory(prefix="scribe_audio_") as tmp:
        tmp_path = Path(tmp)
        if chosen == "none":
            audio_for_asr = audio_path
        else:
            audio_for_asr = apply_preset(
                audio_path, chosen, tmp_path / f"{audio_path.stem}_{chosen}.wav"
            )

        if separator == "pixit":
            from .separate import separate, is_subtitle_hallucination  # local import — optional codepath
            console.print("[bold cyan]Separating[/bold cyan] with PixIT (this is slow on CPU)")
            streams = separate(audio_for_asr, cfg, tmp_path)
            console.print(f"  PixIT produced [bold]{len(streams)}[/bold] stream(s)")

            merged_segments: list[Segment] = []
            stream_language = language
            total_duration = 0.0
            dropped_hallucinations = 0
            for stream in streams:
                console.print(f"  [cyan]ASR on {stream.speaker}[/cyan] ({stream.wav_path.name})")
                # diarize=False — PixIT already gave us the speaker identity per stream.
                stream_result = transcribe(
                    stream.wav_path, cfg, language=stream_language, diarize=False
                )
                # Lock language detection after the first stream so all streams agree.
                stream_language = stream_language or stream_result.language
                total_duration = max(total_duration, stream_result.duration)
                for seg in stream_result.segments:
                    text = seg.text.strip()
                    if not text:
                        continue
                    if is_subtitle_hallucination(text):
                        dropped_hallucinations += 1
                        continue
                    merged_segments.append(
                        Segment(
                            start=seg.start,
                            end=seg.end,
                            speaker=stream.speaker,
                            text=seg.text,
                        )
                    )

            if dropped_hallucinations:
                console.print(
                    f"  [yellow]Dropped {dropped_hallucinations} subtitle-credit "
                    f"hallucination(s)[/yellow]"
                )
            merged_segments.sort(key=lambda s: s.start)
            result = TranscriptResult(
                language=stream_language or "unknown",
                duration=total_duration,
                segments=merged_segments,
            )
        else:
            result = transcribe(audio_for_asr, cfg, language=language, diarize=diarize)
            # Apply the same subtitle-credit hallucination filter as the PixIT
            # path. Whisper occasionally emits these on low-signal/silent
            # spans regardless of whether we ran source separation first.
            from .separate import is_subtitle_hallucination  # local import
            kept: list[Segment] = []
            dropped = 0
            for seg in result.segments:
                if is_subtitle_hallucination(seg.text.strip()):
                    dropped += 1
                    continue
                kept.append(seg)
            if dropped:
                console.print(
                    f"  [yellow]Dropped {dropped} subtitle-credit "
                    f"hallucination(s)[/yellow]"
                )
                result = TranscriptResult(
                    language=result.language,
                    duration=result.duration,
                    segments=kept,
                )

    console.print(
        f"  language=[green]{result.language}[/green] "
        f"duration={result.duration:.1f}s segments={len(result.segments)} "
        f"preset=[magenta]{chosen}[/magenta] "
        f"separator=[magenta]{separator}[/magenta]"
    )

    summary: str | None = None
    if do_summary:
        console.print(f"[bold cyan]Summarizing[/bold cyan] with {cfg.ollama_model}")
        transcript_text = "\n".join(seg.text for seg in result.segments if seg.text)
        summary = summarize(transcript_text, result.language, cfg)

    md = render(
        source_file=audio_path,
        result=result,
        summary=summary,
        whisper_model=cfg.whisper_model,
        ollama_model=cfg.ollama_model if do_summary else None,
        audio_preset=chosen,
        audio_metrics=audio_metrics_dict,
    )
    output_path.write_text(md, encoding="utf-8")
    console.print(f"[bold green]Wrote[/bold green] {output_path}")
    return output_path


def process_directory(
    directory: Path,
    cfg: Config,
    *,
    language: str | None = None,
    do_summary: bool = True,
    diarize: bool = True,
    force: bool = False,
    output_dir: Path = OUTPUT_DIR,
    audio_preset: str = "auto",
    separator: str = "none",
) -> list[Path]:
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        console.print(f"[yellow]No audio files found in {directory}[/yellow]")
        return []
    console.print(f"Found [bold]{len(files)}[/bold] audio file(s) in {directory}")
    outputs: list[Path] = []
    for f in files:
        outputs.append(
            process_file(
                f,
                cfg,
                language=language,
                do_summary=do_summary,
                diarize=diarize,
                force=force,
                output_dir=output_dir,
                audio_preset=audio_preset,
                separator=separator,
            )
        )
    return outputs
