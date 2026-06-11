"""End-to-end pipeline: audio file → Markdown transcript."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .config import Config, OUTPUT_DIR
from .render import render
from .summarize import summarize
from .transcribe import transcribe

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

console = Console()


def process_file(
    audio_path: Path,
    cfg: Config,
    *,
    language: str | None = None,
    do_summary: bool = True,
    diarize: bool = True,
    force: bool = False,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Transcribe one audio file and write a Markdown result. Returns the output path."""
    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{audio_path.stem}.md"

    if output_path.exists() and not force:
        console.print(f"[yellow]Skipping[/yellow] {audio_path.name} (output exists, use --force)")
        return output_path

    console.print(f"[bold cyan]Transcribing[/bold cyan] {audio_path.name}")
    result = transcribe(audio_path, cfg, language=language, diarize=diarize)
    console.print(
        f"  language=[green]{result.language}[/green] "
        f"duration={result.duration:.1f}s segments={len(result.segments)}"
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
            )
        )
    return outputs
