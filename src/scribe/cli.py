"""Command-line interface for SCRIBE."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import Config, INPUT_DIR, OUTPUT_DIR
from .pipeline import process_directory, process_file

app = typer.Typer(help="Local interview transcription with diarization and summary.")


@app.command()
def transcribe(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Force language (e.g. da, en). Default: auto-detect."),
    no_summary: bool = typer.Option(False, "--no-summary", help="Skip LLM summarization."),
    no_diarize: bool = typer.Option(False, "--no-diarize", help="Skip speaker diarization (no HF token needed)."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing output."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Whisper model override (e.g. medium, large-v3)."),
    output_dir: Path = typer.Option(OUTPUT_DIR, "--output-dir", "-o"),
) -> None:
    """Transcribe a single audio file to Markdown."""
    cfg = Config.from_env()
    if model:
        cfg = Config(**{**cfg.__dict__, "whisper_model": model})
    process_file(
        file,
        cfg,
        language=language,
        do_summary=not no_summary,
        diarize=not no_diarize,
        force=force,
        output_dir=output_dir,
    )


@app.command()
def batch(
    directory: Path = typer.Option(INPUT_DIR, "--dir", "-d", exists=True, file_okay=False),
    language: Optional[str] = typer.Option(None, "--language", "-l"),
    no_summary: bool = typer.Option(False, "--no-summary"),
    no_diarize: bool = typer.Option(False, "--no-diarize"),
    force: bool = typer.Option(False, "--force", "-f"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    output_dir: Path = typer.Option(OUTPUT_DIR, "--output-dir", "-o"),
) -> None:
    """Transcribe every supported audio file in a directory."""
    cfg = Config.from_env()
    if model:
        cfg = Config(**{**cfg.__dict__, "whisper_model": model})
    process_directory(
        directory,
        cfg,
        language=language,
        do_summary=not no_summary,
        diarize=not no_diarize,
        force=force,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    app()
