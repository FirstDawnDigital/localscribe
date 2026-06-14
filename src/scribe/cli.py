"""Command-line interface for SCRIBE."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import Config, INPUT_DIR, OUTPUT_DIR
from .pipeline import process_directory, process_file

app = typer.Typer(help="Local interview transcription with diarization and summary.")
eval_app = typer.Typer(help="Score SCRIBE transcripts against a manual reference (tcpWER).")
app.add_typer(eval_app, name="eval")


@app.command()
def transcribe(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Force language (e.g. da, en). Default: auto-detect."),
    no_summary: bool = typer.Option(False, "--no-summary", help="Skip LLM summarization."),
    no_diarize: bool = typer.Option(False, "--no-diarize", help="Skip speaker diarization (no HF token needed)."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing output."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Whisper model override (e.g. medium, large-v3)."),
    output_dir: Path = typer.Option(OUTPUT_DIR, "--output-dir", "-o"),
    audio_preset: str = typer.Option(
        "auto", "--audio-preset", "-a",
        help="Audio preprocessing: none | clean | meeting | auto (probe & decide).",
    ),
    separator: str = typer.Option(
        "none", "--separator", "-s",
        help="Speech separation pre-stage: none | pixit. 'pixit' replaces pyannote diarization.",
    ),
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
        audio_preset=audio_preset,
        separator=separator,
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
    audio_preset: str = typer.Option(
        "auto", "--audio-preset", "-a",
        help="Audio preprocessing: none | clean | meeting | auto (probe & decide).",
    ),
    separator: str = typer.Option(
        "none", "--separator", "-s",
        help="Speech separation pre-stage: none | pixit.",
    ),
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
        audio_preset=audio_preset,
        separator=separator,
    )


@eval_app.command("bootstrap")
def eval_bootstrap(
    input_md: Path = typer.Option(..., "--input", "-i", exists=True, dir_okay=False, readable=True, help="SCRIBE Markdown transcript to seed the reference from."),
    output: Path = typer.Option(..., "--output", "-o", help="Where to write the editable reference (.stm or .seglst.json)."),
    session_id: Optional[str] = typer.Option(None, "--session-id", help="Override the session id (default: input filename stem)."),
) -> None:
    """Create a starting-point reference from a hypothesis transcript.

    The output file is meant to be hand-edited: fix the words, tighten the
    timestamps and correct speaker labels until it matches what is actually
    said in the audio. Then use `scribe eval score` to grade future runs
    against it.
    """
    from .eval import parse_scribe_markdown, write_seglst, write_stm

    entries = parse_scribe_markdown(input_md, session_id=session_id)
    if not entries:
        typer.echo("No transcript segments parsed from input.", err=True)
        raise typer.Exit(code=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".stm":
        write_stm(entries, output)
    elif suffix in (".json", ".seglst.json") or output.name.endswith(".seglst.json"):
        write_seglst(entries, output)
    else:
        typer.echo(f"Unsupported output extension: {suffix}. Use .stm or .seglst.json.", err=True)
        raise typer.Exit(code=2)

    typer.echo(f"Wrote {len(entries)} segments to {output}")
    typer.echo("Now hand-edit the file to match the ground truth, then run `scribe eval score`.")


@eval_app.command("score")
def eval_score(
    reference: Path = typer.Option(..., "--reference", "-r", exists=True, dir_okay=False, readable=True, help="Reference transcript (.stm, .seglst.json, or .md)."),
    hypothesis: Path = typer.Option(..., "--input", "-i", exists=True, dir_okay=False, readable=True, help="Hypothesis transcript to grade (.stm, .seglst.json, or SCRIBE .md)."),
    collar: float = typer.Option(5.0, "--collar", help="Timing collar in seconds for tcpWER (default 5.0)."),
) -> None:
    """Score a hypothesis against a reference using meeteval's tcpWER."""
    try:
        from .eval import score_tcpwer
    except ImportError as e:
        typer.echo(f"Failed to import eval module: {e}", err=True)
        raise typer.Exit(code=1)

    try:
        result = score_tcpwer(reference, hypothesis, collar=collar)
    except RuntimeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    typer.echo(f"=== tcpWER (collar={result['collar']}s) ===")
    typer.echo(f"  tcpWER:           {result['tcpwer'] * 100:6.2f}%")
    typer.echo(f"  Reference words:  {result['length']}")
    typer.echo(f"  Errors:           {result['errors']}")
    typer.echo(f"    substitutions:  {result['substitutions']}")
    typer.echo(f"    insertions:     {result['insertions']}")
    typer.echo(f"    deletions:      {result['deletions']}")
    typer.echo(f"  Missed speakers:  {result['missed_speaker']}")
    typer.echo(f"  False speakers:   {result['falarm_speaker']}")
    typer.echo(f"  Sessions scored:  {result['sessions']}")


if __name__ == "__main__":
    app()
