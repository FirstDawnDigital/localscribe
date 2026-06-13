"""Reference-scored evaluation of SCRIBE transcripts using meeteval (tcpWER).

Workflow:

    1. Pick a representative recording, transcribe it with SCRIBE.
    2. `scribe eval bootstrap -i output/foo.md -o references/foo.ref.stm`
       This writes a starting-point STM with the hypothesis text and timings.
    3. Open the .stm in an editor (and the audio in any player) and correct
       the words, timestamps and speaker labels until they match what is
       actually said. This is your ground-truth reference.
    4. Re-run SCRIBE with whatever change you want to evaluate (model swap,
       preset change, ...) and produce a new `output/foo.md`.
    5. `scribe eval score -r references/foo.ref.stm -i output/foo.md`
       Prints tcpWER + speaker-attribution errors so you can compare runs.

meeteval is an optional dependency: `pip install scribe[eval]`.
"""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_TS_RE = re.compile(
    r"^\*\*\[(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\]"
    r"(?:\s+(?P<spk>[A-Za-z0-9_]+):)?\*\*\s*(?P<text>.*)$"
)


@dataclass
class SegLstEntry:
    session_id: str
    start_time: float
    end_time: float
    speaker: str
    words: str

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "speaker": self.speaker,
            "words": self.words,
        }


def _hms_to_seconds(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s)


def _sanitize_session_id(s: str) -> str:
    """STM is whitespace-delimited, so session ids must not contain spaces or
    other tokens that confuse the parser."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "session"


def parse_scribe_markdown(md_path: Path, session_id: str | None = None) -> list[SegLstEntry]:
    """Parse a SCRIBE-rendered Markdown transcript back into SegLST entries.

    Recognises both formats produced by ``render.render``::

        **[HH:MM:SS] SPEAKER_XX:** text...
        **[HH:MM:SS]** text...   (continuation of previous speaker)

    The end_time of each segment is approximated as the start_time of the next
    segment (or +duration of a typical segment for the last one). This is
    deliberately coarse — for tcpWER with a 5s collar the exact boundary
    barely matters, but speaker attribution does.
    """
    session = _sanitize_session_id(session_id or md_path.stem)
    text = md_path.read_text(encoding="utf-8")

    # Strip frontmatter if present
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]

    entries: list[SegLstEntry] = []
    current_speaker: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        m = _TS_RE.match(line)
        if not m:
            continue
        start = _hms_to_seconds(m["h"], m["m"], m["s"])
        spk = m["spk"]
        if spk:
            current_speaker = spk
        if current_speaker is None:
            # Continuation line before any speaker tag — skip rather than
            # invent a speaker.
            continue
        words = m["text"].strip()
        if not words:
            continue
        entries.append(
            SegLstEntry(
                session_id=session,
                start_time=float(start),
                end_time=float(start),  # patched below
                speaker=current_speaker,
                words=words,
            )
        )

    # Fill end_time = next entry start, with a 3s pad on the last one
    for i, e in enumerate(entries[:-1]):
        e.end_time = max(entries[i + 1].start_time, e.start_time + 0.5)
    if entries:
        entries[-1].end_time = entries[-1].start_time + 3.0
    return entries


def write_seglst(entries: Iterable[SegLstEntry], path: Path) -> None:
    path.write_text(
        json.dumps([e.to_dict() for e in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_stm(entries: Iterable[SegLstEntry], path: Path) -> None:
    """Write entries as NIST STM (one line per segment).

    Format: ``<session> 1 <speaker> <start> <end> <words>``.
    meeteval reads this directly and it's the easiest format for a human to
    hand-edit.
    """
    lines = []
    for e in entries:
        lines.append(
            f"{e.session_id} 1 {e.speaker} "
            f"{e.start_time:.2f} {e.end_time:.2f} {e.words}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_with_meeteval(path: Path):
    """Return a meeteval transcript object from .stm / .seglst.json / .md."""
    try:
        from meeteval.io import load  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "meeteval is not installed. Install with: pip install scribe[eval]"
        ) from e

    suffix = path.suffix.lower()
    if suffix == ".md":
        # Convert on the fly via a temp STM file. We use STM (not SegLST JSON)
        # because meeteval's SegLST loader requires the optional `simplejson`
        # package, while its STM loader has no extra dependency. We write to
        # a temp dir to avoid colliding with any user-managed reference next
        # to the markdown file.
        entries = parse_scribe_markdown(path)
        if not entries:
            raise ValueError(f"No transcript segments parsed from {path}")
        tmp_dir = Path(tempfile.mkdtemp(prefix="scribe_eval_"))
        tmp = tmp_dir / f"{path.stem}.stm"
        write_stm(entries, tmp)
        return load(tmp, parse_float=float)
    return load(path, parse_float=float)


def score_tcpwer(reference: Path, hypothesis: Path, collar: float = 5.0) -> dict:
    """Compute tcpWER between reference and hypothesis transcripts.

    Both arguments accept .stm, .seglst.json, or SCRIBE .md files.
    Returns a dict with the aggregated meeteval result.
    """
    try:
        from meeteval.wer import tcpwer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "meeteval is not installed. Install with: pip install scribe[eval]"
        ) from e

    ref = _load_with_meeteval(reference)
    hyp = _load_with_meeteval(hypothesis)
    # We only have segment-level timings (one entry per Whisper segment, not
    # per word). Ask meeteval to interpolate per-word timestamps by spreading
    # them evenly across each segment's character span — the standard
    # workaround for ASR systems that don't emit word alignments.
    result = tcpwer(
        reference=ref,
        hypothesis=hyp,
        collar=collar,
        ref_pseudo_word_timing="character_based",
        hyp_pseudo_word_timing="character_based",
    )

    # meeteval returns a dict-like keyed by session; aggregate
    if hasattr(result, "values"):
        sessions = list(result.values())
    else:
        sessions = [result]

    total_errors = sum(getattr(s, "errors", 0) for s in sessions)
    total_length = sum(getattr(s, "length", 0) for s in sessions)
    total_ins = sum(getattr(s, "insertions", 0) for s in sessions)
    total_del = sum(getattr(s, "deletions", 0) for s in sessions)
    total_sub = sum(getattr(s, "substitutions", 0) for s in sessions)
    total_missed_spk = sum(getattr(s, "missed_speaker", 0) for s in sessions)
    total_false_spk = sum(getattr(s, "falarm_speaker", 0) for s in sessions)

    wer = (total_errors / total_length) if total_length else float("nan")

    return {
        "tcpwer": wer,
        "errors": total_errors,
        "length": total_length,
        "insertions": total_ins,
        "deletions": total_del,
        "substitutions": total_sub,
        "missed_speaker": total_missed_spk,
        "falarm_speaker": total_false_spk,
        "sessions": len(sessions),
        "collar": collar,
    }
