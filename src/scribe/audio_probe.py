"""Audio analysis via ffmpeg — no extra deps. Returns metrics used to decide which
restoration filters to apply.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AudioMetrics:
    path: str
    duration_s: float
    mean_volume_db: float | None  # ffmpeg volumedetect, dBFS
    max_volume_db: float | None
    lufs_i: float | None          # integrated loudness (EBU R128)
    lufs_lra: float | None        # loudness range
    true_peak_db: float | None
    silence_ratio: float | None   # fraction of duration below -40 dBFS for >=0.5s
    clip_pct: float | None        # % of samples at full scale


def _ffprobe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def _volumedetect(path: Path) -> tuple[float | None, float | None]:
    """Return (mean_volume_db, max_volume_db) from ffmpeg volumedetect."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    text = r.stderr
    mean = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", text)
    peak = re.search(r"max_volume:\s*(-?[\d.]+)\s*dB", text)
    return (
        float(mean.group(1)) if mean else None,
        float(peak.group(1)) if peak else None,
    )


def _loudnorm_analyze(path: Path) -> tuple[float | None, float | None, float | None]:
    """First-pass loudnorm — returns (LUFS-I, LRA, true_peak_dbtp)."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    # ffmpeg prints the JSON block at the end of stderr.
    m = re.search(r"\{[^{}]*\"input_i\".*?\}", r.stderr, re.DOTALL)
    if not m:
        return None, None, None
    try:
        data = json.loads(m.group(0))
        return (
            float(data["input_i"]),
            float(data["input_lra"]),
            float(data["input_tp"]),
        )
    except (KeyError, ValueError):
        return None, None, None


def _silence_ratio(path: Path, duration_s: float,
                   noise_db: float = -40.0, min_dur: float = 0.5) -> float | None:
    """Fraction of audio classified as silence by ffmpeg silencedetect."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    total = 0.0
    for m in re.finditer(r"silence_duration:\s*([\d.]+)", r.stderr):
        total += float(m.group(1))
    if duration_s <= 0:
        return None
    return min(1.0, total / duration_s)


def _clip_pct(path: Path) -> float | None:
    """Estimate clipping % from astats Peak_count + sample count."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "astats=metadata=0:measure_perchannel=none:measure_overall=Peak_count+Number_of_samples",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    text = r.stderr
    peaks = re.search(r"Peak count:\s*(\d+)", text)
    samples = re.search(r"Number of samples:\s*(\d+)", text)
    if not (peaks and samples):
        return None
    n = int(samples.group(1))
    if n == 0:
        return None
    return 100.0 * int(peaks.group(1)) / n


def probe(path: Path) -> AudioMetrics:
    """Run full probe on an audio file. Cheap: ~5-15 s for a 5-min file."""
    duration = _ffprobe_duration(path)
    mean_db, peak_db = _volumedetect(path)
    lufs_i, lufs_lra, true_peak = _loudnorm_analyze(path)
    silence = _silence_ratio(path, duration)
    clip = _clip_pct(path)
    return AudioMetrics(
        path=str(path),
        duration_s=duration,
        mean_volume_db=mean_db,
        max_volume_db=peak_db,
        lufs_i=lufs_i,
        lufs_lra=lufs_lra,
        true_peak_db=true_peak,
        silence_ratio=silence,
        clip_pct=clip,
    )


def format_report(metrics: AudioMetrics) -> str:
    name = Path(metrics.path).name
    lines = [
        f"  {name}",
        f"    duration       : {metrics.duration_s:6.1f} s",
        f"    mean / max vol : {_fmt(metrics.mean_volume_db)} / {_fmt(metrics.max_volume_db)} dB",
        f"    LUFS-I / LRA   : {_fmt(metrics.lufs_i)} / {_fmt(metrics.lufs_lra)} LU",
        f"    true peak      : {_fmt(metrics.true_peak_db)} dBTP",
        f"    silence ratio  : {_fmt_pct(metrics.silence_ratio)}",
        f"    clipping       : {_fmt_pct(metrics.clip_pct, scale=False)}",
    ]
    return "\n".join(lines)


def _fmt(x: float | None) -> str:
    return f"{x:6.1f}" if x is not None else "  n/a "


def _fmt_pct(x: float | None, scale: bool = True) -> str:
    if x is None:
        return " n/a"
    return f"{x * 100:5.1f} %" if scale else f"{x:5.2f} %"


def to_json(metrics_list: list[AudioMetrics]) -> str:
    return json.dumps([asdict(m) for m in metrics_list], indent=2)
