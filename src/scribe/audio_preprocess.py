"""ffmpeg-based audio restoration presets.

Design rule: chain is highpass → (dereverb) → (denoise) → dynaudnorm → loudnorm.
We only use ffmpeg-native filters here so there are no new Python deps. AI
restoration (DeepFilterNet, WPE, Resemble Enhance) can be plugged in later as
optional extras.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

PRESETS: dict[str, list[str]] = {
    # No processing — pass-through. Useful for A/B comparisons.
    "none": [],
    # Safe-on-everything: kill subsonic rumble, normalize loudness.
    "clean": [
        "highpass=f=80",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
    ],
    # Conference/meeting: also level uneven speakers.
    "meeting": [
        "highpass=f=80",
        "dynaudnorm=f=150:g=15:p=0.95",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
    ],
    # Noisy: deprecated — afftdn produced repetition-loop hallucinations on
    # our conference sample (see vad_sweep results 2026-06-12). Kept as alias
    # for 'meeting' until we plug in a real AI denoiser (DeepFilterNet).
    "noisy": [
        "highpass=f=80",
        "dynaudnorm=f=150:g=15:p=0.95",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
    ],
}


def apply_preset(
    input_path: Path,
    preset: str,
    output_path: Path,
    *,
    sample_rate: int = 16000,
    overwrite: bool = True,
) -> Path:
    """Run the preset's filter chain via ffmpeg, writing a 16kHz mono WAV.

    If preset == 'none', we copy the file unchanged (still resampled to 16k mono
    so downstream Whisper input is uniform).
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from {list(PRESETS)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if overwrite:
        cmd.append("-y")
    cmd += ["-i", str(input_path)]

    filters = PRESETS[preset]
    if filters:
        cmd += ["-af", ",".join(filters)]

    cmd += [
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH — install it first")
