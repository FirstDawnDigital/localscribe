"""Configuration loaded from environment and .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Config:
    hf_token: str | None
    ollama_model: str
    ollama_url: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            hf_token=os.environ.get("HF_TOKEN") or None,
            ollama_model=os.environ.get("OLLAMA_MODEL", "gemma4:12b-mlx"),
            ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            whisper_model=os.environ.get("WHISPER_MODEL", "large-v3"),
            whisper_device=os.environ.get("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
        )
