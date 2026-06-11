"""Local LLM summarization via Ollama HTTP API."""
from __future__ import annotations

import httpx

from .config import Config

_PROMPTS = {
    "da": (
        "Du er en assistent der opsummerer interview-transskriptioner. "
        "Skriv et koncist resumé på dansk med:\n"
        "1. 1-2 sætninger der beskriver interviewets overordnede emne.\n"
        "2. 5-8 punktopstillede nøglepointer.\n"
        "3. En kort liste over centrale temaer.\n\n"
        "Transskription:\n{text}\n\nResumé:"
    ),
    "en": (
        "You are an assistant that summarizes interview transcripts. "
        "Write a concise summary in English with:\n"
        "1. 1-2 sentences describing the overall topic.\n"
        "2. 5-8 bullet points with key takeaways.\n"
        "3. A short list of central themes.\n\n"
        "Transcript:\n{text}\n\nSummary:"
    ),
}


def summarize(transcript_text: str, language: str, cfg: Config) -> str:
    """Generate a summary in the given language via Ollama."""
    template = _PROMPTS.get(language, _PROMPTS["en"])
    prompt = template.format(text=transcript_text)

    response = httpx.post(
        f"{cfg.ollama_url}/api/generate",
        json={
            "model": cfg.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=600.0,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
