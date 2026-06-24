from __future__ import annotations

from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPTS = {
    "baseline": ("baseline_prompt.txt", "baseline_v1"),
    "improved_v1": ("improved_v1_prompt.txt", "improved_v1"),
    "improved_v2": ("improved_v2_prompt.txt", "improved_v2"),
    "improved_v3": ("improved_prompt.txt", "improved_v3"),
    "improved": ("improved_prompt.txt", "improved_v3"),
}


def load_prompt(mode: str) -> tuple[str, str]:
    """Return prompt text and its stable version identifier."""

    try:
        filename, version = PROMPTS[mode]
    except KeyError as error:
        raise ValueError(f"Unknown prompt mode: {mode}") from error
    return (PROMPT_DIR / filename).read_text(encoding="utf-8"), version
