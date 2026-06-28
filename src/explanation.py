from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .inference import medgemma_predict
from .medgemma import DEFAULT_MODEL_ID, MedGemmaBackend
from .prompting import load_prompt


ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}


def hash_image(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def agreement_status(primary_class: str, independent_class: str) -> str:
    if primary_class not in ALLOWED_CLASSES:
        raise ValueError(f"Unknown primary class: {primary_class}")
    if independent_class not in ALLOWED_CLASSES:
        raise ValueError(f"Unknown independent class: {independent_class}")
    if primary_class == independent_class:
        return "agreement"
    if "uncertain" in {primary_class, independent_class}:
        return "partial_disagreement"
    return "disagreement"


class ExplanationService:
    """Generate and cache an independent MedGemma image commentary."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        model_backend: MedGemmaBackend | None = None,
    ):
        default_cache = os.getenv(
            "EXPLANATION_CACHE_DIR",
            ".cache/arvi_explanations",
        )
        self.cache_dir = Path(cache_dir or default_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_backend = model_backend

    def explain(self, image_path: str | Path, primary_class: str) -> dict[str, Any]:
        prompt, prompt_version = load_prompt("explanation")
        model_id = getattr(self.model_backend, "model_id", None) or os.getenv(
            "MEDGEMMA_MODEL_ID",
            DEFAULT_MODEL_ID,
        )
        image_digest = hash_image(image_path)
        cache_material = f"{image_digest}:{model_id}:{prompt_version}"
        cache_key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            analysis = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = True
        else:
            analysis = medgemma_predict(
                image_path,
                prompt,
                prompt_version,
                self.model_backend,
            )
            cache_path.write_text(
                json.dumps(analysis, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            cached = False

        independent_class = analysis["predicted_class"]
        return {
            "decision_source": "medsiglip_only",
            "primary_class": primary_class,
            "primary_class_unchanged": True,
            "agreement_status": agreement_status(
                primary_class,
                independent_class,
            ),
            "image_sha256": image_digest,
            "cached": cached,
            "analysis": analysis,
        }
