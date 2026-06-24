from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import time
from typing import Any

from .guardrails import apply_safety_guardrails
from .medgemma import MedGemmaBackend
from .preprocessing import basic_quality_flag
from .preprocessing import load_image
from .prompting import load_prompt
from .schemas import PredictionParseError, parse_prediction_json

WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Deterministic toy predictor used to validate the repo pipeline.

    It reads synthetic labels from filenames. This is not medical inference.
    """
    start = time.perf_counter()
    name = Path(image_path).name.lower()
    quality = basic_quality_flag(image_path)

    if "suspected_opacity" in name:
        pred = "suspected_opacity"
        conf = 0.78 if mode == "baseline" else 0.72
        evidence = ["synthetic opacity-like area visible in the lung field"]
        justification = "The synthetic image contains a localized brighter region compatible with the toy opacity class. This is a pipeline validation result, not a medical interpretation."
    elif "normal" in name:
        pred = "normal"
        conf = 0.72 if mode == "baseline" else 0.68
        evidence = ["no synthetic opacity marker detected"]
        justification = "The synthetic image does not contain the opacity marker used by the toy generator. This conclusion is limited to the synthetic validation setting."
    else:
        pred = "uncertain"
        conf = 0.52
        evidence = ["limited synthetic image quality"]
        justification = "The image is treated as limited quality in the toy catalog. The safe output is uncertainty rather than a forced class."

    # Improved mode is more conservative.
    if mode == "improved" and quality != "good":
        pred = "uncertain"
        conf = min(conf, 0.55)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": quality,
        "predicted_class": pred,
        "confidence": round(float(conf), 3),
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": ["synthetic toy image", "no clinical context", "not a validated medical model"],
        "warning": WARNING,
        "model_name": f"toy-rule-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


def vlm_predict_placeholder(image_path: str | Path, prompt: str) -> dict[str, Any]:
    """Backward-compatible entry point for a custom MedGemma prompt."""

    return medgemma_predict(image_path, prompt, prompt_version="custom_v1")


@lru_cache(maxsize=1)
def get_medgemma_backend() -> MedGemmaBackend:
    """Create one shared backend and load model weights only on first use."""

    return MedGemmaBackend()


def _uncertain_model_output(reason: str) -> dict[str, Any]:
    return {
        "image_quality": "limited",
        "predicted_class": "uncertain",
        "confidence": 0.0,
        "visual_evidence": [],
        "justification": "The model response could not be validated, so no image interpretation is returned.",
        "limitations": [reason, "uncalibrated generative model output"],
        "warning": WARNING,
    }


def medgemma_predict(
    image_path: str | Path,
    prompt: str,
    prompt_version: str,
    model_backend: MedGemmaBackend | None = None,
) -> dict[str, Any]:
    """Run MedGemma, validate its JSON, and apply the safety contract."""

    start = time.perf_counter()
    backend = model_backend or get_medgemma_backend()
    # The model processor owns resizing and normalization for real inference.
    image = load_image(image_path, size=None)
    raw_output = backend.generate(image, prompt)
    try:
        prediction = parse_prediction_json(raw_output)
    except PredictionParseError as error:
        prediction = _uncertain_model_output(str(error))

    prediction.update(
        {
            "model_name": backend.model_id,
            "prompt_version": prompt_version,
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "confidence_type": "model_reported_uncalibrated",
        }
    )
    return apply_safety_guardrails(prediction)


def predict(
    image_path: str | Path,
    mode: str = "baseline",
    backend: str | None = None,
    model_backend: MedGemmaBackend | None = None,
) -> dict[str, Any]:
    """Common inference entry point for notebooks, API and user interfaces."""

    selected_backend = (backend or os.getenv("MODEL_BACKEND", "toy")).lower()
    if selected_backend == "toy":
        return apply_safety_guardrails(toy_predict(image_path, mode=mode))
    if selected_backend == "medgemma":
        prompt, prompt_version = load_prompt(mode)
        return medgemma_predict(image_path, prompt, prompt_version, model_backend)
    raise ValueError(f"Unknown model backend: {selected_backend}")
