from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import time
from typing import Any

from .guardrails import apply_safety_guardrails
from .onnx_backend import OnnxClassifierBackend
from .preprocessing import basic_quality_flag
from .preprocessing import load_image

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


@lru_cache(maxsize=8)
def get_onnx_backend(config_path: str | None = None) -> OnnxClassifierBackend:
    """Create one shared ONNX Runtime session for each configured variant."""

    return OnnxClassifierBackend(config_path=config_path)


def predict(
    image_path: str | Path,
    mode: str = "baseline",
    backend: str | None = None,
    model_backend: Any | None = None,
    backend_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Common inference entry point for notebooks, API and user interfaces."""

    selected_backend = (backend or os.getenv("MODEL_BACKEND", "onnx")).lower()
    if selected_backend == "toy":
        return apply_safety_guardrails(toy_predict(image_path, mode=mode))
    if selected_backend == "onnx":
        config_cache_key = str(backend_config_path) if backend_config_path is not None else None
        selected_model = model_backend or get_onnx_backend(config_cache_key)
        image = load_image(image_path, size=None)
        return apply_safety_guardrails(selected_model.predict(image))
    raise ValueError(f"Unknown model backend: {selected_backend}")
