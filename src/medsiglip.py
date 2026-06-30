from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "medsiglip_zero_shot_v1.json"
WARNING = (
    "Prototype pédagogique. Non destiné au diagnostic. "
    "Validation par un professionnel qualifié requise."
)


def classify_opacity_score(
    score: float,
    low_threshold: float,
    high_threshold: float,
) -> str:
    """Convert the frozen relative opacity score into the project classes."""

    if not 0 <= score <= 1:
        raise ValueError("Opacity score must be inside [0, 1]")
    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= low < high <= 1")
    if score >= high_threshold:
        return "suspected_opacity"
    if score <= low_threshold:
        return "normal"
    return "uncertain"


class MedSigLIPBackend:
    """Lazy implementation of the frozen MedSigLIP zero-shot classifier."""

    def __init__(self, config_path: str | Path | None = None):
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.config = json.loads(path.read_text(encoding="utf-8"))
        self.model_id = self.config["model_id"]
        self.model_version = self.config["version"]
        self.normal_prompts = list(self.config["normal_prompts"])
        self.opacity_prompts = list(self.config["opacity_prompts"])
        self.low_threshold = float(self.config["low_threshold"])
        self.high_threshold = float(self.config["high_threshold"])
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._dtype: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "MedSigLIP requires torch, transformers and accelerate."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError(
                "MedSigLIP live inference requires a CUDA GPU in this deployment."
            )

        token = os.getenv("HF_TOKEN") or None
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                token=token,
            )
            self._model = AutoModel.from_pretrained(
                self.model_id,
                token=token,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device).eval()
        except Exception as error:
            raise RuntimeError(
                f"Unable to load {self.model_id}. Check model access, HF_TOKEN and GPU memory: {error}"
            ) from error

        self._torch = torch
        self._device = device
        self._dtype = dtype

    def score(self, image: Image.Image) -> float:
        """Return the frozen relative similarity score for lung opacity."""

        self._load()
        prompts = self.normal_prompts + self.opacity_prompts
        inputs = self._processor(
            text=prompts,
            images=[image.convert("RGB")],
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)

        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        logits = outputs.logits_per_image.float()
        normal_count = len(self.normal_prompts)
        normal_logit = logits[:, :normal_count].mean(dim=1)
        opacity_logit = logits[:, normal_count:].mean(dim=1)
        paired_logits = self._torch.stack([normal_logit, opacity_logit], dim=1)
        return float(self._torch.softmax(paired_logits, dim=1)[0, 1].cpu())

    def predict(self, image: Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        score = self.score(image)
        predicted_class = classify_opacity_score(
            score,
            self.low_threshold,
            self.high_threshold,
        )
        return {
            "predicted_class": predicted_class,
            "score_opacity": score,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "confidence_type": self.config["score_type"],
            "model_name": self.model_id,
            "model_version": self.model_version,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "warning": WARNING,
        }
