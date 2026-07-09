from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "onnx_u_ones.json"
ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
WARNING = (
    "Prototype pédagogique. Non destiné au diagnostic. "
    "Validation par un professionnel qualifié requise."
)


class OnnxConfigurationError(ValueError):
    """Raised when the ONNX integration contract is incomplete or invalid."""


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


class OnnxClassifierBackend:
    """Configuration-driven adapter for the classifier delivered as ONNX.

    The session can be injected by tests, so the integration contract is
    verifiable before the real model file is available.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        model_path: str | Path | None = None,
        session: Any | None = None,
    ):
        path = Path(
            config_path or os.getenv("ONNX_CONFIG_PATH", DEFAULT_CONFIG_PATH)
        )
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise OnnxConfigurationError(f"ONNX configuration not found: {path}")

        self.config_path = path
        self.config = json.loads(path.read_text(encoding="utf-8"))
        configured_model = (
            model_path
            or os.getenv("ONNX_MODEL_PATH")
            or self.config.get("model_path")
        )
        if not configured_model:
            raise OnnxConfigurationError(
                "Set model_path in the ONNX configuration or ONNX_MODEL_PATH."
            )
        self.model_path = Path(configured_model)
        if not self.model_path.is_absolute():
            self.model_path = ROOT / self.model_path
        self.companion_paths = []
        for configured_companion in self.config.get("companion_files", []):
            companion_path = Path(configured_companion)
            if not companion_path.is_absolute():
                companion_path = self.model_path.parent / companion_path
            self.companion_paths.append(companion_path)

        self.model_name = str(self.config.get("model_name", "arvi-onnx-classifier"))
        self.model_version = str(self.config.get("version", "onnx_unversioned"))
        self.class_names = list(self.config.get("class_names", []))
        self.input_size = tuple(self.config.get("input_size", []))
        self.input_layout = str(self.config.get("input_layout", "NCHW")).upper()
        self.color_mode = str(self.config.get("color_mode", "RGB")).upper()
        self.resize_resample = str(
            self.config.get("resize_resample", "bilinear")
        ).lower()
        self.output_type = str(self.config.get("output_type", "logits")).lower()
        self.confidence_threshold = float(
            self.config.get("confidence_threshold", 0.0)
        )
        self.min_top2_margin = float(self.config.get("min_top2_margin", 0.0))
        self._session = session
        self._validate_config()

    def missing_model_files(self) -> list[Path]:
        """Return every graph or external-data file required for inference."""

        required_paths = [self.model_path, *self.companion_paths]
        return [path for path in required_paths if not path.is_file()]

    def _validate_config(self) -> None:
        if len(self.input_size) != 2 or any(int(value) <= 0 for value in self.input_size):
            raise OnnxConfigurationError("input_size must contain two positive integers")
        if self.input_layout not in {"NCHW", "NHWC"}:
            raise OnnxConfigurationError("input_layout must be NCHW or NHWC")
        if self.color_mode not in {"RGB", "L"}:
            raise OnnxConfigurationError("color_mode must be RGB or L")
        if self.resize_resample not in {"nearest", "bilinear", "bicubic", "lanczos"}:
            raise OnnxConfigurationError(
                "resize_resample must be nearest, bilinear, bicubic or lanczos"
            )
        if self.output_type not in {"logits", "probabilities"}:
            raise OnnxConfigurationError("output_type must be logits or probabilities")
        if len(self.class_names) < 2 or not set(self.class_names) <= ALLOWED_CLASSES:
            raise OnnxConfigurationError(
                "class_names must contain at least two project classes"
            )
        if len(set(self.class_names)) != len(self.class_names):
            raise OnnxConfigurationError("class_names must not contain duplicates")
        if not 0 <= self.confidence_threshold <= 1:
            raise OnnxConfigurationError("confidence_threshold must be inside [0, 1]")
        if not 0 <= self.min_top2_margin <= 1:
            raise OnnxConfigurationError("min_top2_margin must be inside [0, 1]")

        normalization = self.config.get("normalization", {})
        channels = 3 if self.color_mode == "RGB" else 1
        mean = normalization.get("mean", [0.0] * channels)
        std = normalization.get("std", [1.0] * channels)
        if (
            len(mean) != channels
            or len(std) != channels
            or any(float(value) == 0 for value in std)
        ):
            raise OnnxConfigurationError(
                "normalization mean/std must match color_mode and contain non-zero std values"
            )

    def _load(self) -> None:
        if self._session is not None:
            return
        missing_files = self.missing_model_files()
        if missing_files:
            missing_names = ", ".join(path.name for path in missing_files)
            raise RuntimeError(
                f"ONNX model files not found: {missing_names}. "
                "Place the graph and every external-data companion file together."
            )
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "ONNX inference requires onnxruntime. Install the project requirements."
            ) from error

        configured_providers = self.config.get("providers")
        providers = configured_providers or ["CPUExecutionProvider"]
        try:
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
            )
        except Exception as error:
            raise RuntimeError(f"Unable to load ONNX model: {error}") from error

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Apply the exact image contract recorded beside the exported model."""

        height, width = (int(value) for value in self.input_size)
        resampling = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }[self.resize_resample]
        image_array = np.asarray(
            image.convert(self.color_mode).resize((width, height), resampling),
            dtype=np.float32,
        )
        if image_array.ndim == 2:
            image_array = image_array[:, :, np.newaxis]
        normalization = self.config.get("normalization", {})
        if normalization.get("scale", "zero_one") == "zero_one":
            image_array /= 255.0
        elif normalization.get("scale") not in {None, "none"}:
            raise OnnxConfigurationError(
                "normalization.scale must be zero_one or none"
            )

        channels = image_array.shape[-1]
        mean = np.asarray(
            normalization.get("mean", [0.0] * channels),
            dtype=np.float32,
        )
        std = np.asarray(
            normalization.get("std", [1.0] * channels),
            dtype=np.float32,
        )
        image_array = (image_array - mean) / std
        if self.input_layout == "NCHW":
            image_array = np.transpose(image_array, (2, 0, 1))
        return np.expand_dims(image_array, axis=0).astype(np.float32, copy=False)

    def _probabilities(self, raw_output: Any) -> np.ndarray:
        values = np.asarray(raw_output, dtype=np.float32).squeeze()
        if values.ndim == 0:
            values = values.reshape(1)
        if values.ndim != 1:
            raise RuntimeError(
                f"Unsupported ONNX output shape after removing batch axes: {values.shape}"
            )

        if len(values) == 1 and len(self.class_names) == 2:
            positive = (
                _sigmoid(float(values[0]))
                if self.output_type == "logits"
                else float(values[0])
            )
            probabilities = np.asarray([1.0 - positive, positive], dtype=np.float32)
        elif len(values) == len(self.class_names):
            probabilities = _softmax(values) if self.output_type == "logits" else values
        else:
            raise RuntimeError(
                "ONNX output size does not match class_names: "
                f"{len(values)} values for {len(self.class_names)} classes"
            )

        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError("ONNX output contains non-finite scores")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise RuntimeError("ONNX probabilities must be inside [0, 1]")
        total = float(probabilities.sum())
        if total <= 0:
            raise RuntimeError("ONNX probabilities sum to zero")
        return probabilities / total

    def predict(self, image: Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        self._load()
        tensor = self.preprocess(image)
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model exposes no input")
        input_name = self.config.get("input_name") or inputs[0].name
        output_name = self.config.get("output_name")
        requested_outputs = [output_name] if output_name else None
        raw_outputs = self._session.run(requested_outputs, {input_name: tensor})
        if not raw_outputs:
            raise RuntimeError("ONNX model returned no output")

        probabilities = self._probabilities(raw_outputs[0])
        ranking = np.argsort(probabilities)[::-1]
        top_index = int(ranking[0])
        confidence = float(probabilities[top_index])
        top2_margin = (
            confidence - float(probabilities[int(ranking[1])])
            if len(ranking) > 1
            else confidence
        )
        predicted_class = self.class_names[top_index]
        abstention_reasons: list[str] = []
        if predicted_class != "uncertain" and confidence < self.confidence_threshold:
            abstention_reasons.append("confidence_below_threshold")
        if predicted_class != "uncertain" and top2_margin < self.min_top2_margin:
            abstention_reasons.append("top2_margin_below_threshold")
        if abstention_reasons:
            predicted_class = "uncertain"

        class_scores = {
            class_name: round(float(probabilities[index]), 6)
            for index, class_name in enumerate(self.class_names)
        }
        return {
            "image_quality": "good",
            "image_quality_assessed": False,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 6),
            "class_scores": class_scores,
            "visual_evidence": [],
            "justification": (
                "The ONNX classifier score was mapped to the project classes. "
                "This classifier does not provide a radiological explanation."
            ),
            "limitations": [
                "model output score is not a calibrated clinical probability",
                "image quality is not independently assessed",
                "no clinical context or visual localization",
            ],
            "warning": WARNING,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "confidence_type": "onnx_output_score_uncalibrated",
            "input_size": [int(value) for value in self.input_size],
            "abstention_reasons": abstention_reasons,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
