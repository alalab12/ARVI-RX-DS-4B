from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PredictionPayload(BaseModel):
    """Structured content that MedGemma must return."""

    model_config = ConfigDict(extra="ignore")

    image_quality: Literal["good", "limited", "poor"]
    predicted_class: Literal["normal", "suspected_opacity", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    visual_evidence: list[str]
    justification: str
    limitations: list[str]
    warning: str


class PredictionParseError(ValueError):
    """Raised when a model response cannot satisfy the output contract."""


def parse_prediction_json(raw_text: str) -> dict:
    """Extract the first JSON object and validate the prediction contract."""

    decoder = json.JSONDecoder()
    for position, character in enumerate(raw_text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw_text[position:])
            if isinstance(value, dict) and "visual_evidence" not in value and "evidence" in value:
                value["visual_evidence"] = value["evidence"]
            return PredictionPayload.model_validate(value).model_dump()
        except (json.JSONDecodeError, ValidationError, TypeError):
            continue
    raise PredictionParseError("MedGemma did not return a valid prediction JSON object")
