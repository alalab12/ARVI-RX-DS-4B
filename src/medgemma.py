from __future__ import annotations

import os
from typing import Any

from PIL import Image


DEFAULT_MODEL_ID = "google/medgemma-4b-it"


class MedGemmaBackend:
    """Lazy Hugging Face backend so local toy tests never load the VLM."""

    def __init__(self, model_id: str | None = None, max_new_tokens: int | None = None):
        self.model_id = model_id or os.getenv("MEDGEMMA_MODEL_ID", DEFAULT_MODEL_ID)
        self.max_new_tokens = max_new_tokens or int(os.getenv("MEDGEMMA_MAX_NEW_TOKENS", "512"))
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._dtype: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "MedGemma requires torch, transformers and accelerate. "
                "Install the project requirements in the GPU environment."
            ) from error

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                device_map="auto",
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"Unable to load {self.model_id}. Check model access, HF_TOKEN and GPU memory."
            ) from error
        self._model.eval()
        self._torch = torch
        self._dtype = dtype

    def generate(self, image: Image.Image, prompt: str) -> str:
        """Generate one deterministic text response from an image and prompt."""

        self._load()
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an educational radiology assistant. "
                            "Do not provide a definitive diagnosis."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device, dtype=self._dtype)
        prompt_length = inputs["input_ids"].shape[-1]

        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated_tokens = generated[0][prompt_length:]
        return self._processor.decode(generated_tokens, skip_special_tokens=True)
