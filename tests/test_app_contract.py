from __future__ import annotations

import json
from pathlib import Path

from api.main import health
from src.guardrails import WARNING_TEXT
from src.inference import predict


ROOT = Path(__file__).resolve().parents[1]


def test_toy_backend_keeps_the_application_testable_without_model_loading() -> None:
    result = predict(
        ROOT / "data" / "sample_images" / "CXR_SYN_001_normal.png",
        backend="toy",
    )

    assert result["predicted_class"] == "normal"
    assert result["warning"] == WARNING_TEXT


def test_api_health_contract() -> None:
    assert health() == {
        "status": "ok",
        "scope": "educational prototype, not diagnosis",
    }


def test_registry_contains_only_the_two_integrated_variants() -> None:
    registry = json.loads(
        (ROOT / "config" / "onnx_models.json").read_text(encoding="utf-8")
    )

    assert registry["default_model"] == "u_ones"
    assert set(registry["models"]) == {"u_ones", "u_zeros"}


def test_legacy_professor_components_are_absent() -> None:
    removed_paths = [
        "README.pdf",
        "app/hf_space_app.py",
        "eval",
        "finetuning",
        "prompts",
        "sql",
        "notebooks/01_baseline_vlm.ipynb",
        "notebooks/02_prompt_comparison.ipynb",
        "notebooks/03_optional_finetuning_lora.ipynb",
        "notebooks/04_medsiglip_zero_shot.ipynb",
        "notebooks/05_medgemma_baseline_final.ipynb",
        "notebooks/06_hybrid_medsiglip_medgemma_dev.ipynb",
        "src/medgemma.py",
        "src/medsiglip.py",
        "src/explanation.py",
        "tests/test_repository_smoke.py",
    ]

    assert [path for path in removed_paths if (ROOT / path).exists()] == []


def test_runtime_dependencies_are_onnx_only() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    for removed_dependency in (
        "gradio",
        "transformers",
        "accelerate",
        "sentencepiece",
        "torch\n",
        "pydicom",
        "opencv-python",
        "scikit-learn",
        "pandas",
    ):
        assert removed_dependency not in requirements
