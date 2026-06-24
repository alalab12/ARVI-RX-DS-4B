from __future__ import annotations

import compileall
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from fastapi.testclient import TestClient

from api.main import app
from api.main import health
from src.guardrails import WARNING_TEXT, apply_safety_guardrails, validate_prediction
from src.inference import predict, toy_predict
from src.metrics import summarize_metrics
from src.prompting import load_prompt
from src.schemas import PredictionParseError, parse_prediction_json


ROOT = Path(__file__).resolve().parents[1]


def test_repository_student_contract_is_present() -> None:
    required_paths = [
        "README.md",
        "requirements.txt",
        "requirements-test.txt",
        ".github/workflows/ci.yml",
        "docs/appel_offre.md",
        "docs/architecture.md",
        "docs/ethique_et_limites.md",
        "docs/evaluation_protocol.md",
        "data/synthetic_cases.csv",
        "src/inference.py",
        "src/guardrails.py",
        "api/main.py",
        "eval/run_evaluation.py",
        "prompts/json_schema.md",
    ]
    forbidden_paths = [
        ".rollback_appel_offre_cleanup_20260516_205745",
        "VALIDATION_REPORT.md",
        "create_remote_repo.sh",
        "docs/expert_review_integration.md",
        "docs/github_push_instructions.md",
        "eval/outputs",
        "medical_ai_evidence.sqlite",
        "assets/assistant_radiologue_v3_notes_professeur_fr.pptx",
        "assets/notes_orales_assistant_radiologue_v3_style_professeur_fr.md",
    ]

    missing = [path for path in required_paths if not (ROOT / path).exists()]
    forbidden = [path for path in forbidden_paths if (ROOT / path).exists()]

    assert missing == []
    assert forbidden == []


def test_synthetic_dataset_contract_is_valid() -> None:
    path = ROOT / "data" / "synthetic_cases.csv"
    required_columns = {"case_id", "image_path", "source", "label", "split", "quality", "notes"}
    allowed_labels = {"normal", "suspected_opacity", "uncertain"}

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) >= 20
    assert required_columns <= set(rows[0])
    assert {row["label"] for row in rows} <= allowed_labels
    for row in rows:
        assert row["source"] == "synthetic_toy"
        assert (ROOT / row["image_path"]).exists()


def test_prediction_schema_warning_and_guardrails() -> None:
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"
    pred = apply_safety_guardrails(toy_predict(image_path, mode="improved"))
    valid, errors = validate_prediction(pred)

    assert valid, errors
    assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}
    assert pred["warning"] == WARNING_TEXT
    assert "not a validated medical model" in pred["limitations"]


def test_python_source_tree_compiles() -> None:
    for folder in ("src", "api", "app", "eval", "finetuning", "tests"):
        assert compileall.compile_dir(ROOT / folder, quiet=1)


def test_invalid_model_output_falls_back_to_uncertain() -> None:
    pred = apply_safety_guardrails({"predicted_class": "diagnosis", "confidence": 0.99})

    assert pred["predicted_class"] == "uncertain"
    assert pred["confidence"] <= 0.5
    assert pred["warning"] == WARNING_TEXT
    assert pred["guardrail_errors"]


def test_quality_guardrails_prevent_unsafe_normal_output() -> None:
    poor = apply_safety_guardrails({
        "image_quality": "poor",
        "predicted_class": "suspected_opacity",
        "confidence": 0.95,
        "visual_evidence": ["Possible opacity"],
        "justification": "The image is poorly exposed.",
        "limitations": ["Poor exposure"],
        "warning": "Educational prototype only",
    })
    limited_normal = apply_safety_guardrails({
        "image_quality": "limited",
        "predicted_class": "normal",
        "confidence": 0.90,
        "visual_evidence": ["No focal opacity identified"],
        "justification": "The projection is limited.",
        "limitations": ["Limited projection"],
        "warning": "Educational prototype only",
    })

    assert poor["predicted_class"] == "uncertain"
    assert poor["confidence"] <= 0.5
    assert poor["raw_predicted_class"] == "suspected_opacity"
    assert poor["raw_confidence"] == 0.95
    assert "poor_quality_to_uncertain" in poor["guardrail_actions"]
    assert limited_normal["predicted_class"] == "uncertain"
    assert limited_normal["confidence"] <= 0.5
    assert limited_normal["raw_predicted_class"] == "normal"
    assert "limited_normal_to_uncertain" in limited_normal["guardrail_actions"]


def test_limited_high_confidence_opacity_is_preserved() -> None:
    pred = apply_safety_guardrails({
        "image_quality": "limited",
        "predicted_class": "suspected_opacity",
        "confidence": 0.80,
        "visual_evidence": ["Focal opacity-like density"],
        "justification": "Visible evidence remains despite limited projection.",
        "limitations": ["Limited projection"],
        "warning": "Educational prototype only",
    })

    assert pred["predicted_class"] == "suspected_opacity"
    assert pred["raw_predicted_class"] == "suspected_opacity"
    assert pred["guardrail_actions"] == []


def test_guardrail_does_not_report_a_no_op_conversion() -> None:
    pred = apply_safety_guardrails({
        "image_quality": "limited",
        "predicted_class": "uncertain",
        "confidence": 0.20,
        "visual_evidence": ["Ambiguous basal density"],
        "justification": "Visible evidence is inconclusive.",
        "limitations": ["Limited projection"],
        "warning": "Educational prototype only",
    })

    assert pred["predicted_class"] == "uncertain"
    assert pred["raw_predicted_class"] == "uncertain"
    assert pred["guardrail_actions"] == []


def test_prompt_history_and_improved_v3_priority() -> None:
    prompt_v1, version_v1 = load_prompt("improved_v1")
    prompt_v2, version_v2 = load_prompt("improved_v2")
    prompt_v3, version_v3 = load_prompt("improved_v3")
    current_prompt, current_version = load_prompt("improved")

    assert [version_v1, version_v2, version_v3] == [
        "improved_v1",
        "improved_v2",
        "improved_v3",
    ]
    assert len({prompt_v1, prompt_v2, prompt_v3}) == 3
    assert current_version == "improved_v3"
    assert current_prompt == prompt_v3
    assert "Positive opacity-related evidence has priority" in prompt_v3
    assert "AP, portable acquisition or visible devices alone" in prompt_v3


def test_medgemma_json_parser_accepts_fenced_output() -> None:
    raw_output = """```json
    {
      "image_quality": "good",
      "predicted_class": "normal",
      "confidence": 0.71,
      "visual_evidence": ["No focal opacity described"],
      "justification": "The visible lung fields do not show a focal opacity.",
      "limitations": ["No clinical context"],
      "warning": "Educational prototype only"
    }
    ```"""

    pred = parse_prediction_json(raw_output)

    assert pred["predicted_class"] == "normal"
    assert pred["confidence"] == 0.71


def test_medgemma_backend_is_testable_without_loading_model() -> None:
    class FakeBackend:
        model_id = "fake-medgemma"

        def generate(self, image, prompt):
            assert image.mode == "RGB"
            assert "Return only valid JSON" in prompt
            return """{
              "image_quality": "limited",
              "predicted_class": "uncertain",
              "confidence": 0.42,
              "visual_evidence": ["Projection is limited"],
              "justification": "The image is not conclusive.",
              "limitations": ["Limited projection"],
              "warning": "Educational prototype only"
            }"""

    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_006_uncertain.png"
    pred = predict(
        image_path,
        mode="baseline",
        backend="medgemma",
        model_backend=FakeBackend(),
    )

    assert pred["predicted_class"] == "uncertain"
    assert pred["model_name"] == "fake-medgemma"
    assert pred["prompt_version"] == "baseline_v1"
    assert pred["confidence_type"] == "model_reported_uncalibrated"
    assert pred["raw_predicted_class"] == "uncertain"
    assert pred["guardrail_actions"] == []


def test_invalid_medgemma_text_is_rejected() -> None:
    try:
        parse_prediction_json("This is not JSON")
    except PredictionParseError:
        pass
    else:
        raise AssertionError("Invalid model output must not pass validation")


def test_metrics_and_api_health_contract() -> None:
    rows = [
        {"label": "normal", "predicted_class": "normal", "json_valid": True, "warning": WARNING_TEXT},
        {"label": "suspected_opacity", "predicted_class": "uncertain", "json_valid": True, "warning": WARNING_TEXT},
    ]
    metrics = summarize_metrics(rows)

    assert health()["status"] == "ok"
    assert health()["scope"] == "educational prototype, not diagnosis"
    assert metrics["n"] == 2
    assert metrics["json_valid_rate"] == 1.0
    assert metrics["warning_rate"] == 1.0


def test_api_predict_preserves_uploaded_case_signal() -> None:
    client = TestClient(app)
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"

    with image_path.open("rb") as file:
        response = client.post(
            "/predict",
            files={"file": (image_path.name, file, "image/png")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["predicted_class"] == "suspected_opacity"
    assert payload["warning"] == WARNING_TEXT
    shutil.rmtree(ROOT / "tmp_uploads", ignore_errors=True)


def test_evaluation_command_runs_and_preserves_warning_contract(tmp_path: Path) -> None:
    db_path = tmp_path / "medical_ai_evidence.sqlite"
    out_dir = tmp_path / "outputs"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "eval/run_evaluation.py",
            "--mode",
            "toy",
            "--out-dir",
            str(out_dir),
            "--db-path",
            str(db_path),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert {row["mode"] for row in summary} == {"baseline", "improved"}
    assert all(row["json_valid_rate"] == 1.0 for row in summary)
    assert all(row["warning_rate"] == 1.0 for row in summary)
    assert (out_dir / "before_after_summary.csv").exists()
    assert db_path.exists()
