from __future__ import annotations

import compileall
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from fastapi.testclient import TestClient

from api.main import app
from api.main import health
from src.explanation import ExplanationService, agreement_status
from src.guardrails import WARNING_TEXT, apply_safety_guardrails, validate_prediction
from src.hybrid import agreement_or_abstain, routing_bounds
from src.inference import predict, toy_predict
from src.medsiglip import classify_opacity_score
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
        "docs/journal_experimental.md",
        "docs/deployment_huggingface_space.md",
        "config/medgemma_baseline_v1.json",
        "config/medgemma_explanation_v1.json",
        "config/medsiglip_zero_shot_v1.json",
        "config/hybrid_search_v1.json",
        "notebooks/04_medsiglip_zero_shot.ipynb",
        "notebooks/05_medgemma_baseline_final.ipynb",
        "notebooks/06_hybrid_medsiglip_medgemma_dev.ipynb",
        "data/synthetic_cases.csv",
        "src/inference.py",
        "src/medsiglip.py",
        "src/explanation.py",
        "src/guardrails.py",
        "app/hf_space_app.py",
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


def test_medsiglip_notebook_contract_is_valid() -> None:
    notebook_path = ROOT / "notebooks" / "04_medsiglip_zero_shot.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    config = json.loads(
        (ROOT / "config" / "medsiglip_zero_shot_v1.json").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert notebook["nbformat"] == 4
    assert "google/medsiglip-448" in source
    assert "SELECTION_CSV" in source
    assert "split `final`" in source
    assert "Metadonnees d'etude introuvables" in source
    assert "relative_similarity_uncalibrated" in source
    assert "RUN_FINAL = False" in source
    assert "medsiglip_final_predictions.csv" in source
    assert config["version"] == "medsiglip_zero_shot_v1"
    assert config["model_id"] == "google/medsiglip-448"
    assert config["low_threshold"] < config["high_threshold"]
    assert config["dev_metrics"]["opacity_sensitivity"] >= 0.85
    assert config["dev_metrics"]["normal_specificity"] >= 0.70

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        python_source = "".join(
            line for line in cell.get("source", [])
            if not line.lstrip().startswith("%")
        )
        compile(python_source, f"medsiglip-notebook-cell-{index}", "exec")


def test_medgemma_baseline_final_notebook_is_frozen() -> None:
    notebook_path = ROOT / "notebooks" / "05_medgemma_baseline_final.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    config = json.loads(
        (ROOT / "config" / "medgemma_baseline_v1.json").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    prompt_bytes = (ROOT / "prompts" / "baseline_prompt.txt").read_bytes()
    prompt_bytes = prompt_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

    assert notebook["nbformat"] == 4
    assert config["model_id"] == "google/medgemma-4b-it"
    assert config["prompt_version"] == "baseline_v1"
    assert (
        hashlib.sha256(prompt_bytes).hexdigest()
        == config["prompt_sha256_canonical_lf"]
    )
    assert "RUN_BASELINE_FINAL = False" in source
    assert "medgemma_baseline_final_predictions.csv" in source

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        python_source = "".join(
            line for line in cell.get("source", [])
            if not line.lstrip().startswith("%")
        )
        compile(python_source, f"medgemma-baseline-notebook-cell-{index}", "exec")


def test_hybrid_dev_notebook_never_loads_final() -> None:
    notebook_path = ROOT / "notebooks" / "06_hybrid_medsiglip_medgemma_dev.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    config = json.loads(
        (ROOT / "config" / "hybrid_search_v1.json").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert notebook["nbformat"] == 4
    assert config["development_split"] == "dev"
    assert config["fusion_rule"] == "agreement_or_abstain"
    assert config["candidate_margins"] == [0.0, 0.025, 0.05, 0.075, 0.1]
    assert "medsiglip_dev_predictions.csv" in source
    assert "medgemma_routed_dev_predictions.csv" in source
    assert "RUN_MEDGEMMA_SMOKE = False" in source
    assert "RUN_MEDGEMMA_DEV = False" in source
    assert 'eq("dev").all()' in source
    assert 'eq("final")' not in source
    assert "final_results" not in source
    assert "no_hybrid_candidate_met_constraints" in source
    assert "conservation de MedSigLIP seul" in source

    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        python_source = "".join(
            line for line in cell.get("source", [])
            if not line.lstrip().startswith("%")
        )
        compile(python_source, f"hybrid-dev-notebook-cell-{index}", "exec")


def test_hybrid_agreement_or_abstention_contract() -> None:
    assert agreement_or_abstain("normal", "normal") == "normal"
    assert agreement_or_abstain("suspected_opacity", "suspected_opacity") == "suspected_opacity"
    assert agreement_or_abstain("normal", "suspected_opacity") == "uncertain"
    assert agreement_or_abstain("suspected_opacity", "normal") == "uncertain"
    assert agreement_or_abstain("uncertain", "normal") == "normal"
    assert agreement_or_abstain("uncertain", "suspected_opacity") == "suspected_opacity"
    assert agreement_or_abstain("normal", None) == "normal"
    assert routing_bounds(0.05, 0.95, 0.10) == (0.0, 1.0)


def test_medsiglip_frozen_score_contract_without_loading_model() -> None:
    class FakeBackend:
        def predict(self, image):
            assert image.mode == "RGB"
            return {
                "predicted_class": "suspected_opacity",
                "score_opacity": 0.41,
            }

    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"
    result = predict(
        image_path,
        backend="medsiglip",
        model_backend=FakeBackend(),
    )

    assert classify_opacity_score(0.20, 0.30, 0.40) == "normal"
    assert classify_opacity_score(0.35, 0.30, 0.40) == "uncertain"
    assert classify_opacity_score(0.45, 0.30, 0.40) == "suspected_opacity"
    assert result["predicted_class"] == "suspected_opacity"
    assert result["score_opacity"] == 0.41


def test_medgemma_explanation_is_cached_and_never_changes_primary(tmp_path: Path) -> None:
    class FakeBackend:
        model_id = "fake-medgemma"

        def generate(self, image, prompt):
            assert image.mode == "RGB"
            assert "independently" in prompt
            return """{
              "image_quality": "good",
              "predicted_class": "suspected_opacity",
              "confidence": 0.60,
              "visual_evidence": ["Focal opacity-like density"],
              "justification": "A focal increased density is visible.",
              "limitations": ["No clinical context"],
              "warning": "Educational prototype only"
            }"""

    service = ExplanationService(
        cache_dir=tmp_path,
        model_backend=FakeBackend(),
    )
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"

    first = service.explain(image_path, primary_class="normal")
    second = service.explain(image_path, primary_class="normal")

    assert first["primary_class"] == "normal"
    assert first["primary_class_unchanged"] is True
    assert first["agreement_status"] == "disagreement"
    assert first["cached"] is False
    assert second["cached"] is True
    assert agreement_status("uncertain", "normal") == "partial_disagreement"


def test_explanation_prompt_and_space_contract_are_frozen() -> None:
    config = json.loads(
        (ROOT / "config" / "medgemma_explanation_v1.json").read_text(encoding="utf-8")
    )
    prompt_bytes = (ROOT / "prompts" / "explanation_prompt.txt").read_bytes()
    prompt_bytes = prompt_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    space_source = (ROOT / "app" / "hf_space_app.py").read_text(encoding="utf-8")

    assert hashlib.sha256(prompt_bytes).hexdigest() == config["prompt_sha256_canonical_lf"]
    assert config["decision_policy"].startswith("MedSigLIP remains")
    assert "sdk: gradio" in readme
    assert "suggested_hardware: t4-small" in readme
    assert 'api_name="classify"' in space_source
    assert 'api_name="explain"' in space_source
    assert space_source.count('concurrency_id="gpu_models"') == 2
    assert 'decision_state["primary_class"]' in space_source
    assert "Cette analyse ne modifie pas la decision MedSigLIP" in space_source


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


def test_prompt_history_and_improved_v5_presence_policy() -> None:
    prompt_v1, version_v1 = load_prompt("improved_v1")
    prompt_v2, version_v2 = load_prompt("improved_v2")
    prompt_v3, version_v3 = load_prompt("improved_v3")
    prompt_v4, version_v4 = load_prompt("improved_v4")
    prompt_v5, version_v5 = load_prompt("improved_v5")
    current_prompt, current_version = load_prompt("improved")

    assert [version_v1, version_v2, version_v3, version_v4, version_v5] == [
        "improved_v1",
        "improved_v2",
        "improved_v3",
        "improved_v4",
        "improved_v5",
    ]
    assert len({prompt_v1, prompt_v2, prompt_v3, prompt_v4, prompt_v5}) == 5
    assert current_version == "improved_v3"
    assert current_prompt == prompt_v3
    assert "Positive opacity-related evidence has priority" in prompt_v3
    assert "Vague basal haze on a limited image" in prompt_v4
    assert "Uncertainty about the cause is not uncertainty about presence" in prompt_v5
    assert 'PRESENT -> "suspected_opacity"' in prompt_v5
    assert "AP, portable acquisition or visible devices alone" in prompt_v3
    assert "AP, portable acquisition or visible devices alone" in prompt_v4
    assert "AP, portable acquisition or visible devices alone" in prompt_v5


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


def test_medgemma_json_parser_accepts_evidence_alias() -> None:
    raw_output = """{
      "image_quality": "limited",
      "predicted_class": "uncertain",
      "confidence": 0.40,
      "evidence": ["Assessment is limited by low volume and basal crowding."],
      "justification": "The visible finding is not clear enough to call opacity.",
      "limitations": ["Limited image quality"],
      "warning": "Educational prototype only"
    }"""

    pred = parse_prediction_json(raw_output)

    assert pred["predicted_class"] == "uncertain"
    assert pred["visual_evidence"] == ["Assessment is limited by low volume and basal crowding."]


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
