from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.guardrails import WARNING_TEXT
from src.inference import predict
from src.onnx_backend import OnnxClassifierBackend, OnnxConfigurationError


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"


def test_u_ones_model_contract_matches_training_notebook() -> None:
    config = json.loads((ROOT / "config" / "onnx_u_ones.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "config" / "onnx_models.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "models" / "u_ones_manifest.json").read_text(encoding="utf-8"))
    notebook = json.loads(
        (ROOT / "notebooks" / "07_train_resnet18_u_ones.ipynb").read_text(encoding="utf-8")
    )
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert registry["default_model"] == "u_ones"
    assert registry["models"]["u_ones"]["config_path"] == "config/onnx_u_ones.json"
    assert config["strategy"] == "u_ones"
    assert config["architecture"] == "torchvision.resnet18"
    assert config["input_name"] == "input"
    assert config["output_name"] == "logits"
    assert config["input_size"] == [224, 224]
    assert config["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert config["normalization"]["std"] == [0.229, 0.224, 0.225]
    assert config["class_names"] == ["normal", "suspected_opacity"]
    assert config["companion_files"] == ["arvi_cxr_classifier_u_ones.onnx.data"]
    assert config["confidence_threshold"] == 0.0
    assert config["min_top2_margin"] == 0.0
    assert manifest["external_data_received"] is True
    assert manifest["external_data_size_bytes"] == 44695552
    assert 'STRATEGY    = "u_ones"' in notebook_source
    assert 'input_names=["input"]' in notebook_source
    assert 'output_names=["logits"]' in notebook_source


def test_u_zeros_model_contract_matches_training_notebook() -> None:
    config = json.loads((ROOT / "config" / "onnx_u_zeros.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "config" / "onnx_models.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "models" / "u_zeros_manifest.json").read_text(encoding="utf-8"))
    notebook = json.loads(
        (ROOT / "notebooks" / "08_train_resnet18_u_zeros.ipynb").read_text(encoding="utf-8")
    )
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert registry["models"]["u_zeros"]["config_path"] == "config/onnx_u_zeros.json"
    assert config["strategy"] == "u_zeros"
    assert config["architecture"] == "torchvision.resnet18"
    assert config["input_name"] == "input"
    assert config["output_name"] == "logits"
    assert config["input_size"] == [224, 224]
    assert config["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert config["normalization"]["std"] == [0.229, 0.224, 0.225]
    assert config["class_names"] == ["normal", "suspected_opacity"]
    assert config["companion_files"] == ["arvi_cxr_classifier_u_zeros.onnx.data"]
    assert config["confidence_threshold"] == 0.0
    assert config["min_top2_margin"] == 0.0
    assert manifest["external_data_received"] is True
    assert manifest["external_data_size_bytes"] == 44695552
    assert manifest["runtime_validation"]["status"] == "passed"
    assert 'STRATEGY    = "u_zeros"' in notebook_source
    assert 'return "normal"  # 0.0 et -1.0' in notebook_source
    assert 'input_names=["input"]' in notebook_source
    assert 'output_names=["logits"]' in notebook_source


class FakeInput:
    name = "pixel_values"


class FakeSession:
    def __init__(self, output: np.ndarray):
        self.output = output
        self.received_tensor: np.ndarray | None = None

    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, inputs):
        assert output_names is None
        self.received_tensor = inputs["pixel_values"]
        return [self.output]


def write_config(tmp_path: Path, **overrides) -> Path:
    config = {
        "version": "onnx_test_v1",
        "model_name": "fake-onnx-classifier",
        "model_path": "models/not-needed-in-injected-test.onnx",
        "input_size": [32, 24],
        "input_layout": "NCHW",
        "normalization": {
            "scale": "zero_one",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "class_names": ["normal", "suspected_opacity"],
        "output_type": "logits",
        "confidence_threshold": 0.65,
        "min_top2_margin": 0.10,
    }
    config.update(overrides)
    config_path = tmp_path / "onnx_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_onnx_backend_maps_logits_to_project_contract(tmp_path: Path) -> None:
    session = FakeSession(np.asarray([[0.1, 2.0]], dtype=np.float32))
    backend = OnnxClassifierBackend(
        config_path=write_config(tmp_path),
        session=session,
    )

    result = predict(SAMPLE_IMAGE, backend="onnx", model_backend=backend)

    assert session.received_tensor is not None
    assert session.received_tensor.shape == (1, 3, 32, 24)
    assert session.received_tensor.dtype == np.float32
    assert result["predicted_class"] == "suspected_opacity"
    assert result["class_scores"]["suspected_opacity"] > 0.85
    assert result["model_name"] == "fake-onnx-classifier"
    assert result["warning"] == WARNING_TEXT
    assert result["guardrail_actions"] == []


def test_onnx_backend_abstains_when_scores_are_too_close(tmp_path: Path) -> None:
    backend = OnnxClassifierBackend(
        config_path=write_config(
            tmp_path,
            confidence_threshold=0.0,
            min_top2_margin=0.20,
        ),
        session=FakeSession(np.asarray([[0.05, 0.0]], dtype=np.float32)),
    )

    result = backend.predict(Image.open(SAMPLE_IMAGE))

    assert result["predicted_class"] == "uncertain"
    assert result["abstention_reasons"] == ["top2_margin_below_threshold"]


def test_onnx_backend_supports_single_binary_probability(tmp_path: Path) -> None:
    backend = OnnxClassifierBackend(
        config_path=write_config(
            tmp_path,
            output_type="probabilities",
            confidence_threshold=0.0,
            min_top2_margin=0.0,
        ),
        session=FakeSession(np.asarray([[0.8]], dtype=np.float32)),
    )

    result = predict(SAMPLE_IMAGE, backend="onnx", model_backend=backend)

    assert result["predicted_class"] == "suspected_opacity"
    assert result["class_scores"]["normal"] == pytest.approx(0.2, abs=1e-6)
    assert result["class_scores"]["suspected_opacity"] == pytest.approx(0.8, abs=1e-6)


def test_onnx_backend_reports_missing_model_only_when_used(tmp_path: Path) -> None:
    backend = OnnxClassifierBackend(config_path=write_config(tmp_path))

    with pytest.raises(RuntimeError, match="ONNX model files not found"):
        predict(SAMPLE_IMAGE, backend="onnx", model_backend=backend)


def test_onnx_backend_reports_missing_external_data_file(tmp_path: Path) -> None:
    graph_path = tmp_path / "model.onnx"
    graph_path.write_bytes(b"graph-placeholder")
    config_path = write_config(
        tmp_path,
        model_path=str(graph_path),
        companion_files=["model.onnx.data"],
    )
    backend = OnnxClassifierBackend(config_path=config_path)

    assert backend.missing_model_files() == [tmp_path / "model.onnx.data"]
    with pytest.raises(RuntimeError, match="model.onnx.data"):
        backend.predict(Image.open(SAMPLE_IMAGE))


def test_onnx_configuration_rejects_unknown_classes(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, class_names=["healthy", "disease"])

    with pytest.raises(OnnxConfigurationError, match="project classes"):
        OnnxClassifierBackend(
            config_path=config_path,
            session=FakeSession(np.zeros((1, 2))),
        )


def test_streamlit_ui_exposes_onnx_without_hard_coding_a_model() -> None:
    source = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    backend_source = (ROOT / "src" / "onnx_backend.py").read_text(encoding="utf-8")
    theme = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")

    assert '"onnx"' in source
    assert "ONNX_MODEL_PATH" in backend_source
    assert "ONNX_REGISTRY_PATH" in source
    assert "Variante ONNX" in source
    assert "Lancer l'analyse" in source
    assert "Trace technique JSON" in source
    assert "Prototype pédagogique" in source
    assert '[data-testid="stMetricValue"]' in source
    assert "color: #172033 !important" in source
    assert '.st-key-analysis_workspace > [data-testid="stHorizontalBlock"]' in source
    assert 'base = "light"' in theme
    assert 'textColor = "#172033"' in theme
    assert '[theme.sidebar]' in theme
    assert 'backgroundColor = "#0b1726"' in theme
