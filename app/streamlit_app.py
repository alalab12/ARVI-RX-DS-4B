from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import streamlit as st

from src.inference import predict
from src.onnx_backend import OnnxClassifierBackend, OnnxConfigurationError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX_REGISTRY_PATH = ROOT / "config" / "onnx_models.json"
WARNING = (
    "Prototype pédagogique. Non destiné au diagnostic. "
    "Validation par un professionnel qualifié requise."
)
BACKENDS = {
    "onnx": {
        "label": "Classifieur ONNX — équipe",
        "short_label": "ONNX",
        "description": "Modèles entraînés par l'équipe, comparables par stratégie de gestion des labels incertains.",
    },
    "toy": {
        "label": "Démonstration synthétique",
        "short_label": "Démo locale",
        "description": "Validation locale du parcours avec les images synthétiques.",
    },
}
CLASS_LABELS = {
    "normal": "Normal",
    "suspected_opacity": "Opacité suspectée",
    "uncertain": "Incertain",
}
QUALITY_LABELS = {
    "good": "Bonne",
    "limited": "Limitée",
    "poor": "Mauvaise",
}
CLASS_STYLES = {
    "normal": {
        "foreground": "#0f766e",
        "background": "#ccfbf1",
        "border": "#5eead4",
    },
    "suspected_opacity": {
        "foreground": "#b45309",
        "background": "#ffedd5",
        "border": "#fdba74",
    },
    "uncertain": {
        "foreground": "#475569",
        "background": "#e2e8f0",
        "border": "#cbd5e1",
    },
}


def _load_onnx_registry() -> tuple[dict[str, dict[str, str]], str, str | None]:
    registry_path = Path(os.getenv("ONNX_REGISTRY_PATH", DEFAULT_ONNX_REGISTRY_PATH))
    if not registry_path.is_absolute():
        registry_path = ROOT / registry_path
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, "", f"Registre ONNX indisponible : {error}"

    raw_models = registry.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        return {}, "", "Le registre ONNX ne contient aucune variante."

    models: dict[str, dict[str, str]] = {}
    for model_id, raw_entry in raw_models.items():
        if not isinstance(raw_entry, dict) or not raw_entry.get("config_path"):
            continue
        config_path = Path(str(raw_entry["config_path"]))
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        models[str(model_id)] = {
            "label": str(raw_entry.get("label", model_id)),
            "description": str(raw_entry.get("description", "")),
            "config_path": str(config_path),
        }
    if not models:
        return {}, "", "Aucune configuration ONNX exploitable dans le registre."

    default_model = str(registry.get("default_model", ""))
    if default_model not in models:
        default_model = next(iter(models))
    return models, default_model, None


def _onnx_status(config_path: str | Path) -> tuple[bool, str]:
    try:
        backend = OnnxClassifierBackend(config_path=config_path)
    except (OSError, json.JSONDecodeError, OnnxConfigurationError) as error:
        return False, f"Configuration ONNX indisponible : {error}"

    missing_files = backend.missing_model_files()
    if missing_files:
        missing_names = ", ".join(path.name for path in missing_files)
        return False, f"Fichier requis manquant : {missing_names}"
    return True, f"{backend.model_path.name} prêt"


def _run_prediction(
    uploaded_file: Any,
    backend: str,
    mode: str,
    backend_config_path: str | Path | None = None,
) -> dict[str, Any]:
    suffix = Path(uploaded_file.name or "image.png").suffix.lower() or ".png"
    stem = Path(uploaded_file.name or "image").stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem) or "image"
    with tempfile.TemporaryDirectory(prefix="arvi_upload_") as directory:
        temporary_path = Path(directory) / f"{safe_stem}{suffix}"
        temporary_path.write_bytes(uploaded_file.getvalue())
        return predict(
            temporary_path,
            mode=mode,
            backend=backend,
            backend_config_path=backend_config_path,
        )


def _score_label(result: dict[str, Any]) -> str:
    confidence_type = str(result.get("confidence_type", ""))
    if "uncalibrated" in confidence_type or "relative" in confidence_type:
        return "Score relatif"
    return "Confiance déclarée"


def _render_score_bars(class_scores: dict[str, float]) -> None:
    for class_name, raw_score in class_scores.items():
        score = max(0.0, min(1.0, float(raw_score)))
        label = CLASS_LABELS.get(class_name, class_name)
        st.markdown(
            f"""
            <div class="score-row">
              <div class="score-copy"><span>{label}</span><strong>{score:.1%}</strong></div>
              <div class="score-track"><div class="score-fill" style="width:{score * 100:.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_result(result: dict[str, Any]) -> None:
    predicted_class = str(result.get("predicted_class", "uncertain"))
    label = CLASS_LABELS.get(predicted_class, predicted_class)
    style = CLASS_STYLES.get(predicted_class, CLASS_STYLES["uncertain"])
    confidence = result.get("confidence")
    opacity_score = result.get("score_opacity")
    displayed_score = confidence if confidence is not None else opacity_score
    quality = (
        result.get("image_quality")
        if result.get("image_quality_assessed", True)
        else None
    )
    quality_label = QUALITY_LABELS.get(str(quality), "Non évaluée")
    latency = result.get("latency_ms")
    model_name = str(result.get("model_name", "Modèle non renseigné"))

    st.markdown(
        f"""
        <section class="result-hero" style="border-color:{style['border']}">
          <div>
            <span class="result-eyebrow">Classe proposée</span>
            <div class="result-title">{label}</div>
            <div class="result-model">{model_name}</div>
          </div>
          <span class="result-badge" style="color:{style['foreground']};background:{style['background']};border-color:{style['border']}">
            Analyse terminée
          </span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(3)
    with metric_columns[0]:
        st.metric(
            _score_label(result),
            f"{float(displayed_score):.1%}" if displayed_score is not None else "—",
            help="Ce score n'est pas une probabilité clinique calibrée.",
        )
    with metric_columns[1]:
        st.metric("Qualité de l'image", quality_label)
    with metric_columns[2]:
        st.metric("Temps d'analyse", f"{latency} ms" if latency is not None else "—")

    summary_tab, scores_tab, trace_tab = st.tabs(
        ["Synthèse", "Scores et limites", "Trace technique JSON"]
    )
    with summary_tab:
        evidence = result.get("visual_evidence") or []
        justification = result.get("justification")
        st.markdown("#### Lecture structurée")
        if evidence:
            for item in evidence:
                st.markdown(f"- {item}")
        else:
            st.caption("Ce moteur ne fournit pas d'élément visuel localisé.")
        if justification:
            st.markdown("#### Interprétation prudente")
            st.write(justification)
        st.markdown(
            f'<div class="inline-warning">{result.get("warning", WARNING)}</div>',
            unsafe_allow_html=True,
        )

    with scores_tab:
        class_scores = result.get("class_scores") or {}
        if class_scores:
            st.markdown("#### Répartition des scores")
            _render_score_bars(class_scores)
        else:
            st.caption("La répartition complète des scores n'est pas fournie par ce moteur.")

        limitations = result.get("limitations") or []
        st.markdown("#### Limites")
        if limitations:
            for limitation in limitations:
                st.markdown(f"- {limitation}")
        else:
            st.caption("Aucune limite spécifique n'a été retournée.")

        guardrail_actions = result.get("guardrail_actions") or []
        if guardrail_actions:
            st.markdown("#### Garde-fous appliqués")
            for action in guardrail_actions:
                st.code(action, language=None)

    with trace_tab:
        st.caption(
            "Sortie complète destinée à la reproductibilité, aux tests et à l'analyse d'erreurs."
        )
        st.json(result)


def _render_empty_result() -> None:
    st.markdown(
        """
        <div class="empty-result">
          <div class="empty-icon">RX</div>
          <div class="empty-title">Aucune analyse en cours</div>
          <p>Ajoutez une radiographie frontale puis lancez l'analyse. La classe, les scores et les limites apparaîtront ici.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="ARVI — Assistant radiologue virtuel",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(
    """
    <style>
    :root {
        --arvi-navy: #0b1726;
        --arvi-ink: #172033;
        --arvi-muted: #64748b;
        --arvi-line: #dbe4ec;
        --arvi-teal: #0f766e;
        --arvi-orange: #c2410c;
        --arvi-surface: rgba(255, 255, 255, .94);
    }
    html {color-scheme: light;}
    .stApp {
        background:
            radial-gradient(circle at 7% 3%, rgba(15, 118, 110, .10), transparent 28rem),
            radial-gradient(circle at 94% 8%, rgba(194, 65, 12, .08), transparent 24rem),
            #f4f7fa;
        color: var(--arvi-ink);
    }
    [data-testid="stHeader"] {background: transparent;}
    .block-container {max-width: 1240px; padding: 1.8rem 2rem 3rem;}
    [data-testid="stMainBlockContainer"] h1,
    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3,
    [data-testid="stMainBlockContainer"] h4 {color: var(--arvi-ink);}
    [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] > p,
    [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] p {color: var(--arvi-muted);}

    [data-testid="stSidebar"] {background: var(--arvi-navy); border-right: 1px solid #203247;}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {color: #e5eef7 !important;}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color: #b8c7d8 !important; line-height: 1.5;}
    [data-testid="stSidebar"] hr {border-color: #29405a;}
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #101c2b !important; border-color: #334b62 !important; color: #f8fafc !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg {fill: #dbeafe !important;}
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #0f1b2a; border-color: #29405a; border-radius: 12px;
    }
    .sidebar-brand {align-items: center; display: flex; gap: .8rem; margin: .4rem 0 1.5rem;}
    .sidebar-mark {
        align-items: center; background: linear-gradient(145deg, #14b8a6, #0f766e);
        border-radius: 12px; color: white; display: flex; font-size: .86rem;
        font-weight: 800; height: 42px; justify-content: center; letter-spacing: .04em; width: 42px;
    }
    .sidebar-name {color: white; font-size: 1.05rem; font-weight: 750; line-height: 1.1;}
    .sidebar-subtitle {color: #93a6ba; font-size: .72rem; margin-top: .24rem; text-transform: uppercase;}
    .model-status {
        align-items: center; background: rgba(15, 118, 110, .16); border: 1px solid rgba(45, 212, 191, .25);
        border-radius: 10px; color: #99f6e4; display: flex; font-size: .8rem; gap: .5rem;
        margin: .75rem 0; padding: .65rem .75rem;
    }
    .model-status.pending {background: rgba(194, 65, 12, .14); border-color: rgba(251, 146, 60, .28); color: #fed7aa;}
    .status-dot {background: currentColor; border-radius: 99px; height: 7px; width: 7px;}

    .topbar {
        align-items: center; background: linear-gradient(120deg, #0b1726 0%, #13283b 62%, #0f4c4a 125%);
        border: 1px solid rgba(255, 255, 255, .08); border-radius: 22px;
        box-shadow: 0 20px 55px rgba(15, 23, 42, .16); color: white;
        display: flex; justify-content: space-between; margin-bottom: 1rem; overflow: hidden;
        padding: 1.7rem 1.9rem; position: relative;
    }
    .topbar::after {
        background: rgba(45, 212, 191, .13); border-radius: 50%; content: "";
        height: 190px; position: absolute; right: -55px; top: -100px; width: 190px;
    }
    .brand-kicker {color: #5eead4; font-size: .72rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase;}
    .brand-title {font-size: 2rem; font-weight: 780; letter-spacing: -.035em; line-height: 1.05; margin: .25rem 0 .45rem;}
    .brand-copy {color: #c8d6e5; font-size: .95rem; margin: 0; max-width: 650px;}
    .header-chips {display: flex; flex-wrap: wrap; gap: .45rem; justify-content: flex-end; position: relative; z-index: 1;}
    .header-chip {background: rgba(255, 255, 255, .09); border: 1px solid rgba(255, 255, 255, .14); border-radius: 999px; color: #e7f1f8; font-size: .75rem; padding: .38rem .65rem;}

    .safety-ribbon {
        align-items: center; background: #fff7ed; border: 1px solid #fed7aa;
        border-left: 5px solid var(--arvi-orange); border-radius: 12px; color: #7c2d12;
        display: flex; font-size: .88rem; gap: .65rem; margin: .9rem 0 1rem; padding: .72rem .9rem;
    }
    .safety-label {font-size: .7rem; font-weight: 850; letter-spacing: .08em; text-transform: uppercase;}

    .workflow {display: grid; gap: .65rem; grid-template-columns: repeat(3, 1fr); margin: .9rem 0 1.4rem;}
    .workflow-step {
        align-items: center; background: var(--arvi-surface); border: 1px solid var(--arvi-line);
        border-radius: 13px; display: flex; gap: .7rem; padding: .72rem .8rem;
    }
    .step-number {
        align-items: center; background: #e6f7f5; border-radius: 9px; color: var(--arvi-teal);
        display: flex; font-size: .78rem; font-weight: 850; height: 30px; justify-content: center; width: 30px;
    }
    .step-label {color: var(--arvi-ink); font-size: .82rem; font-weight: 720;}
    .step-copy {color: var(--arvi-muted); font-size: .7rem; margin-top: .08rem;}

    .section-kicker {color: var(--arvi-teal); font-size: .7rem; font-weight: 850; letter-spacing: .11em; margin-bottom: .2rem; text-transform: uppercase;}
    [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--arvi-surface); border-color: var(--arvi-line) !important;
        border-radius: 18px !important; box-shadow: 0 10px 32px rgba(15, 23, 42, .055);
    }
    [data-testid="stFileUploader"] {
        background: #f8fafc; border: 1.5px dashed #9fb6c8; border-radius: 14px; padding: .45rem;
    }
    [data-testid="stFileUploader"] section {padding: 1.15rem .8rem;}
    [data-testid="stFileUploaderDropzone"] {background: #f8fafc !important; border-color: #b7c7d4 !important;}
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small {color: #526174 !important;}
    [data-testid="stFileUploaderDropzone"] button {
        background: white !important; border-color: #b7c7d4 !important; color: #243248 !important;
    }
    [data-testid="stFileUploaderDropzone"] button p {color: #243248 !important;}
    [data-testid="stFileUploaderDropzone"] > div > div {color: #334155 !important;}
    .file-meta {color: var(--arvi-muted); font-size: .76rem; margin: .4rem 0 .7rem;}
    .privacy-note {color: var(--arvi-muted); font-size: .74rem; line-height: 1.45; margin-top: .55rem;}

    .stButton > button[kind="primary"] {
        background: linear-gradient(110deg, #0f766e, #0d9488); border: 0; border-radius: 11px;
        box-shadow: 0 8px 20px rgba(15, 118, 110, .20); font-weight: 750; min-height: 2.8rem;
    }
    .stButton > button[kind="primary"]:hover {background: linear-gradient(110deg, #115e59, #0f766e); box-shadow: 0 10px 24px rgba(15, 118, 110, .27);}

    .empty-result {
        align-items: center; border: 1.5px dashed #c7d4df; border-radius: 16px; display: flex;
        flex-direction: column; justify-content: center; min-height: 355px; padding: 2rem; text-align: center;
    }
    .empty-icon {
        align-items: center; background: #e6f7f5; border: 1px solid #b8e7e1; border-radius: 16px;
        color: var(--arvi-teal); display: flex; font-size: .88rem; font-weight: 850;
        height: 54px; justify-content: center; margin-bottom: .9rem; width: 54px;
    }
    .empty-title {color: var(--arvi-ink); font-size: 1rem; font-weight: 760;}
    .empty-result p {color: var(--arvi-muted); font-size: .84rem; line-height: 1.55; max-width: 390px;}

    .result-hero {
        align-items: center; background: #fbfdff; border: 1px solid; border-left-width: 5px;
        border-radius: 14px; display: flex; justify-content: space-between; margin-bottom: .9rem; padding: 1rem 1.05rem;
    }
    .result-eyebrow {color: var(--arvi-muted); font-size: .68rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase;}
    .result-title {color: var(--arvi-ink); font-size: 1.45rem; font-weight: 800; letter-spacing: -.025em; margin: .15rem 0;}
    .result-model {color: var(--arvi-muted); font-size: .72rem;}
    .result-badge {border: 1px solid; border-radius: 999px; font-size: .72rem; font-weight: 800; padding: .4rem .65rem;}
    [data-testid="stMetric"] {
        background: #f8fafc; border: 1px solid #d8e2eb; border-radius: 12px;
        color: var(--arvi-ink) !important; min-height: 92px; padding: .7rem .8rem;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span {
        color: #526174 !important; font-size: .76rem !important; opacity: 1 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] span {
        color: #172033 !important; font-size: 1.35rem !important;
        font-weight: 780 !important; opacity: 1 !important;
    }
    [data-testid="stMetricDelta"] {color: #0f766e !important; opacity: 1 !important;}
    .score-row {margin: .8rem 0;}
    .score-copy {color: var(--arvi-ink); display: flex; font-size: .8rem; justify-content: space-between; margin-bottom: .32rem;}
    .score-track {background: #e7edf2; border-radius: 99px; height: 7px; overflow: hidden;}
    .score-fill {background: linear-gradient(90deg, #0f766e, #2dd4bf); border-radius: 99px; height: 100%;}
    .inline-warning {background: #fff7ed; border-radius: 9px; color: #9a3412; font-size: .76rem; margin-top: 1rem; padding: .65rem .75rem;}
    [data-baseweb="tab-list"] {gap: .4rem;}
    [data-baseweb="tab"] {color: #526174 !important; font-size: .8rem;}
    [data-baseweb="tab"][aria-selected="true"] {color: var(--arvi-teal) !important; font-weight: 750;}
    [data-testid="stAlert"] p {color: inherit !important;}
    .app-footer {color: #7b8da0; font-size: .72rem; margin-top: 1.6rem; text-align: center;}

    @media (max-width: 1100px) {
        .st-key-analysis_workspace > [data-testid="stHorizontalBlock"] {
            flex-direction: column !important; gap: 1rem !important;
        }
        .st-key-analysis_workspace > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 100% !important; min-width: 0 !important; width: 100% !important;
        }
    }

    @media (max-width: 900px) {
        .block-container {padding: 1.1rem .8rem 2rem;}
        .topbar {align-items: flex-start; flex-direction: column; gap: 1rem; padding: 1.35rem;}
        .brand-title {font-size: 1.65rem;}
        .header-chips {justify-content: flex-start;}
        .workflow {grid-template-columns: 1fr;}
        .result-hero {align-items: flex-start; flex-direction: column; gap: .8rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-mark">AR</div>
          <div><div class="sidebar-name">ARVI</div><div class="sidebar-subtitle">Console d'analyse</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Moteur")
    backend_options = list(BACKENDS)
    default_backend = os.getenv("MODEL_BACKEND", "onnx").lower()
    default_backend_index = (
        backend_options.index(default_backend) if default_backend in backend_options else 0
    )
    selected_backend = st.selectbox(
        "Moteur d'analyse",
        options=backend_options,
        format_func=lambda value: BACKENDS[value]["label"],
        index=default_backend_index,
        label_visibility="collapsed",
    )
    st.caption(BACKENDS[selected_backend]["description"])

    backend_ready = True
    selected_onnx_config_path: str | None = None
    analysis_key = selected_backend
    active_model_label = BACKENDS[selected_backend]["short_label"]
    if selected_backend == "onnx":
        onnx_models, default_onnx_model, registry_error = _load_onnx_registry()
        if registry_error:
            backend_ready = False
            onnx_message = registry_error
        else:
            st.markdown("### Variante ONNX")
            onnx_model_ids = list(onnx_models)
            default_onnx_index = onnx_model_ids.index(default_onnx_model)
            selected_onnx_model = st.selectbox(
                "Variante ONNX",
                options=onnx_model_ids,
                format_func=lambda value: onnx_models[value]["label"],
                index=default_onnx_index,
                label_visibility="collapsed",
            )
            selected_onnx_entry = onnx_models[selected_onnx_model]
            selected_onnx_config_path = selected_onnx_entry["config_path"]
            st.caption(selected_onnx_entry["description"])
            active_model_label = selected_onnx_entry["label"]
            analysis_key = f"onnx:{selected_onnx_model}"
            backend_ready, onnx_message = _onnx_status(selected_onnx_config_path)
        status_class = "" if backend_ready else " pending"
        st.markdown(
            f'<div class="model-status{status_class}"><span class="status-dot"></span>{onnx_message}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="model-status"><span class="status-dot"></span>Moteur sélectionné</div>',
            unsafe_allow_html=True,
        )

    selected_mode = "baseline"
    if selected_backend == "toy":
        st.markdown("### Stratégie")
        selected_mode = st.radio(
            "Mode",
            options=["baseline", "improved"],
            format_func=lambda value: "Baseline" if value == "baseline" else "Amélioré",
            horizontal=True,
            label_visibility="collapsed",
        )

    st.divider()
    with st.expander("Périmètre et formats"):
        st.caption("Radiographie thoracique frontale uniquement.")
        st.caption("Formats : PNG, JPG, JPEG et BMP.")
        st.caption("Trois sorties : normal, opacité suspectée ou incertain.")

st.markdown(
    """
    <header class="topbar">
      <div>
        <div class="brand-kicker">Assistant radiologue virtuel</div>
        <div class="brand-title">Une analyse structurée, une décision prudente.</div>
        <p class="brand-copy">Classification ONNX de radiographies thoraciques frontales avec sorties structurées et traçables.</p>
      </div>
      <div class="header-chips">
        <span class="header-chip">CXR frontale</span>
        <span class="header-chip">3 classes</span>
        <span class="header-chip">Sortie traçable</span>
      </div>
    </header>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="safety-ribbon">
      <span class="safety-label">Cadre pédagogique</span><span>{WARNING}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="workflow">
      <div class="workflow-step"><div class="step-number">01</div><div><div class="step-label">Importer</div><div class="step-copy">Radiographie frontale dé-identifiée</div></div></div>
      <div class="workflow-step"><div class="step-number">02</div><div><div class="step-label">Analyser</div><div class="step-copy">Moteur sélectionné et garde-fous</div></div></div>
      <div class="workflow-step"><div class="step-number">03</div><div><div class="step-label">Examiner</div><div class="step-copy">Résultat, limites et trace JSON</div></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

workspace = st.container(key="analysis_workspace")
left_column, right_column = workspace.columns([.92, 1.08], gap="large")
with left_column:
    with st.container(border=True):
        st.markdown('<div class="section-kicker">Étape 01 · Image</div>', unsafe_allow_html=True)
        st.subheader("Ajouter une radiographie")
        st.caption("Utilisez une vue thoracique frontale sans donnée identifiante.")
        uploaded = st.file_uploader(
            "Radiographie thoracique frontale",
            type=["png", "jpg", "jpeg", "bmp"],
            label_visibility="collapsed",
        )
        current_digest = None
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            current_digest = hashlib.sha256(image_bytes).hexdigest()
            st.image(image_bytes, caption=uploaded.name, use_container_width=True)
            st.markdown(
                f'<div class="file-meta">{uploaded.name} · {len(image_bytes) / 1024:.1f} Ko · empreinte {current_digest[:10]}…</div>',
                unsafe_allow_html=True,
            )
        analyze = st.button(
            "Lancer l'analyse",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None or not backend_ready,
        )
        if selected_backend == "onnx" and not backend_ready:
            st.warning(f"Le modèle ne peut pas encore être exécuté. {onnx_message}")
        elif uploaded is None:
            st.caption("Vous pouvez tester avec une image de `data/sample_images`.")
        st.markdown(
            '<div class="privacy-note">Confidentialité : n’importez aucune image contenant un nom, un identifiant patient ou une information clinique personnelle.</div>',
            unsafe_allow_html=True,
        )

with right_column:
    with st.container(border=True):
        st.markdown('<div class="section-kicker">Étape 02 · Résultat</div>', unsafe_allow_html=True)
        st.subheader("Compte rendu de l'analyse")
        st.caption(
            f"Moteur actif : {active_model_label} · sortie pédagogique non clinique"
        )

        if analyze and uploaded is not None:
            st.session_state.pop("last_prediction", None)
            st.session_state.pop("last_image_digest", None)
            st.session_state.pop("last_analysis_key", None)
            try:
                with st.spinner(f"Analyse avec {active_model_label}…"):
                    result = _run_prediction(
                        uploaded,
                        selected_backend,
                        selected_mode,
                        backend_config_path=selected_onnx_config_path,
                    )
                st.session_state["last_prediction"] = result
                st.session_state["last_image_digest"] = current_digest
                st.session_state["last_analysis_key"] = analysis_key
            except (ValueError, RuntimeError) as error:
                st.error(str(error))

        saved_result = st.session_state.get("last_prediction")
        saved_digest = st.session_state.get("last_image_digest")
        previous_analysis_key = st.session_state.get("last_analysis_key")
        if saved_result is not None and saved_digest == current_digest:
            if previous_analysis_key != analysis_key:
                st.info(
                    "Ce résultat provient d'un autre moteur. Relancez l'analyse pour appliquer la sélection actuelle."
                )
            _render_result(saved_result)
        else:
            _render_empty_result()

st.markdown(
    """
    <footer class="app-footer">
      ARVI · Projet EFREI Solution Delivery Data · Prototype pédagogique responsable
    </footer>
    """,
    unsafe_allow_html=True,
)
