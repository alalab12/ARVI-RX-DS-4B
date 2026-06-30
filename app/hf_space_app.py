from __future__ import annotations

from typing import Any

import gradio as gr

from src.explanation import ExplanationService, hash_image
from src.inference import predict


LABELS = {
    "normal": "Normal",
    "suspected_opacity": "Opacité suspectée",
    "uncertain": "Incertain",
}
QUALITY_LABELS = {
    "good": "Bonne",
    "limited": "Limitée",
    "poor": "Mauvaise",
}
AGREEMENT_LABELS = {
    "agreement": "Accord entre les deux analyses",
    "partial_disagreement": "Désaccord partiel : au moins un modèle est incertain",
    "disagreement": "Désaccord entre MedSigLIP et l'analyse MedGemma",
}
explanation_service = ExplanationService()


def classify_image(image_path: str | None) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not image_path:
        raise gr.Error("Ajoutez une radiographie thoracique frontale.")

    result = predict(image_path, backend="medsiglip")
    predicted_class = result["predicted_class"]
    score = result["score_opacity"]
    summary = (
        f"## {LABELS[predicted_class]}\n"
        f"**Score relatif d'opacité :** `{score:.3f}`  \n"
        f"**Seuils figés :** normal <= `{result['low_threshold']:.3f}`, "
        f"opacité >= `{result['high_threshold']:.3f}`  \n"
        f"**Latence :** `{result['latency_ms']} ms`\n\n"
        "Le score est une similarité relative non calibrée, pas une probabilité clinique."
    )
    state = {
        "image_sha256": hash_image(image_path),
        "primary_class": predicted_class,
    }
    return summary, result, state


def explain_image(
    image_path: str | None,
    decision_state: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if not image_path:
        raise gr.Error("Ajoutez une radiographie thoracique frontale.")
    if not decision_state:
        raise gr.Error("Lancez d'abord la classification MedSigLIP.")
    if decision_state.get("image_sha256") != hash_image(image_path):
        raise gr.Error("L'image a change. Relancez d'abord la classification.")

    response = explanation_service.explain(
        image_path,
        decision_state["primary_class"],
    )
    analysis = response["analysis"]
    agreement = AGREEMENT_LABELS[response["agreement_status"]]
    cache_note = "résultat mis en cache" if response["cached"] else "nouvelle inférence"
    evidence = analysis.get("visual_evidence") or ["Aucune observation valide retournée"]
    evidence_markdown = "\n".join(f"- {item}" for item in evidence)
    quality_label = QUALITY_LABELS.get(
        analysis["image_quality"],
        analysis["image_quality"],
    )
    summary = (
        f"## Analyse textuelle indépendante\n"
        f"**Concordance :** {agreement}  \n"
        f"**Qualité déclarée :** `{quality_label}`  \n"
        f"**Évaluation MedGemma :** `{LABELS[analysis['predicted_class']]}`  \n"
        f"**Exécution :** {cache_note}\n\n"
        f"**Éléments visibles rapportés**\n{evidence_markdown}\n\n"
        f"**Justification prudente**\n{analysis['justification']}\n\n"
        "Cette analyse ne modifie pas la décision MedSigLIP."
    )
    return summary, response


theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans")],
)

CSS = """
html, body {
  background: #edf3ef;
  color-scheme: light;
}
.gradio-container {
  --arvi-ink: #172033;
  --arvi-muted: #526174;
  --arvi-surface: rgba(255, 255, 255, 0.92);
  color: var(--arvi-ink) !important;
  min-height: 100vh;
  background:
    radial-gradient(circle at 10% 7%, rgba(217, 119, 6, 0.20), transparent 30rem),
    radial-gradient(circle at 88% 18%, rgba(15, 118, 110, 0.15), transparent 26rem),
    linear-gradient(145deg, #fffaf0 0%, #edf6f2 52%, #e7eef4 100%);
}
.hero-block {
  background: var(--arvi-surface);
  border: 1px solid rgba(71, 85, 105, 0.18);
  border-radius: 18px;
  box-shadow: 0 18px 44px rgba(30, 41, 59, 0.08);
  margin-bottom: 1.25rem;
  padding: 1.2rem 1.35rem;
}
.hero-block .prose,
.hero-block .prose p,
.hero-block p {
  color: var(--arvi-muted) !important;
}
.hero-block .prose h1,
.hero-block h1 {
  border-left: 6px solid #b45309;
  color: var(--arvi-ink) !important;
  letter-spacing: -0.035em;
  margin-bottom: 0.65rem;
  padding: 0.25rem 0 0.25rem 1rem;
}
.hero-block .prose blockquote,
.hero-block blockquote {
  background: #fff7df;
  border: 1px solid #d6a74f;
  border-left: 5px solid #b45309;
  border-radius: 12px;
  margin: 0.9rem 0 0;
  padding: 0.75rem 0.95rem;
}
.hero-block .prose blockquote p,
.hero-block blockquote p {
  color: #5c3a05 !important;
  margin: 0;
}
.results-column {
  gap: 1rem;
}
.result-card {
  background: var(--arvi-surface) !important;
  border: 1px solid rgba(71, 85, 105, 0.18) !important;
  border-radius: 16px !important;
  box-shadow: 0 14px 34px rgba(30, 41, 59, 0.08);
  padding: 1rem 1.15rem !important;
}
.result-card .prose,
.result-card .prose :where(h1, h2, h3, h4, p, li, strong, em, span) {
  color: var(--arvi-ink) !important;
}
.result-card .prose p,
.result-card .prose li {
  line-height: 1.55;
}
.result-card .prose code {
  background: #ffedd5 !important;
  border: 1px solid #fed7aa;
  border-radius: 5px;
  color: #7c2d12 !important;
  padding: 0.08rem 0.32rem;
}
.decision-card {
  border-top: 4px solid #c2410c !important;
}
.decision-card .prose h2 {
  color: #9a3412 !important;
}
.explanation-card {
  border-top: 4px solid #0f766e !important;
}
.explanation-card .prose h2 {
  color: #115e59 !important;
}
.trace-card {
  border: 1px solid rgba(30, 41, 59, 0.24) !important;
  border-radius: 14px !important;
  box-shadow: 0 12px 28px rgba(30, 41, 59, 0.10);
  overflow: hidden;
}
@media (max-width: 768px) {
  .gradio-container {
    padding-left: 0.7rem !important;
    padding-right: 0.7rem !important;
  }
  .hero-block {
    border-radius: 14px;
    padding: 0.9rem;
  }
  .hero-block .prose h1,
  .hero-block h1 {
    padding-left: 0.75rem;
  }
  .result-card {
    padding: 0.85rem 0.9rem !important;
  }
}
"""

with gr.Blocks(theme=theme, css=CSS, title="ARVI - Assistant radiologue virtuel") as demo:
    decision_state = gr.State({})
    gr.Markdown(
        """
# ARVI
Lecture assistée d'une radiographie thoracique, du signal rapide au commentaire prudent.

> **Prototype pédagogique.** Non destiné au diagnostic. Toute image doit être vérifiée par un professionnel qualifié.
        """,
        elem_classes=["hero-block"],
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image = gr.Image(
                type="filepath",
                label="Radiographie thoracique frontale",
                height=520,
            )
            with gr.Row():
                classify_button = gr.Button(
                    "1. Classifier avec MedSigLIP",
                    variant="primary",
                )
                explain_button = gr.Button(
                    "2. Générer l'analyse MedGemma",
                    variant="secondary",
                )
        with gr.Column(scale=6, elem_classes=["results-column"]):
            classification_summary = gr.Markdown(
                "## Décision principale\nEn attente d'une image.",
                elem_classes=["result-card", "decision-card"],
            )
            classification_json = gr.JSON(
                label="Trace MedSigLIP",
                open=False,
                elem_classes=["trace-card"],
            )
            explanation_summary = gr.Markdown(
                "## Analyse textuelle\nDéclenchée uniquement à la demande.",
                elem_classes=["result-card", "explanation-card"],
            )
            explanation_json = gr.JSON(
                label="Trace MedGemma",
                open=False,
                elem_classes=["trace-card"],
            )

    classify_button.click(
        classify_image,
        inputs=[image],
        outputs=[classification_summary, classification_json, decision_state],
        api_name="classify",
        concurrency_limit=1,
        concurrency_id="gpu_models",
    )
    explain_button.click(
        explain_image,
        inputs=[image, decision_state],
        outputs=[explanation_summary, explanation_json],
        api_name="explain",
        concurrency_limit=1,
        concurrency_id="gpu_models",
    )

demo.queue(default_concurrency_limit=1, max_size=8)

if __name__ == "__main__":
    demo.launch()
