from __future__ import annotations

from typing import Any

import gradio as gr

from src.explanation import ExplanationService, hash_image
from src.inference import predict


LABELS = {
    "normal": "Normal",
    "suspected_opacity": "Opacite suspectee",
    "uncertain": "Incertain",
}
AGREEMENT_LABELS = {
    "agreement": "Accord entre les deux analyses",
    "partial_disagreement": "Desaccord partiel: au moins un modele est incertain",
    "disagreement": "Desaccord entre MedSigLIP et l'analyse MedGemma",
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
        f"**Score relatif d'opacite :** `{score:.3f}`  \n"
        f"**Seuils figes :** normal <= `{result['low_threshold']:.3f}`, "
        f"opacite >= `{result['high_threshold']:.3f}`  \n"
        f"**Latence :** `{result['latency_ms']} ms`\n\n"
        "Le score est une similarite relative non calibree, pas une probabilite clinique."
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
    cache_note = "resultat mis en cache" if response["cached"] else "nouvelle inference"
    evidence = analysis.get("visual_evidence") or ["Aucune observation valide retournee"]
    evidence_markdown = "\n".join(f"- {item}" for item in evidence)
    summary = (
        f"## Analyse textuelle independante\n"
        f"**Concordance :** {agreement}  \n"
        f"**Qualite declaree :** `{analysis['image_quality']}`  \n"
        f"**Evaluation MedGemma :** `{LABELS[analysis['predicted_class']]}`  \n"
        f"**Execution :** {cache_note}\n\n"
        f"**Elements visibles rapportes**\n{evidence_markdown}\n\n"
        f"**Justification prudente**\n{analysis['justification']}\n\n"
        "Cette analyse ne modifie pas la decision MedSigLIP."
    )
    return summary, response


theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans")],
)

CSS = """
.gradio-container {
  background:
    radial-gradient(circle at 12% 8%, rgba(217, 119, 6, 0.15), transparent 28rem),
    linear-gradient(145deg, #f8f4ea 0%, #edf4f1 52%, #e7edf3 100%);
}
.hero {
  border-left: 6px solid #b45309;
  padding: 0.4rem 0 0.4rem 1.2rem;
  margin-bottom: 1rem;
}
.hero h1 { color: #172033; letter-spacing: -0.035em; }
.safety-note {
  background: #fff7df;
  border: 1px solid #d6a74f;
  border-radius: 12px;
  color: #5c3a05;
  padding: 0.85rem 1rem;
}
"""

with gr.Blocks(theme=theme, css=CSS, title="ARVI - Assistant radiologue virtuel") as demo:
    decision_state = gr.State({})
    gr.Markdown(
        """
        <div class="hero">
          <h1>ARVI</h1>
          <p>Lecture assistee d'une radiographie thoracique, du signal rapide au commentaire prudent.</p>
        </div>
        <div class="safety-note">
          Prototype pedagogique. Non destine au diagnostic. Toute image doit etre verifiee par un professionnel qualifie.
        </div>
        """
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
                    "2. Generer l'analyse MedGemma",
                    variant="secondary",
                )
        with gr.Column(scale=6):
            classification_summary = gr.Markdown("## Decision principale\nEn attente d'une image.")
            classification_json = gr.JSON(label="Trace MedSigLIP", open=False)
            explanation_summary = gr.Markdown("## Analyse textuelle\nDeclenchee uniquement a la demande.")
            explanation_json = gr.JSON(label="Trace MedGemma", open=False)

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
