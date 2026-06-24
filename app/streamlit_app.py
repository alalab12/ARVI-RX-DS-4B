from __future__ import annotations

import tempfile
import re
from pathlib import Path
import streamlit as st
from PIL import Image

from src.inference import predict

st.set_page_config(page_title="Assistant radiologue virtuel", layout="wide")
st.title("Assistant radiologue virtuel — prototype pédagogique")
st.warning("Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise.")

uploaded = st.file_uploader("Déposer une radiographie thoracique frontale", type=["png", "jpg", "jpeg"])
mode = st.selectbox("Mode", ["baseline", "improved"])

if uploaded:
    original_name = Path(uploaded.name or "image.png").name
    suffix = Path(original_name).suffix or ".png"
    stem = Path(original_name).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    tmp_dir = Path(tempfile.mkdtemp(prefix="assistant_radio_"))
    tmp_path = tmp_dir / f"{safe_stem}{suffix}"
    tmp_path.write_bytes(uploaded.read())

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(Image.open(tmp_path), caption="Image uploadée", use_container_width=True)
    with col2:
        pred = predict(tmp_path, mode=mode)
        st.metric("Classe", pred["predicted_class"])
        st.metric("Confiance", pred["confidence"])
        st.write("**Observations**", pred["visual_evidence"])
        st.write("**Justification**", pred["justification"])
        st.write("**Limites**", pred["limitations"])
        st.json(pred)
else:
    st.info("Utiliser les images synthétiques dans data/sample_images pour tester le flux.")
