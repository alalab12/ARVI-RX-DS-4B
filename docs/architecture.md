# Architecture ARVI

## Pipeline

```text
Image → validation du format → prétraitement configuré
      → ONNX Runtime → softmax → classe et scores
      → abstention éventuelle → garde-fous → Streamlit/API
```

## Composants conservés

- `app/streamlit_app.py` : interface principale et sélection des variantes.
- `api/main.py` : endpoint HTTP `POST /predict`.
- `src/onnx_backend.py` : chargement ONNX, prétraitement et post-traitement.
- `src/inference.py` : orchestration ONNX et mode synthétique de test.
- `src/preprocessing.py` : chargement sécurisé des images.
- `src/guardrails.py` : validation de la sortie et avertissement obligatoire.
- `config/onnx_models.json` : registre des variantes disponibles.
- `config/onnx_*.json` : contrat exact de chaque modèle.

Une session ONNX Runtime est chargée à la demande et mise en cache pour chaque
configuration. Les fichiers `.onnx.data` sont vérifiés avant le chargement.

## Sortie commune

Chaque prédiction contient notamment :

```json
{
  "predicted_class": "normal | suspected_opacity | uncertain",
  "confidence": 0.0,
  "class_scores": {},
  "limitations": [],
  "warning": "Prototype pédagogique...",
  "model_name": "...",
  "model_version": "...",
  "latency_ms": 0
}
```

Les scores du modèle ne sont pas présentés comme des probabilités cliniques
calibrées.
