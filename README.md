# ARVI — Assistant Radiologue Virtuel

Prototype pédagogique de classification de radiographies thoraciques avec des
modèles ONNX entraînés par l'équipe.

> Ce projet n'est pas destiné au diagnostic. Toute sortie doit être validée
> par un professionnel qualifié.

## Modèles disponibles

| Variante | Politique des labels incertains |
|---|---|
| `u_ones` | `-1` et valeurs manquantes → `suspected_opacity` |
| `u_zeros` | `-1` et valeurs manquantes → `normal` |

Les deux variantes utilisent un ResNet-18, une entrée RGB
`[batch, 3, 224, 224]` normalisée avec les statistiques ImageNet et une sortie
`logits` ordonnée comme suit : `normal`, `suspected_opacity`.

## Installation

Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Sous Windows :

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Fichiers des modèles

Les binaires ne sont pas versionnés dans Git. Ils doivent être placés ensemble
dans `models/` :

```text
arvi_cxr_classifier_u_ones.onnx
arvi_cxr_classifier_u_ones.onnx.data
arvi_cxr_classifier_u_zeros.onnx
arvi_cxr_classifier_u_zeros.onnx.data
```

Les contrats d'inférence sont définis dans `config/onnx_u_ones.json` et
`config/onnx_u_zeros.json`.

## Lancer Streamlit

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Puis ouvrir [http://localhost:8501](http://localhost:8501). Dans la barre
latérale, choisir le classifieur ONNX puis la variante à tester.

## Lancer l'API

```bash
MODEL_BACKEND=onnx .venv/bin/python -m uvicorn api.main:app --reload
```

Endpoint : `POST /predict` avec une image PNG, JPG, JPEG ou BMP envoyée en
`multipart/form-data` dans le champ `file`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

Les images de `data/sample_images/` vérifient uniquement la chaîne logicielle ;
elles ne mesurent pas les performances médicales des modèles.

## Structure

```text
app/          interface Streamlit
api/          API FastAPI
config/       registre et contrats ONNX
data/         images synthétiques de démonstration
docs/         architecture actuelle
models/       manifests et binaires locaux ignorés par Git
notebooks/    entraînement des variantes u_ones et u_zeros
src/          inférence, prétraitement et garde-fous
tests/        tests unitaires et d'intégration
```
