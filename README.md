---
title: ARVI - Assistant radiologue virtuel
sdk: gradio
sdk_version: 5.49.1
app_file: app/hf_space_app.py
python_version: "3.11"
suggested_hardware: t4-small
models:
  - google/medsiglip-448
  - google/medgemma-4b-it
pinned: false
---

# ARVI-RX-DS-4B

Assistant radiologue virtuel responsable - projet EFREI Solution Delivery Data.

Ce depot contient le prototype de notre groupe pour analyser une radiographie
thoracique frontale dans un cadre pedagogique. Le projet ne vise pas le
diagnostic medical : il sert a construire une chaine IA prudente, tracable et
evaluee.

## Objectif du projet

L'application doit recevoir une image de radiographie thoracique et retourner
une sortie JSON structuree avec :

- `image_quality` : qualite de l'image (`good`, `limited`, `poor`)
- `predicted_class` : classe predite (`normal`, `suspected_opacity`, `uncertain`)
- `confidence` : score de confiance entre 0 et 1
- `visual_evidence` : observations visuelles courtes
- `justification` : justification prudente et limitee a ce qui est visible
- `limitations` : limites de l'analyse
- `warning` : avertissement obligatoire de non-usage clinique

Warning obligatoire :

```text
Prototype pedagogique. Non destine au diagnostic. Validation par un professionnel qualifie requise.
```

## Installation Windows

Cloner le depot :

```powershell
git clone https://github.com/alalab12/ARVI-RX-DS-4B.git
cd ARVI-RX-DS-4B
```

Creer l'environnement virtuel :

```powershell
python -m venv .venv
```

Installer les dependances minimales pour lancer l'application et les tests :

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install streamlit pillow pandas numpy scikit-learn pytest fastapi uvicorn python-multipart httpx
```

Installation complete du projet, plus longue car elle peut installer `torch` et
`transformers` :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Lancer l'application Streamlit

Depuis la racine du depot :

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

L'application s'ouvre normalement sur :

```text
http://localhost:8501/
```

Si la commande `streamlit` n'est pas reconnue, utiliser toujours :

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

## Lancer l'API FastAPI

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Endpoint principal :

```http
POST /predict
Content-Type: multipart/form-data
```

Exemple de test local :

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@data/sample_images/CXR_SYN_002_suspected_opacity.png"
```

## Verifier que le projet fonctionne

Lancer les tests :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Lancer l'evaluation jouet :

```powershell
.\.venv\Scripts\python.exe eval\run_evaluation.py --mode toy
```

Les sorties d'evaluation sont generees dans `eval/outputs/` et les logs SQLite
dans `medical_ai_evidence.sqlite`. Ces fichiers sont locaux et ne doivent pas
etre commits.

## Executer MedGemma sur Kaggle

Le backend reste `toy` par defaut. Le modele MedGemma n'est donc ni telecharge
ni charge pendant les tests locaux.

Dans un notebook Kaggle avec GPU, cloner le depot puis installer les
dependances :

```python
!git clone https://github.com/alalab12/ARVI-RX-DS-4B.git
%cd ARVI-RX-DS-4B
!pip install -q -r requirements.txt
```

Ajouter `HF_TOKEN` dans les secrets Kaggle apres avoir obtenu l'acces au modele,
puis selectionner le vrai backend :

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["MODEL_BACKEND"] = "medgemma"
os.environ["MEDGEMMA_MODEL_ID"] = "google/medgemma-4b-it"

from src.inference import predict

result = predict("/kaggle/input/chexpert/path/to/frontal-image.jpg", mode="baseline")
result
```

Le modele est charge au premier appel puis reutilise. La confiance retournee
est declaree par le modele generatif et n'est pas une probabilite calibree.
Les modes `improved_v1` a `improved_v5` conservent l'historique experimental.
Le mode `improved` reste un alias vers `v3`, la meilleure version validee sur
le pilote a ce stade. La v4 est rejetee car elle produit 90 % de sorties
`uncertain` et ne detecte aucune des 10 opacites du dev pilot. La v5 est une
candidate a tester explicitement avec `mode="improved_v5"`; elle distingue
l'incertitude sur la presence d'une opacite de l'incertitude sur sa cause.

## Evaluer MedSigLIP sur Kaggle

Le notebook `notebooks/04_medsiglip_zero_shot.ipynb` evalue
`google/medsiglip-448` sur le split dev sans modifier le backend MedGemma. Il
produit les scores bruts, calibre une zone `uncertain` sur les seuls labels
definitifs, mesure la latence et compare les etudes mono-image aux etudes
multi-images. Les seuils et le split final ne doivent pas etre ajustes ensemble.
La configuration retenue est gelee dans
`config/medsiglip_zero_shot_v1.json`. La cellule d'evaluation finale reste
desactivee par defaut avec `RUN_FINAL = False`.

Le notebook `notebooks/05_medgemma_baseline_final.ipynb` applique ensuite le
baseline MedGemma fige sur le meme final. Son manifeste
`config/medgemma_baseline_v1.json` contient notamment l'empreinte SHA-256 du
prompt afin d'interdire une modification silencieuse apres evaluation.

Le notebook `notebooks/06_hybrid_medsiglip_medgemma_dev.ipynb` developpe enfin
un routage selectif sur `dev` uniquement. Il reutilise les scores MedSigLIP,
appelle MedGemma autour des seuils et compare des politiques conservatrices
`agreement_or_abstain`. Il refuse toute ligne du split `final`; la politique
candidate doit etre evaluee sur une nouvelle cohorte patient-disjointe.

## Deployer la demonstration GPU

Le point d'entree `app/hf_space_app.py` fournit une interface Gradio adaptee a
un Hugging Face Space GPU. MedSigLIP produit la seule decision applicative;
MedGemma genere, a la demande, une analyse textuelle independante et un statut
de concordance. Le guide complet se trouve dans
`docs/deployment_huggingface_space.md`.

## Documentation projet

- `docs/appel_offre.md` : cadrage et attendus du projet.
- `docs/architecture.md` : pipeline cible et composants techniques.
- `docs/evaluation_protocol.md` : metriques et protocole de validation.
- `docs/journal_experimental.md` : resultats, erreurs et decisions accumules.
- `docs/deployment_huggingface_space.md` : deploiement GPU distant et cout.
- `docs/ethique_et_limites.md` : avertissements, limites et garde-fous.
- `docs/etat_de_l_art_choix_technos.md` : etat de l'art, choix technologiques
  et justification autour de CheXpert Small.

## Structure du depot

```text
ARVI-RX-DS-4B/
|-- api/            # API FastAPI, endpoint /predict
|-- app/            # Interfaces Streamlit et Gradio
|-- data/           # Images synthetiques et fichier de cas
|-- docs/           # Appel d'offre, architecture, ethique, evaluation
|-- eval/           # Script d'evaluation et registre d'erreurs
|-- finetuning/     # Stubs experimentaux, non prioritaires
|-- notebooks/      # Notebooks de demarrage
|-- prompts/        # Prompts baseline, improved et schema JSON
|-- sql/            # Schema SQLite
|-- src/            # Preprocessing, inference, guardrails, metrics, database
|-- tests/          # Smoke tests du projet
```

## Repartition conseillee

- Integration et coordination : installation, GitHub, branches, merge final.
- Baseline IA : tester le prompt baseline et documenter les resultats.
- Amelioration : prompt renforce, seuil d'incertitude, comparaison avant/apres.
- Evaluation : metriques, CSV, registre d'erreurs sur 20 a 30 cas.
- Rapport et soutenance : dataset, licences, limites, ethique, slides.

## Workflow Git conseille

Avant de travailler :

```powershell
git pull
```

Creer une branche par tache :

```powershell
git checkout -b feature/nom-de-la-tache
```

Verifier les changements :

```powershell
git status
git diff
```

Committer :

```powershell
git add .
git commit -m "Description courte de la modification"
git push -u origin feature/nom-de-la-tache
```

Ensuite, ouvrir une Pull Request sur GitHub.

## Regles importantes

- Ne jamais presenter le prototype comme un outil de diagnostic.
- Ne jamais supprimer la classe `uncertain`.
- Ne jamais committer de donnees patient reelles ou identifiantes.
- Toujours garder le warning dans l'interface, le JSON, le rapport et la soutenance.
- Toujours montrer des erreurs et limites en soutenance, pas seulement les cas reussis.

## Priorite du groupe

Pour un delai de 3 semaines, la priorite est :

1. Faire tourner le pipeline existant de bout en bout.
2. Avoir une baseline reproductible.
3. Ajouter une amelioration simple et mesuree.
4. Produire les metriques, les logs et le registre d'erreurs.
5. Finaliser le rapport et une demonstration stable.

Les pistes LoRA, MedGemma ou fine-tuning sont optionnelles. Elles ne doivent
etre tentees qu'apres une baseline fonctionnelle et evaluee.
