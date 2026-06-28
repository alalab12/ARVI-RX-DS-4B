# Protocole d'évaluation
> **Author :** Badr TAJINI 
> **Solution Delivery - filière Data** 
>  **Année académique :** 2025-2026
## Jeux de cas

- `smoke` : 20 images pour vérifier la chaîne.
- `dev` : 100 à 150 cas si un vrai dataset est utilisé.
- `final` : 20 à 30 cas commentés pour la soutenance.

Le jeu synthétique fourni sert uniquement à valider le pipeline logiciel : chargement, inférence jouet, JSON, logs, métriques et garde-fous. Un score parfait sur ce jeu ne constitue pas une performance médicale.

## Métriques minimales

- Accuracy.
- Macro-F1.
- Sensibilité sur les cas `suspected_opacity`.
- Spécificité sur les cas `normal`.
- Taux de JSON valide.
- Taux de warning présent.
- Taux d'incertitude.
- Hallucinations textuelles détectées manuellement.
- Latence médiane.

## Taxonomie d'erreurs

| Code | Signification | Exemple |
|---|---|---|
| FN | Faux négatif | anomalie présente prédite normale |
| FP | Faux positif | image normale prédite suspecte |
| UA | Incertitude acceptable | signes faibles ou image limitée |
| JF | JSON format error | sortie non exploitable |
| HT | Hallucination textuelle | mention d'un signe non visible |

## Règle de soutenance

Ne jamais montrer seulement des réussites. Une bonne défense montre aussi les faux positifs, les faux négatifs, les incertitudes et les limites de qualité image.

## Décisions sur les prompts

Le prompt `improved_v1` est rejeté après le smoke test MedGemma : il augmente
les opacités prédites normales et renforce le biais vers la classe `normal`.
La version `improved_v2` réduit les opacités prédites normales, mais ne produit
plus aucune détection positive et atteint un taux d'incertitude de 75 % sur le
smoke test. Elle est conservée comme expérience, mais rejetée pour la suite.

La version `improved_v3` donne la priorité à une opacité visible, même sur une
vue AP ou portable, sauf si l'image est réellement non interprétable. Elle
interdit aussi l'invention de causes ou d'antécédents. Les garde-fous
convertissent `poor` et `limited + normal` en `uncertain`.

Les sorties `raw_predicted_class`, `raw_confidence` et `raw_image_quality`
conservent la décision du modèle avant garde-fous. `guardrail_actions` ne doit
contenir que les transformations effectivement appliquées.

Critères smoke pour `improved_v3` : au moins 3 opacités détectées sur 7, au
maximum une opacité prédite normale et un taux d'incertitude inférieur à 75 %.

La version `improved_v4` vient du dev pilot de 30 images CheXpert. Elle garde
la priorite aux opacites visibles, mais corrige l'effet principal observe avec
v3 : les images `limited` etaient trop souvent classees `suspected_opacity` sur
des formulations vagues comme "hazy lower lung fields" ou "could be due to
various factors". La v4 demande donc `uncertain` quand l'evidence est non
specifique, tout en gardant `suspected_opacity` pour une opacite visible,
localisee ou clairement decrite.

La v4 est rejetee apres evaluation sur les 30 images du dev pilot : accuracy
0,367, macro-F1 0,273, sensibilite opacite 0 % et taux d'incertitude 90 %. Les
10 opacites ont ete classees `uncertain`. L'analyse montre que le prompt a
confondu l'incertitude sur la cause d'une densite visible avec l'incertitude
sur sa presence.

La version `improved_v5` est une candidate non encore validee. Elle evalue
d'abord si l'opacite est PRESENT, INDETERMINATE ou ABSENT, puis mappe cette
presence vers la classe projet. Une opacite visible dont la cause est
incertaine reste `suspected_opacity`. Jusqu'a validation, l'alias `improved`
reste fixe sur v3. Pour etre promue, v5 doit obtenir sur le meme pilote une
sensibilite opacite d'au moins 80 %, au maximum une opacite predite `normal` et
un taux d'incertitude inferieur ou egal a 50 %.

## Smoke test attendu

Avant toute démonstration, le dépôt doit passer un contrôle court :

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python eval/run_evaluation.py --mode toy --out-dir /tmp/assistant-radio-eval --db-path /tmp/assistant-radio-evidence.sqlite
```

Ce test ne remplace pas l'analyse d'erreurs. Il vérifie seulement que le dépôt est exécutable, que les avertissements sont présents et que les sorties restent structurées.
