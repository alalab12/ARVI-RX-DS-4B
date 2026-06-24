# Etat de l'art et choix technologiques

## Contexte du projet

Le projet vise un prototype pedagogique d'assistant radiologue pour radiographies
thoraciques. Le systeme ne doit pas poser de diagnostic. Il doit produire une
sortie structuree, prudente et tracable :

- classe predite : `normal`, `suspected_opacity`, `uncertain`
- score de confiance
- observations visuelles limitees a l'image
- limites et avertissement non clinique
- logs, metriques et registre d'erreurs

Le groupe utilisera CheXpert-v1.0-small depuis Kaggle comme dataset de travail.
CheXpert est un dataset de radiographies thoraciques avec labels d'incertitude.
Le papier original indique 224 316 radiographies de 65 240 patients et 14
observations radiologiques annotees comme positives, negatives ou incertaines.

Sources principales :

- CheXpert paper : https://arxiv.org/abs/1901.07031
- Site Stanford CheXpert : https://stanfordmlgroup.github.io/competitions/chexpert/
- CheXpert-v1.0-small Kaggle : https://www.kaggle.com/datasets/ashery/chexpert
- Datasheet CheXpert : https://arxiv.org/abs/2105.03020

## Etat de l'art

### 1. CNN supervises pour classification CXR

Les methodes historiques fortes pour la classification de radiographies
thoraciques reposent sur des CNN supervises, notamment DenseNet, ResNet et
EfficientNet. CheXNet a popularise l'utilisation de DenseNet-121 pour la
detection de pneumonie sur radiographies thoraciques.

Sur CheXpert, les meilleurs resultats historiques du leaderboard sont souvent
des ensembles de CNN, avec des AUC autour de 0.93. Le baseline Stanford
historique est aussi base sur des CNN et sert de reference solide.

Conclusion pour le projet :

- c'est la famille la plus realiste a implementer en 3 semaines ;
- elle produit des probabilites exploitables pour `confidence` ;
- elle s'integre facilement dans l'API et l'evaluation existantes ;
- elle permet Grad-CAM pour une visualisation simple des zones influentes.

Sources :

- CheXNet : https://arxiv.org/abs/1711.05225
- CheXpert : https://arxiv.org/abs/1901.07031
- CheXtransfer : https://arxiv.org/abs/2101.06871

### 2. Modeles pre-entraines specialises CXR

TorchXRayVision fournit des datasets, pretraitements et modeles pre-entraines
pour la radiographie thoracique. Il propose notamment des DenseNet-121
pre-entrainees sur CheXpert, NIH, RSNA, MIMIC-CXR ou des combinaisons de
datasets.

Conclusion pour le projet :

- c'est le meilleur choix pour une baseline rapide et credible ;
- on evite de partir de zero ;
- on beneficie d'un preprocessing CXR standardise ;
- on peut comparer une inference pre-entrainee a un fine-tuning leger.

Source :

- TorchXRayVision : https://github.com/mlmed/torchxrayvision
- Papier TorchXRayVision : https://arxiv.org/abs/2111.00595

### 3. Apprentissage vision-langage medical

Les modeles vision-langage medicaux apprennent a aligner images et textes
radiologiques. Les travaux importants incluent ConVIRT, MedCLIP, BioViL-T,
GLoRIA et CheXzero. Leur interet est la meilleure reutilisation des rapports
radiologiques et la possibilite de zero-shot ou few-shot.

Conclusion pour le projet :

- interessant pour l'etat de l'art et la discussion ;
- trop risque comme axe principal en 3 semaines ;
- possible en option pour une comparaison qualitative ou une ouverture.

Sources :

- ConVIRT : https://arxiv.org/abs/2010.00747
- MedCLIP : https://arxiv.org/abs/2210.10163
- BioViL-T : https://arxiv.org/abs/2301.04558
- CheXzero : https://arxiv.org/abs/2201.11117

### 4. VLM generatifs et agents CXR

Les modeles recents comme CheXagent ou CheXOne visent l'interpretation CXR avec
raisonnement, reponse textuelle ou generation de compte-rendu. Ils representent
la frontiere recherche 2024-2026.

Conclusion pour le projet :

- utile pour montrer que l'on connait les tendances recentes ;
- non prioritaire pour l'implementation ;
- risque de hallucination textuelle et de cout technique eleve ;
- a garder comme perspective, pas comme socle de livraison.

Sources :

- CheXagent : https://arxiv.org/abs/2401.12208
- CheXOne : https://arxiv.org/abs/2604.00493

## Choix dataset

### Dataset retenu

Dataset : CheXpert-v1.0-small.

Justification :

- dataset reconnu pour la radiographie thoracique ;
- labels multi-pathologies et labels d'incertitude ;
- format exploitable en CSV ;
- version small plus compatible avec les contraintes machines du groupe ;
- documentable et defendable dans le rapport.

### Perimetre recommande

Pour rester coherent avec le projet, il faut reduire CheXpert a une tache
simple :

- utiliser uniquement les vues frontales : `Frontal`, `AP`, `PA` ;
- construire une tache binaire prudente autour de `Lung Opacity` ;
- convertir ensuite en trois classes applicatives.

Mapping propose :

- `normal` : `No Finding = 1` et absence des observations ciblees ;
- `suspected_opacity` : `Lung Opacity = 1`, eventuellement avec
  `Consolidation = 1` ou `Pneumonia = 1` comme signaux secondaires ;
- `uncertain` : label CheXpert incertain (`-1`), qualite faible, conflit de
  labels ou confiance modele inferieure au seuil choisi.

Important :

- ne pas traiter CheXpert comme une verite clinique parfaite ;
- les labels d'entrainement viennent de rapports et contiennent du bruit ;
- faire un split par patient pour eviter la fuite de donnees.

## Choix technologiques recommandes

### Langage et framework ML

Choix : Python + PyTorch.

Justification :

- ecosysteme standard en deep learning medical ;
- compatible avec TorchXRayVision, torchvision, timm et Grad-CAM ;
- deja coherent avec le depot actuel ;
- facilite l'integration dans FastAPI et Streamlit.

### Baseline modele

Choix principal du cahier des charges : MedGemma 4B instruction-tuned en
prompting, sans entrainement initial.

Justification :

- modele vision-langage medical capable de recevoir une radiographie et une
  instruction textuelle ;
- generation directe du JSON, des observations, des limites et du warning ;
- aucune phase d'entrainement pour etablir la baseline ;
- comparaison rapide de plusieurs prompts ;
- alignement avec l'architecture et la voie rapide du cahier des charges.

Important : la confiance generee par le VLM est auto-declaree et non calibree.
Elle doit etre presentee comme telle et encadree par les garde-fous.

### Classifieur complementaire

Choix optionnel : DenseNet-121 pre-entrainee via TorchXRayVision.

Role :

- recuperer la probabilite `Lung Opacity` ;
- optionnellement combiner avec `Consolidation` et `Pneumonia` ;
- fournir un second signal quantitatif au VLM ;
- retourner `uncertain` en cas de faible confiance ou de desaccord.

Exemple de regle simple :

```text
score_opacity = max(P(Lung Opacity), P(Consolidation), P(Pneumonia))

si score_opacity >= 0.60 -> suspected_opacity
si score_opacity <= 0.35 et No Finding eleve -> normal
sinon -> uncertain
```

### Amelioration modele

Choix : prompt ameliore et garde-fous d'abord, LoRA seulement apres validation
de la baseline.

Justification :

- aucune optimisation de poids avant d'avoir mesure les erreurs ;
- comparaison mesurable entre prompt baseline et prompt ameliore ;
- compatible avec les ressources d'un groupe et un delai court ;
- defendable en soutenance.

Ameliorations possibles :

- validation stricte du JSON ;
- regles `uncertain` pour les cas ambigus ;
- ajout optionnel du classifieur DenseNet ;
- calibration des seuils du classifieur ;
- LoRA sur MedGemma uniquement si sa valeur ajoutee peut etre evaluee.

### Interface

Choix : Streamlit.

Justification :

- deja present dans le depot ;
- ideal pour demo rapide ;
- upload image, affichage JSON, warning et metriques simples ;
- moins couteux qu'un frontend React pour ce projet.

### API

Choix : FastAPI.

Justification :

- deja present dans le depot ;
- endpoint `/predict` simple ;
- permet de separer le modele de l'interface ;
- utile pour tester automatiquement le contrat JSON.

### Evaluation

Choix : scikit-learn + pandas + SQLite.

Justification :

- pandas pour construire `metadata.csv` et les CSV de resultats ;
- scikit-learn pour AUC, F1, precision, recall, matrice de confusion ;
- SQLite pour journaliser les predictions, modeles, prompts et latences ;
- deja aligne avec le depot.

Metriques prioritaires :

- ROC-AUC sur `Lung Opacity` ;
- PR-AUC si classes desequilibrees ;
- macro-F1 sur les 3 classes projet ;
- sensibilite pour `suspected_opacity` ;
- specificite pour `normal` ;
- taux d'incertitude ;
- taux de JSON valide ;
- taux de warning present ;
- latence mediane.

### Interpretabilite

Choix : Grad-CAM sur le CNN.

Justification :

- methode standard pour visualiser les zones influentes d'un CNN ;
- utile en demo pour expliquer la prediction ;
- compatible avec DenseNet/ResNet ;
- doit rester presente comme aide visuelle, pas comme preuve clinique.

Source :

- Grad-CAM : https://arxiv.org/abs/1610.02391

## Architecture cible

```text
CheXpert small CSV
-> preparation metadata.csv
-> filtrage frontal AP/PA
-> mapping labels projet
-> preprocessing image
-> MedGemma 4B + prompt versionne
-> parsing JSON + garde-fous
-> classifieur DenseNet optionnel
-> JSON de sortie
-> Streamlit + FastAPI
-> logs SQLite
-> evaluation CSV/JSON
```

## Plan technique recommande

### Priorite 1 - Dataset

- telecharger CheXpert-v1.0-small localement 
- ne pas committer les images dans Git ;
- creer `data/metadata_chexpert.csv` ;
- filtrer les vues frontales ;
- definir `normal`, `suspected_opacity`, `uncertain` ;
- documenter source, licence, taille, limites.
          OU 
ou optionnellement l'utiliser directement dans Kaggle (juste pour la phase d'entrainement. On pourrait télécharger le modèle final plus tard en local);

### Priorite 2 - Baseline

- charger MedGemma 4B sur Kaggle avec `src/medgemma.py` ;
- comparer les prompts baseline et improved ;
- valider strictement le JSON et appliquer les garde-fous ;
- retourner le meme schema JSON que le prototype actuel.

### Priorite 3 - Evaluation

- adapter `eval/run_evaluation.py` pour CheXpert ;
- produire `predictions.csv`, `metrics.json`, `before_after_summary.csv` ;
- garder 20 a 30 cas commentes dans un registre d'erreurs.

### Priorite 4 - Amelioration

- ameliorer le prompt et les garde-fous ;
- optionnellement ajouter puis calibrer DenseNet-121 ;
- envisager LoRA uniquement apres validation ;
- comparer baseline vs improved.

### Priorite 5 - Rapport et soutenance

- presenter le dataset et ses limites ;
- expliquer pourquoi le modele n'est pas clinique ;
- montrer les metriques et les erreurs ;
- montrer le warning et le JSON ;
- expliquer les choix technologiques.

## Decision recommandee

Le choix le plus solide pour le groupe est :

```text
CheXpert-v1.0-small
+ PyTorch
+ MedGemma 4B instruction-tuned + prompting
+ parsing JSON strict + classe uncertain
+ DenseNet-121 optionnelle comme signal complementaire
+ Streamlit/FastAPI
+ pandas/scikit-learn/SQLite
```

Ce choix est assez proche de l'etat de l'art pour etre credible, mais assez
simple pour etre livre en 3 semaines. LoRA reste une extension conditionnelle,
pas un prerequis de la baseline.
