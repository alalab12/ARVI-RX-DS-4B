# Journal experimental

Ce document centralise les decisions, resultats et limites observes pendant
le developpement. Les valeurs proviennent des executions Kaggle du groupe.
Elles doivent rester accompagnees du protocole et ne constituent pas une
validation clinique.

## Regles de protocole

- CheXpert Small est utilise depuis Kaggle.
- Les vues analysees sont frontales AP ou PA.
- Les patients sont disjoints entre `smoke`, `dev` et `final`.
- Le dev contient 120 images : 40 `normal`, 40 `suspected_opacity` et 40
  labels CheXpert `-1` ranges dans la cohorte `uncertain`.
- Le final contient 30 images : 10 par groupe.
- Les labels `-1` proviennent d'une incertitude dans le rapport et ne sont pas
  une verite terrain visuelle pour la sortie applicative `uncertain`.
- Les labels CheXpert sont lies a l'etude. Une etude multi-vues peut donc
  attribuer a une frontale un signe surtout visible sur la vue laterale.
- Le split final ne sert jamais a modifier un prompt, un modele ou un seuil.

## Experiences de prompting MedGemma

Les confiances MedGemma sont auto-declarees et non calibrees.

| Version | Jeu | Accuracy | Macro-F1 | Sensibilite opacite | Taux uncertain | Decision |
|---|---:|---:|---:|---:|---:|---|
| baseline | pilote 30 | 0,600 | 0,578 | 0,600 | 0,233 | reference |
| improved_v3 | pilote 30 | 0,500 | 0,464 | 0,900 | 0,200 | sensible mais trop de surclassement |
| improved_v4 | pilote 30 | 0,367 | 0,273 | 0,000 | 0,900 | rejetee |
| improved_v5 | pilote 30 | 0,533 | 0,452 | 1,000 | 0,033 | candidate haute sensibilite |

Observations :

- `improved_v1` renforcait le biais vers `normal` et a ete rejetee.
- `improved_v2` ne produisait plus de detection positive et atteignait 75 %
  d'incertitude sur le smoke test.
- `improved_v3` a augmente la sensibilite, mais a surclasse des images
  limitees sur des formulations vagues.
- `improved_v4` a confondu incertitude sur la cause et incertitude sur la
  presence. Les dix opacites du pilote sont devenues `uncertain`.
- `improved_v5` a restaure la sensibilite, mais a deplace le point de
  fonctionnement vers davantage de faux positifs.

## Comparaison sur le dev definitif

Les 80 labels definitifs regroupent 40 `normal` et 40
`suspected_opacity`. La cohorte CheXpert `-1` est exclue de l'accuracy.

| Modele | Accuracy stricte | Sensibilite | Opacite vers normal | Specificite | Normal vers opacite | Abstention |
|---|---:|---:|---:|---:|---:|---:|
| MedGemma baseline | 0,6625 | 0,500 | 0,250 | 0,825 | 0,100 | 0,1625 |
| MedGemma improved_v5 | 0,7250 | 0,850 | 0,150 | 0,600 | 0,350 | 0,0250 |
| MedSigLIP v1 | 0,8125 | 0,875 | 0,075 | 0,750 | 0,200 | 0,0500 |

Conclusion : le prompting modifie fortement le compromis
sensibilite/specificite, mais ne corrige pas de maniere stable la perception
visuelle. MedSigLIP offre le meilleur compromis sur le dev.

## Calibration MedSigLIP

Modele : `google/medsiglip-448`.

- Trois textes zero-shot de classe `normal` et trois de classe
  `suspected_opacity` sont agreges par moyenne des logits.
- ROC-AUC dev : 0,863125.
- Average Precision dev : 0,817782.
- Latence mediane batch : environ 589 ms par image.
- Seuil bas fige : 0,3162987232.
- Seuil haut fige : 0,35.
- Les scores sont des similarites relatives non calibrees.

Deux calibrations ont ete rejetees :

- Des seuils presque identiques autour de 0,33 donnaient 0 % d'abstention.
- Les seuils 0,1763 et 0,2052 supprimaient les faux negatifs, mais reduisaient
  la specificite a 10 % et classaient 82,5 % des normaux comme suspects.

La configuration retenue impose simultanement sensibilite, specificite,
limitation des erreurs dangereuses et une petite zone d'abstention.

## Evaluation finale MedSigLIP

Le manifeste `medsiglip_zero_shot_v1` a ete applique une seule fois au final,
sans recalibrage.

| Metrique | Valeur |
|---|---:|
| Cas definitifs | 20 |
| ROC-AUC | 0,925 |
| Average Precision | 0,937374 |
| Accuracy stricte | 0,850 |
| Sensibilite opacite | 0,800 |
| Opacite vers normal | 0,200 |
| Specificite normale | 0,900 |
| Normal vers opacite | 0,100 |
| Taux uncertain definitif | 0,000 |
| Latence mediane batch | 543,6 ms/image |

Matrice sur les labels definitifs :

| Attendu | normal | suspected_opacity |
|---|---:|---:|
| normal | 9 | 1 |
| suspected_opacity | 2 | 8 |

Sur les dix labels CheXpert `-1`, MedSigLIP retourne 70 %
`suspected_opacity` et 30 % `normal`. Aucune accuracy n'est calculee sur cette
cohorte.

## Registre qualitatif des erreurs finales

Cette revue visuelle est pedagogique et non clinique. Elle ne modifie pas les
metriques officielles.

| Patient | Attendu | Predit | Score | Categorie de revue | Observation |
|---|---|---|---:|---|---|
| patient60277 | normal | suspected_opacity | 0,387381 | FP_projection_or_overlap | Projection semi-verticale et superpositions basales pouvant expliquer le score. |
| patient16222 | suspected_opacity | normal | 0,281406 | FN_subtle_single_view | Asymetrie subtile potentiellement manquee sur la vue frontale unique. |
| patient24846 | suspected_opacity | normal | 0,245085 | FN_multiview_label_mismatch | Etude frontale et laterale, alors que le modele ne recoit que la frontale. |

Ces scores sont eloignes de la zone d'abstention. Les recuperer par un simple
elargissement des seuils degraderait fortement la specificite.

## Decision d'architecture

- MedSigLIP v1 devient le modele de decision principal candidat.
- MedGemma reste utile pour l'explication textuelle et les garde-fous.
- Un desaccord futur entre classifieur et VLM devra etre expose, pas masque.
- LoRA MedGemma reste une extension optionnelle et non prioritaire.
- La comparaison finale MedGemma baseline reste a executer avec sa
  configuration figee.
