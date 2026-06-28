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

## Evaluation finale MedGemma baseline

Le manifeste `medgemma_baseline_v1`, le prompt `baseline_v1` et la generation
deterministe ont ete appliques au meme final, sans modification apres
observation des resultats MedSigLIP. Un premier lancement entierement invalide
a produit 30 erreurs d'environnement avant toute prediction. Il a ete relance
apres correction des dependances et n'est pas compte comme une evaluation du
modele.

| Metrique | Valeur |
|---|---:|
| Cas definitifs | 20 |
| Accuracy stricte | 0,700 |
| Sensibilite opacite stricte | 0,500 |
| Opacite vers normal | 0,200 |
| Specificite normale | 0,900 |
| Normal vers opacite | 0,000 |
| Taux uncertain definitif | 0,200 |
| Latence mediane | 21 697,5 ms/image |
| Erreurs techniques du lancement valide | 0 |

Matrice sur les labels definitifs :

| Attendu | normal | suspected_opacity | uncertain |
|---|---:|---:|---:|
| normal | 9 | 0 | 1 |
| suspected_opacity | 2 | 5 | 3 |

Sur les dix labels CheXpert `-1`, MedGemma retourne 50 % `normal`, 30 %
`suspected_opacity` et 20 % `uncertain`. Aucune accuracy n'est calculee sur
cette cohorte.

Pour une lecture de triage ou `suspected_opacity` et `uncertain` signifient
`a revoir`, MedGemma atteint une sensibilite de 0,800 et une specificite de
0,900 sur ce petit final. Ce point de fonctionnement est identique a celui de
MedSigLIP en triage binaire, mais MedGemma est environ 40 fois plus lent et son
accuracy stricte est inferieure de 0,15.

## Registre qualitatif des erreurs finales

Cette revue visuelle est pedagogique et non clinique. Elle ne modifie pas les
metriques officielles.

### MedSigLIP

| Patient | Attendu | Predit | Score | Categorie de revue | Observation |
|---|---|---|---:|---|---|
| patient60277 | normal | suspected_opacity | 0,387381 | FP_projection_or_overlap | Projection semi-verticale et superpositions basales pouvant expliquer le score. |
| patient16222 | suspected_opacity | normal | 0,281406 | FN_subtle_single_view | Asymetrie subtile potentiellement manquee sur la vue frontale unique. |
| patient24846 | suspected_opacity | normal | 0,245085 | FN_multiview_label_mismatch | Etude frontale et laterale, alors que le modele ne recoit que la frontale. |

Ces scores sont eloignes de la zone d'abstention. Les recuperer par un simple
elargissement des seuils degraderait fortement la specificite.

### MedGemma baseline

| Patient | Attendu | Predit | Confiance | Qualite | Observation |
|---|---|---|---:|---|---|
| patient28185 | normal | uncertain | 0,2 | good | Abstention sur un cas normal. |
| patient16222 | suspected_opacity | uncertain | 0,2 | good | Evite le verdict normal produit par MedSigLIP, mais sans detection stricte. |
| patient40396 | suspected_opacity | uncertain | 0,5 | limited | Opacite non tranchee avec qualite declaree limitee. |
| patient17480 | suspected_opacity | normal | 0,9 | good | Faux rassurant a confiance auto-declaree elevee. |
| patient06523 | suspected_opacity | uncertain | 0,2 | good | Opacite non tranchee malgre une qualite declaree bonne. |
| patient24846 | suspected_opacity | normal | 0,8 | good | Faux rassurant commun avec MedSigLIP et possible decalage de label multi-vues. |

L'analyse appariee montre une complementarite partielle : MedGemma abstient
sur `patient16222`, manque `patient17480` que MedSigLIP detecte, et les deux
modeles manquent `patient24846`. Cette observation motive un routage hybride,
mais ne permet pas d'en definir les regles sur le final.

## Decision d'architecture

- MedSigLIP v1 devient le modele de decision principal candidat.
- MedGemma reste utile pour l'explication textuelle, l'abstention et les
  garde-fous, mais pas comme classifieur principal dans sa version baseline.
- Un desaccord futur entre classifieur et VLM devra etre expose, pas masque.
- Les regles du pipeline hybride seront developpees exclusivement sur le dev.
- Toute performance revendiquee pour ce nouveau pipeline exigera une nouvelle
  cohorte de patients non observee.
- LoRA MedGemma reste une extension optionnelle et non prioritaire.

## Demarrage du developpement hybride

Le notebook `06_hybrid_medsiglip_medgemma_dev.ipynb` formalise une recherche
sur `dev` uniquement. MedSigLIP reste le modele primaire. MedGemma est route
dans des bandes de marge predefinies autour des seuils et la fusion suit la
regle `agreement_or_abstain` : accord conserve, abstention primaire resolue
par une classe secondaire definitive, autre desaccord transforme en
`uncertain`.

Les marges candidates sont 0,000, 0,025, 0,050, 0,075 et 0,100. La selection
interdit de degrader le taux `opacity -> normal` de MedSigLIP et contraint la
specificite, l'abstention et le taux de routage. Le resultat restera une
candidate de developpement jusqu'a son evaluation sur une nouvelle cohorte.
