# Modèles ONNX

Les binaires restent locaux et ne sont pas versionnés dans Git. Chaque variante
possède une configuration dans `config/` et une entrée dans
`config/onnx_models.json`.

## Modèle 1 — `u_ones`

Le premier export doit être constitué de **deux fichiers placés ensemble** :

```text
models/arvi_cxr_classifier_u_ones.onnx
models/arvi_cxr_classifier_u_ones.onnx.data
```

Le graphe `.onnx` reçu pèse 90 449 octets et référence des poids externes dans
le fichier `.onnx.data`. Sans ce second fichier, ONNX Runtime ne peut pas créer
la session. Les deux artefacts sont maintenant disponibles sur le poste de
développement ; l'interface signale explicitement toute copie incomplète.

Le contrat vérifié dans `notebooks/07_train_resnet18_u_ones.ipynb` est décrit
par `config/onnx_u_ones.json` :

- ResNet-18 binaire ;
- entrée `input`, RGB NCHW, `1 × 3 × 224 × 224` ;
- normalisation ImageNet ;
- sortie `logits` dans l'ordre `normal`, `suspected_opacity` ;
- stratégie `u_ones` : les labels `-1` et manquants sont regroupés avec
  `suspected_opacity`.

Le notebook n'a pas calibré de seuil d'abstention. Les seuils restent donc à
zéro jusqu'à une validation sur le split de développement.

## Modèle 2 — `u_zeros`

Le deuxième export suit le même contrat technique et utilise :

```text
models/arvi_cxr_classifier_u_zeros.onnx
models/arvi_cxr_classifier_u_zeros.onnx.data
```

Sa configuration est `config/onnx_u_zeros.json` et son entraînement est
reproduit dans `notebooks/08_train_resnet18_u_zeros.ipynb`. La différence
expérimentale est la politique de labels : les valeurs `-1` et manquantes sont
regroupées avec `normal`, tandis que seule la valeur `1` produit
`suspected_opacity`.

## Ajouter le modèle 3

Pour le dernier modèle :

1. placer le ou les fichiers exportés dans ce dossier ;
2. ajouter une configuration dédiée dans `config/` ;
3. ajouter la variante dans `config/onnx_models.json` ;
4. vérifier le contrat suivant :

- dimensions, mode couleur et disposition de l'entrée (`NCHW` ou `NHWC`) ;
- interpolation utilisée pour redimensionner l'image ;
- mise à l'échelle, moyenne et écart-type de normalisation ;
- ordre des classes en sortie ;
- nature de la sortie (`logits` ou `probabilities`) ;
- noms d'entrée et de sortie s'ils doivent être imposés ;
- seuil de confiance et marge d'abstention validés sur le jeu de développement.

Pour utiliser temporairement un autre emplacement du graphe sans modifier le
dépôt :

```bash
export ONNX_MODEL_PATH=/chemin/vers/modele.onnx
```

Lorsque le modèle emploie des données externes, son fichier `.onnx.data` doit
se trouver dans le même dossier que le graphe.
