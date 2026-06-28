# Deploiement Hugging Face Space

Le Space execute MedSigLIP et MedGemma sur un GPU distant. Aucun poids de
modele n'est charge sur les postes de l'equipe.

## Prerequis

1. Accepter les conditions des modeles `google/medsiglip-448` et
   `google/medgemma-4b-it` sur Hugging Face.
2. Creer un token Hugging Face avec un acces en lecture aux modeles gates.
3. Creer un Space Gradio. Le depot suggere un `t4-small`, mais le materiel doit
   etre selectionne manuellement dans les parametres du Space.
4. Ajouter le token comme Secret nomme `HF_TOKEN`. Ne jamais l'ecrire dans le
   code, le README ou une variable publique.

## Publier le depot dans le Space

Depuis le clone local, ajouter le Space comme second remote puis pousser :

```powershell
git remote add hf-space https://huggingface.co/spaces/UTILISATEUR/NOM_DU_SPACE
git push hf-space main
```

Le fichier `README.md` configure `app/hf_space_app.py` comme point d'entree.
Chaque Space Gradio expose egalement les fonctions `classify` et `explain`
comme API.

## Utilisation

- Le premier clic de classification charge MedSigLIP puis conserve le modele
  en memoire.
- Le premier clic d'explication charge MedGemma. Les appels suivants sur la
  meme image utilisent le cache de l'application.
- Les deux fonctions sont limitees a une execution GPU simultanee afin
  d'eviter les pics de memoire.
- MedGemma produit une analyse independante. Sa classe ne remplace jamais la
  decision MedSigLIP.

## Cout et arret

Le tarif affiche par Hugging Face pour un `t4-small` est de 0,40 USD par heure
au 28 juin 2026. Le materiel payant reste facture lorsqu'il est actif. Mettre
le Space en pause depuis `Settings` des que la session de travail ou la
demonstration est terminee; le temps en pause n'est pas facture.

Si les deux modeles depassent la memoire du T4 lors du test reel, utiliser un
L4 24 Go plutot qu'un T4 medium : le T4 medium augmente la RAM hote, mais garde
16 Go de VRAM.

References officielles :

- https://huggingface.co/docs/hub/main/spaces-overview
- https://huggingface.co/docs/hub/main/spaces-gpus
- https://huggingface.co/docs/hub/en/spaces-config-reference

## Limites

- Utiliser uniquement des donnees de projet de-identifiees et autorisees.
- Le cache et le disque standard d'un Space ne sont pas persistants.
- Une reconstruction ou une pause impose un nouveau chargement des poids.
- Ce deploiement est une demonstration pedagogique, pas un service clinique.
