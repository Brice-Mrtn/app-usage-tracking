# Qui utilise mon appli ?

Identifier l'utilisateur du logiciel Copilote (Infologic) à partir de la trace d'une session : 247 utilisateurs possibles, 3 279 sessions étiquetées pour apprendre, 324 à prédire.

Le modèle soumis obtient un F1 macro de 0,962 en validation croisée, soit 96,6 % des sessions bien attribuées, et 0,976 sur le leaderboard public Kaggle (dépôt du 2 novembre 2026). La démarche, les résultats et leur analyse sont dans le rapport PDF ; ce document décrit le code qui les produit.

## Lancer le code

Python 3.14. Les données ne sont pas fournies : placer `train.csv` et `test.csv` (page Kaggle du challenge) dans un dossier `data/` à côté des scripts.

```bash
pip install -r requirements.txt
python explorer.py    # statistiques descriptives (quelques secondes)
python evaluer.py     # validation croisée, ablations, analyse des erreurs (environ 4 minutes)
python soumettre.py   # entraîne sur tout le train et écrit submission.csv
```

Chaque chiffre du rapport est imprimé par l'un de ces trois scripts. Les découpages de validation croisée et le classifieur ont une graine fixée : deux exécutions donnent les mêmes résultats.

## Les fichiers

| Fichier | Ce qu'il contient | Ce qu'il produit |
|---|---|---|
| `traces.py` | lecture ligne par ligne de `train.csv` et `test.csv`, réparation de la virgule manquante (ligne 2685), découpage de chaque action en cinq attributs (`decouper`), datation par les marqueurs `tXX` ; `charger_train()` et `charger_test()` | des listes de `Session` (navigateur, actions, temps) |
| `modele.py` | les six fonctions qui transforment une session en liste de « mots » (types d'actions, enchaînements, écrans, configurations, fiches, navigateur), leur assemblage en blocs TF-IDF (`assembler`), les deux modèles `modele_simple()` et `modele_complet()` | des pipelines scikit-learn prêts à `fit` / `predict` |
| `explorer.py` | statistiques du train et du test : sessions par utilisateur, longueurs, vocabulaire, part de variance expliquée par l'utilisateur | les chiffres de la partie « Les données » du rapport |
| `evaluer.py` | validation croisée stratifiée à 4 folds des modèles (dont la forêt aléatoire), variantes de pondération TF-IDF, modèle privé d'un bloc à la fois, précision et rappel par utilisateur, erreurs selon la longueur des sessions et la taille des classes, confusions | les tableaux et figures des parties « Évaluation » et « Ce qui compte, ce qui échoue » |
| `soumettre.py` | entraîne le modèle complet sur tout le train et prédit le test | `submission.csv` |

`traces.py` et `modele.py` sont des bibliothèques : ils ne s'exécutent pas seuls, les trois scripts les importent. `explorer.py` n'utilise que `traces.py` ; `evaluer.py` et `soumettre.py` utilisent les deux. Le chemin d'une prédiction est donc `charger_test()` (traces) → `modele_complet()` (modele) → `fit` sur le train puis `predict` sur le test (soumettre).

## Le modèle en bref

- **Lecture** : les fichiers ne sont pas de vrais CSV (de 4 à 14 470 champs par ligne), on les lit ligne par ligne. Chaque champ est découpé en type d'action, écran `( )`, configuration `< >`, fiche `$ $` et mode modification (suffixe `1`). Les marqueurs `tXX` datent les actions (dernier marqueur franchi) et ne sont pas comptés comme des actions.
- **Caractéristiques** : une session devient une ligne de 15 772 colonnes en comptant ce qu'elle contient, par bloc : types d'actions (35), paires et triplets d'actions consécutives (15 023), écrans (415), configurations (219), fiches (76), navigateur (4). Les comptes sont pondérés par TF-IDF (`TfidfVectorizer`, `sublinear_tf=True`), ce qui empêche les sessions longues et les actions banales de dominer.
- **Classifieur** : SVM linéaire (`LinearSVC`, C = 1), un contre tous. Le modèle simple de départ (régression logistique sur 39 colonnes) est conservé dans `modele.py` pour la comparaison.
- **Validation** : `StratifiedKFold(4, shuffle=True, random_state=0)`, quatre étant le maximum puisque l'utilisateur le moins représenté a 4 sessions. Tout (vocabulaire, idf, classifieur) est ré-appris sur le train de chaque fold.
