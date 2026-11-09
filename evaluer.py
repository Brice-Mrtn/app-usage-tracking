"""Validation croisée sur le train.

Trois sorties : la comparaison des modèles, l'influence de chaque bloc de
caractéristiques (modèle complet privé d'un bloc), et le détail des erreurs
du modèle complet.
"""

from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from modele import BLOCS_COMPLET, BLOCS_SIMPLE, modele_complet, modele_simple
from traces import Session, charger_train

# 4 folds au maximum : l'utilisateur le moins représenté n'a que 4 sessions.
# Chacun a ainsi au moins 1 session en validation et 3 à l'entraînement.
# random_state fixé pour comparer tous les modèles sur les mêmes découpages.
FOLDS = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)


def valider(
    modele: Pipeline, sessions: list[Session], utilisateurs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Entraîne et évalue le modèle sur chaque découpage.

    Renvoie une ligne de scores par fold (accuracy, F1 macro, accuracy sur le
    train) et, pour chaque session, la prédiction faite quand elle était en
    validation, donc par un modèle qui ne l'avait jamais vue.
    """
    scores = []
    predictions = np.empty(len(sessions), dtype=object)
    for train, validation in FOLDS.split(sessions, utilisateurs):
        sessions_train = [sessions[i] for i in train]
        # tout est ré-appris sur le train du fold (vocabulaire, IDF, échelle,
        # classifieur) : rien ne fuit de la validation vers l'apprentissage
        modele.fit(sessions_train, utilisateurs[train])
        predictions[validation] = modele.predict([sessions[i] for i in validation])
        scores.append(
            (
                accuracy_score(utilisateurs[validation], predictions[validation]),
                f1_score(
                    utilisateurs[validation],
                    predictions[validation],
                    average="macro",
                    zero_division=0,
                ),
                accuracy_score(utilisateurs[train], modele.predict(sessions_train)),
            )
        )
    return np.array(scores), predictions


def afficher(nom: str, scores: np.ndarray) -> None:
    moyenne, ecart = scores.mean(axis=0), scores.std(axis=0)
    print(
        f"{nom:28s} {moyenne[0]:.3f} ± {ecart[0]:.3f}   "
        f"{moyenne[1]:.3f} ± {ecart[1]:.3f}   {moyenne[2]:.3f}",
        flush=True,
    )


sessions, utilisateurs = charger_train()
utilisateurs = np.array(utilisateurs)

print(f"{'Modèle':28s} {'accuracy':15s} {'F1 macro':15s} accuracy train")
afficher("simple (rég. logistique)", valider(modele_simple(), sessions, utilisateurs)[0])
afficher(
    "simple + rythme",
    valider(modele_simple((*BLOCS_SIMPLE, "rythme")), sessions, utilisateurs)[0],
)
# un modèle non linéaire sur le même petit tableau, pour comparaison
foret = modele_simple()
foret.set_params(classifieur=RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))
afficher("simple, forêt aléatoire", valider(foret, sessions, utilisateurs)[0])
scores, predictions = valider(modele_complet(), sessions, utilisateurs)
afficher("complet (SVM linéaire)", scores)
afficher(
    "complet + rythme",
    valider(modele_complet((*BLOCS_COMPLET, "rythme")), sessions, utilisateurs)[0],
)
# Mêmes colonnes que le modèle complet, classifieur du modèle simple : sépare
# l'effet des caractéristiques de celui de l'algorithme. C élevé parce que les
# colonnes TF-IDF sont petites (lignes de norme 1) et seraient trop pénalisées.
logistique = modele_complet().set_params(classifieur=LogisticRegression(C=100, max_iter=1000))
afficher("complet, rég. logistique", valider(logistique, sessions, utilisateurs)[0])

print("\nInfluence de la pondération TF-IDF : modèle complet, blocs de mots autrement pondérés")
# Les trois morceaux de TF-IDF retirés un à un (le modèle complet ci-dessus est
# la référence : 1 + ln(compte), × idf, ÷ norme de la ligne).
BLOCS_MOTS = ("actions", "enchainements", "ecrans", "confs", "chaines")
for nom, reglages in [
    ("compte brut", dict(sublinear_tf=False, use_idf=False, norm=None)),
    ("compte ÷ norme", dict(sublinear_tf=False, use_idf=False, norm="l2")),
    ("1 + ln(compte) ÷ norme", dict(sublinear_tf=True, use_idf=False, norm="l2")),
    ("compte × idf ÷ norme", dict(sublinear_tf=False, use_idf=True, norm="l2")),
]:
    variante = modele_complet().set_params(
        **{f"caracteristiques__{bloc}__{cle}": valeur for bloc in BLOCS_MOTS for cle, valeur in reglages.items()}
    )
    afficher(nom, valider(variante, sessions, utilisateurs)[0])

print("\nInfluence des blocs : modèle complet privé d'un bloc")
for bloc in BLOCS_COMPLET:
    restants = [b for b in BLOCS_COMPLET if b != bloc]
    afficher(f"sans {bloc}", valider(modele_complet(restants), sessions, utilisateurs)[0])

juste = predictions == utilisateurs
print(f"\nErreurs du modèle complet : {(~juste).sum()} sessions sur {len(juste)}")

print("\nSelon le nombre de sessions de l'utilisateur dans le train")
effectifs = Counter(utilisateurs)
n_sessions = np.array([effectifs[u] for u in utilisateurs])
for bas, haut in [(4, 5), (6, 9), (10, 19), (20, n_sessions.max())]:
    groupe = (n_sessions >= bas) & (n_sessions <= haut)
    print(
        f"  {bas:2d} à {haut:2d} sessions : {len(set(utilisateurs[groupe])):3d} utilisateurs, "
        f"{groupe.sum():4d} sessions, accuracy {juste[groupe].mean():.3f}"
    )

print("\nSelon la longueur de la session (quarts du train)")
longueurs = np.array([len(s.actions) for s in sessions])
quart = np.digitize(longueurs, np.percentile(longueurs, [25, 50, 75]), right=True)
for numero in range(4):
    groupe = quart == numero
    print(
        f"  {longueurs[groupe].min():5d} à {longueurs[groupe].max():5d} actions : "
        f"{(~juste[groupe]).sum():3d} erreurs, accuracy {juste[groupe].mean():.3f}"
    )

print("\nAccuracy et F1 macro sur les 3279 prédictions hors-fold réunies")
print(
    f"  accuracy {accuracy_score(utilisateurs, predictions):.4f}   "
    f"F1 macro {f1_score(utilisateurs, predictions, average='macro', zero_division=0):.4f}"
)

print("\nPrécision, rappel et F1 par utilisateur : les deux plus actifs, puis les trois plus mal reconnus")
classes = sorted(effectifs)
precision, rappel, f1_par, _ = precision_recall_fscore_support(
    utilisateurs, predictions, labels=classes, zero_division=0
)
actifs = [classes.index(u) for u, _ in effectifs.most_common(2)]
pires = sorted(range(len(classes)), key=lambda i: f1_par[i])[:3]
for i in actifs + pires:
    u = classes[i]
    bons = int(((utilisateurs == u) & (predictions == u)).sum())
    print(
        f"  {u} : {effectifs[u]:2d} sessions, {int((predictions == u).sum()):2d} prédites comme lui, "
        f"{bons:2d} bien attribuées -> précision {precision[i]:.3f}, rappel {rappel[i]:.3f}, F1 {f1_par[i]:.3f}"
    )
print(f"  {int((f1_par == 1).sum())} utilisateurs sur {len(classes)} ont un F1 de 1")

# Ce que le F1 macro voit et que l'accuracy ne voit pas : des erreurs concentrées
# sur les petits utilisateurs. Simulation sur ces mêmes prédictions.
petits = [u for u in classes if effectifs[u] <= 5]
plus_actif = effectifs.most_common(1)[0][0]
simulation = predictions.copy()
for u in petits:
    simulation[utilisateurs == u] = plus_actif
print(
    f"\nSi les {len(petits)} utilisateurs à 4 ou 5 sessions étaient tous attribués à {plus_actif} : "
    f"accuracy {accuracy_score(utilisateurs, simulation):.3f}, "
    f"F1 macro {f1_score(utilisateurs, simulation, average='macro', zero_division=0):.3f}"
)

print("\nConfusions les plus fréquentes (utilisateur réel -> prédit)")
confusions = Counter(zip(utilisateurs[~juste], predictions[~juste]))
for (reel, predit), n in confusions.most_common(8):
    print(f"  {reel} -> {predit} : {n} sessions")
mal_reconnus = Counter(utilisateurs[~juste])
print(f"\n{len(mal_reconnus)} utilisateurs sur {len(effectifs)} ont au moins une erreur")
