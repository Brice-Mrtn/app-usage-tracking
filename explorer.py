"""Statistiques descriptives du train et du test.

Ce script ne modifie rien et n'entraîne aucun modèle : il lit les deux
fichiers avec traces.py et imprime les chiffres de la partie « Les données »
du rapport (volumes, longueurs, navigateurs, vocabulaire, anomalies, et ce
qui distingue un utilisateur d'un autre)
"""

from collections import Counter

import numpy as np

from modele import COLONNES_RYTHME, rythme
from traces import DOSSIER_DONNEES, VIRGULE_MANQUANTE, Session, charger_test, charger_train


# --- Trois petits outils ----------------------------------------------------


def quartiles(valeurs: list[int]) -> str:
    """Résume une liste de nombres en « min / q1 / médiane / q3 / max »."""
    return " / ".join(str(int(q)) for q in np.percentile(valeurs, [0, 25, 50, 75, 100]))


def vocabulaire(sessions: list[Session], champ: str) -> set[str]:
    """Ensemble des valeurs distinctes d'un attribut (type, écran, conf ou chaine)
    sur toutes les actions de toutes les sessions. Les valeurs vides (attribut
    absent de l'action) ne comptent pas."""
    return {getattr(a, champ) for s in sessions for a in s.actions if getattr(a, champ)}


def part_expliquee(valeurs: np.ndarray, utilisateurs: np.ndarray) -> float:
    """Part de la variance d'une mesure expliquée par l'utilisateur """
    moyenne = valeurs.mean()
    entre = sum(
        (utilisateurs == u).sum() * (valeurs[utilisateurs == u].mean() - moyenne) ** 2
        for u in np.unique(utilisateurs)
    )
    return entre / ((valeurs - moyenne) ** 2).sum()


# --- Chargement -------------------------------------------------------------

# train : liste de Session ; utilisateurs : le graphe de chaque session, dans le meme ordre
train, utilisateurs = charger_train()
test = charger_test()
utilisateurs = np.array(utilisateurs)  # tableau numpy pour les comparaisons


# --- 1. Volumes : combien de sessions, combien d'utilisateurs, classes déséquilibrées

print("== Volumes")
print(f"train : {len(train)} sessions, {len(set(utilisateurs))} utilisateurs ; test : {len(test)} sessions")
par_utilisateur = Counter(utilisateurs)  # {trigramme: nombre de sessions}
print(f"sessions par utilisateur (min / q1 / médiane / q3 / max) : {quartiles(list(par_utilisateur.values()))}")
print(f"utilisateurs avec moins de 10 sessions : {sum(n < 10 for n in par_utilisateur.values())}")


# --- 2. Longueur des sessions, en actions et en secondes, train et test côte à côte
# (si les deux distributions se ressemblent, la validation croisée sur le train
# est représentative du test)

print("\n== Longueur des sessions (min / q1 / médiane / q3 / max)")
for nom, sessions in (("train", train), ("test", test)):
    print(f"{nom} : {quartiles([len(s.actions) for s in sessions])} actions, "
          f"{quartiles([s.duree for s in sessions])} secondes")


# --- 3. Navigateurs : lesquels, et un utilisateur en change-t-il ?
# (s'il n'en change jamais, le navigateur est un indice fiable pour le modèle)

print("\n== Navigateurs")
for nom, sessions in (("train", train), ("test", test)):
    print(f"{nom} : {dict(Counter(s.navigateur for s in sessions).most_common())}")
navigateurs = {u: set() for u in par_utilisateur}  # {trigramme: navigateurs vus}
for session, utilisateur in zip(train, utilisateurs):
    navigateurs[utilisateur].add(session.navigateur)
print(f"utilisateurs vus avec plusieurs navigateurs : {sum(len(n) > 1 for n in navigateurs.values())}")


# --- 4. Vocabulaire : combien de valeurs distinctes par attribut, et combien
# de valeurs du test n'ont jamais été vues dans le train (le modèle ne pourra
# pas les utiliser : ses colonnes sont apprises sur le train)

print("\n== Vocabulaire du train (et part du test absente du train)")
for nom, champ in (("types d'actions", "type"), ("écrans", "ecran"), ("configurations", "conf"), ("chaînes", "chaine")):
    connus, vus = vocabulaire(train, champ), vocabulaire(test, champ)
    print(f"{nom} : {len(connus)} ; test : {len(vus)} dont {len(vus - connus)} inconnus")


# --- 5. Anomalies : ce que le nettoyage a trouvé

print("\n== Anomalies")
# la virgule manquante est repérée par la même expression régulière que celle
# qui la répare à la lecture (traces.VIRGULE_MANQUANTE) ; on relit le fichier
# brut pour la compter
with open(DOSSIER_DONNEES / "train.csv", encoding="utf-8") as fichier:
    print(f"lignes du train avec une virgule manquante avant un tXX : {sum(bool(VIRGULE_MANQUANTE.search(l)) for l in fichier)}")
print(f"sessions sans aucune action : train {sum(not s.actions for s in train)}, test {sum(not s.actions for s in test)}")
# deux sessions sont « en double » si navigateur, actions et temps sont identiques
contenus = Counter((s.navigateur, tuple(s.actions), tuple(s.temps)) for s in train)
print(f"sessions du train en double : {sum(n - 1 for n in contenus.values())}")


# --- 6. Ce qui signe un utilisateur : rythme ou contenu ?
# Pour chaque mesure, la part de variance expliquée par l'utilisateur dit si
# la mesure est stable chez une même personne. 

print("\n== Part de la variance expliquée par l'utilisateur")
mesures = rythme(train)  # tableau (sessions × 8 mesures), même fonction que le bloc « rythme » du modèle
for colonne, nom in enumerate(COLONNES_RYTHME):
    print(f"{part_expliquee(mesures[:, colonne], utilisateurs):.2f}  {nom}")
# part de chaque type d'action dans chaque session : compte / nombre total d'actions
comptes = [Counter(a.type for a in s.actions) for s in train]
parts = {
    type_: np.array([c[type_] / max(c.total(), 1) for c in comptes])  # max(..., 1) : la session vide
    for type_ in vocabulaire(train, "type")
}
scores = sorted(((part_expliquee(v, utilisateurs), t) for t, v in parts.items()), reverse=True)
for score, type_ in scores[:5]:  # les cinq types d'actions les plus « signants »
    print(f"{score:.2f}  part de « {type_} »")
