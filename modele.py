"""Caractéristiques et modèles.

Une session est une liste d'actions de longueur variable, alors qu'un
classifieur attend un tableau à colonnes fixes. Chaque « bloc » ci-dessous
produit un groupe de colonnes à partir d'une session ; un modèle est un
assemblage de blocs suivi d'un classifieur.

evaluer.py et soumettre.py utilisent ces mêmes fonctions : ce qui est mesuré
en validation croisée est exactement ce qui est soumis.
"""

from collections.abc import Callable, Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from traces import Session


# --- Des sessions aux « mots » ----------------------------------------------
# Chaque fonction transforme une session en liste de mots ; un vectoriseur
# crée ensuite une colonne par mot distinct vu à l'entraînement et les compte.


def types_actions(session: Session) -> list[str]:
    return [action.type for action in session.actions]


def enchainements(session: Session) -> list[str]:
    """Paires et triplets d'actions consécutives : conserve l'ordre local."""
    types = types_actions(session)
    paires = [f"{a} > {b}" for a, b in zip(types, types[1:])]
    triplets = [f"{a} > {b} > {c}" for a, b, c in zip(types, types[1:], types[2:])]
    return paires + triplets


def ecrans(session: Session) -> list[str]:
    return [action.ecran for action in session.actions if action.ecran]


def confs(session: Session) -> list[str]:
    return [action.conf for action in session.actions if action.conf]


def chaines(session: Session) -> list[str]:
    return [action.chaine for action in session.actions if action.chaine]


def navigateur(session: Session) -> list[str]:
    return [session.navigateur]


# --- Rythme de la session ---------------------------------------------------

COLONNES_RYTHME = (
    "log du nombre d'actions",
    "log de la durée",
    "actions par seconde",
    "médiane d'actions par fenêtre de 5 s",
    "écart-type d'actions par fenêtre",
    "maximum d'actions par fenêtre",
    "part de fenêtres sans action",
    "log de la plus longue pause",
)


def _plus_longue_suite(vrais: np.ndarray) -> int:
    """Longueur de la plus longue suite de True consécutifs."""
    bords = np.flatnonzero(np.diff(np.concatenate(([0], vrais.astype(np.int8), [0]))))
    return (bords[1::2] - bords[::2]).max(initial=0)


def rythme(sessions: list[Session]) -> np.ndarray:
    """Une ligne par session, une colonne par mesure de COLONNES_RYTHME."""
    lignes = []
    for session in sessions:
        n_actions = len(session.actions)
        # nombre d'actions dans chaque fenêtre de 5 secondes de la session
        par_fenetre = np.bincount(
            np.array(session.temps, dtype=int) // 5, minlength=session.duree // 5 + 1
        )
        vides = par_fenetre == 0
        lignes.append(
            [
                np.log1p(n_actions),
                np.log1p(session.duree),
                n_actions / session.duree,
                np.median(par_fenetre),
                par_fenetre.std(),
                par_fenetre.max(),
                vides.mean(),
                np.log1p(5 * _plus_longue_suite(vides)),
            ]
        )
    return np.array(lignes)


# --- Assemblage des blocs ---------------------------------------------------


def tfidf(mots: Callable[[Session], list[str]]) -> TfidfVectorizer:
    # sublinear_tf remplace le compte par 1 + log(compte) : un mot répété
    # 1 000 fois ne pèse pas 1 000 fois plus. L'IDF réduit le poids des mots
    # communs à toutes les sessions, et chaque ligne est ramenée à la norme 1
    # pour qu'une session longue ne pèse pas plus qu'une courte.
    return TfidfVectorizer(analyzer=mots, sublinear_tf=True)


def assembler(blocs: Iterable[str]) -> FeatureUnion:
    """Met côte à côte les colonnes des blocs demandés."""
    disponibles = {
        # part de chaque type d'action dans la session (compte / total)
        "parts_actions": TfidfVectorizer(analyzer=types_actions, use_idf=False, norm="l1"),
        "actions": tfidf(types_actions),
        "enchainements": tfidf(enchainements),
        "ecrans": tfidf(ecrans),
        "confs": tfidf(confs),
        "chaines": tfidf(chaines),
        # une colonne 0/1 par navigateur (codage disjonctif complet)
        "navigateur": CountVectorizer(analyzer=navigateur),
        "rythme": make_pipeline(FunctionTransformer(rythme), StandardScaler()),
    }
    return FeatureUnion([(nom, disponibles[nom]) for nom in blocs])


# --- Modèles ----------------------------------------------------------------

BLOCS_SIMPLE = ("parts_actions", "navigateur")
BLOCS_COMPLET = ("actions", "enchainements", "ecrans", "confs", "chaines", "navigateur")


def modele_simple(blocs: Iterable[str] = BLOCS_SIMPLE) -> Pipeline:
    """Régression logistique sur un petit tableau lisible : le point de départ."""
    return Pipeline(
        [
            ("caracteristiques", assembler(blocs)),
            # même échelle pour toutes les colonnes ; with_mean=False parce
            # que le tableau est creux (on ne peut pas le centrer)
            ("echelle", StandardScaler(with_mean=False)),
            ("classifieur", LogisticRegression(C=1.0, max_iter=2000)),
        ]
    )


def modele_complet(blocs: Iterable[str] = BLOCS_COMPLET) -> Pipeline:
    """SVM linéaire sur les mots et enchaînements pondérés : le modèle soumis."""
    return Pipeline(
        [
            ("caracteristiques", assembler(blocs)),
            # max_iter élevé : sans effet sur les blocs TF-IDF, mais nécessaire
            # pour converger quand on ajoute le bloc rythme, non borné
            ("classifieur", LinearSVC(C=1.0, max_iter=20000, random_state=0)),
        ]
    )
