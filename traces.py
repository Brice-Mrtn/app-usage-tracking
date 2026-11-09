"""Chargement et découpage des traces d'utilisation (train.csv / test.csv)."""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

# train.csv et test.csv (téléchargés depuis Kaggle) sont à placer dans data/
DOSSIER_DONNEES = Path(__file__).parent / "data"

FENETRE = re.compile(r"t(\d+)")
# action(écran)<conf>$chaîne$1 : seule l'action est toujours présente
TOKEN = re.compile(r"(.*?)(?:\((.*?)\))?(?:<(.*?)>)?(?:\$(.*?)\$)?(1)?")
# une ligne du train a une virgule manquante : "...AccueilController)t5"
VIRGULE_MANQUANTE = re.compile(r"([)>$])(t\d+)(?=,|$)")


class Action(NamedTuple):
    type: str
    ecran: str | None  # indiqué seulement quand l'utilisateur change d'écran
    conf: str | None  # indiqué seulement quand la configuration d'écran change
    chaine: str | None  # répétée tant que l'utilisateur travaille sur une fiche
    modification: bool


@dataclass
class Session:
    navigateur: str
    actions: list[Action]
    temps: list[int]  # temps[i] : dernière borne tXX franchie avant actions[i], en secondes
    duree: int  # dernière borne tXX de la session, en secondes


@lru_cache(maxsize=None)
def decouper(token: str) -> Action:
    type_, ecran, conf, chaine, modification = TOKEN.fullmatch(token).groups()
    return Action(type_, ecran, conf, chaine, modification is not None)


def _session(navigateur: str, tokens: list[str]) -> Session:
    actions, temps, maintenant = [], [], 0
    for token in tokens:
        fenetre = FENETRE.fullmatch(token)
        if fenetre:
            maintenant = int(fenetre.group(1))
        else:
            actions.append(decouper(token))
            temps.append(maintenant)
    return Session(navigateur, actions, temps, duree=maintenant)


def _lire(nom: str, etiquete: bool) -> tuple[list[Session], list[str]]:
    sessions, utilisateurs = [], []
    # pas de pd.read_csv : le nombre de champs varie d'une ligne à l'autre
    with open(DOSSIER_DONNEES / nom, encoding="utf-8") as fichier:
        for ligne in fichier:
            champs = VIRGULE_MANQUANTE.sub(r"\1,\2", ligne.rstrip("\n")).split(",")
            if etiquete: #donc seulement pour le train on va isoler l'utilisateur à part
                utilisateurs.append(champs.pop(0))
            sessions.append(_session(champs[0], champs[1:]))
    return sessions, utilisateurs


def charger_train() -> tuple[list[Session], list[str]]:
    """Renvoie les sessions du train et, dans le même ordre, leurs utilisateurs."""
    return _lire("train.csv", etiquete=True)


def charger_test() -> list[Session]:
    return _lire("test.csv", etiquete=False)[0]
