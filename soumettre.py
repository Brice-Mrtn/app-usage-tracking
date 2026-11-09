"""Entraîne le modèle complet sur tout le train et écrit submission.csv pour le test."""

import csv
from pathlib import Path

from modele import modele_complet
from traces import charger_test, charger_train

SORTIE = Path(__file__).parent / "submission.csv"

sessions, utilisateurs = charger_train()
modele = modele_complet().fit(sessions, utilisateurs)
predictions = modele.predict(charger_test())

# même format que sample_submission.csv : RowId à partir de 1, dans l'ordre du test
with open(SORTIE, "w", newline="", encoding="utf-8") as fichier:
    lignes = csv.writer(fichier)
    lignes.writerow(["RowId", "prediction"])
    lignes.writerows(enumerate(predictions, start=1))

print(f"{len(predictions)} prédictions écrites dans {SORTIE.name}")
