"""
Fonctions de calcul du score + seuil
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calcul_seuil(vector_historic, m="cosinus", k=1, q=0.95):
    """ Determination du seuil """

    """ Entrées :
                  - vector_historic: les vecteurs représentatifs des documents de l'historique (np.array)
                  - m: la mesure utilisée, Cosinus ou Pearson (string)
                  - k: le nombre de plus proches voisins utilisés (entier strictement positif)
                  - q: ordre du quantile d'ordre q (pourcentage)
        Sorties :
                  Renvoie un seuil pour la detection de la nouveauté (float)
    """

    res = score_novelty(vector_historic, vector_historic, m=m, k=k, s=1)
    return pd.Series(res).quantile(q)


def calcul_score(vector_historic, vector_context, m="cosinus", k=1, s=0):
    """ Score de nouveauté """

    """ Entrées :
                  - vector_historic: les vecteurs représentatifs des documents de l'historique (np.array)
                  - vector_context: les vecteurs représentatifs des documents du contexte (np.array)
                  - m: la mesure utilisée, Cosinus ou Pearson (string)
                  - k: le nombre de plus proches voisins utilisés (entier > 0)
                  - s: un pas de décalage pour considérer le (s+1)eme à (s+k+1)eme plus proche voisin (entier >= 0)
        Sorties :
                  Renvoie un vecteur de score pour chaque document du contexte testé (np.array)
    """

    if m.lower() == "pearson":
        vector_context = np.transpose(
            np.transpose(vector_context) - vector_context.mean(axis=1)
        )
        vector_historic = np.transpose(
            np.transpose(vector_historic) - vector_historic.mean(axis=1)
        )

    c = abs(cosine_similarity(vector_context, vector_historic))
    d = 1 - c
    d.sort(axis=1)

    if k < 1:
        k = 1

    if k < s:
        k = s + 1

    return np.sum(d[:, s : (k + s)], axis=1) / k


def score_novelty(vector_historic, vector_context, m="cosinus", k=1, s=0):
    """ Score de nouveauté """

    """ Entrées :
                  - vector_historic: les vecteurs représentatifs des documents de l'historique (np.array)
                  - vector_context: les vecteurs représentatifs des documents du contexte (np.array)
                  - m: la mesure utilisée, Cosinus ou Pearson (string)
                  - k: le nombre de plus proches voisins utilisés (entier > 0)
                  - s: un pas de décalage pour considérer le (s+1)eme à (s+k+1)eme plus proche voisin (entier >= 0)
        Sorties :
                  Renvoie un vecteur de score pour chaque document du contexte testé (np.array)
    """

    if m.lower() == "pearson":
        vector_context = np.transpose(
            np.transpose(vector_context) - vector_context.mean(axis=1)
        )
        vector_historic = np.transpose(
            np.transpose(vector_historic) - vector_historic.mean(axis=1)
        )

    c = abs(cosine_similarity(vector_context, vector_historic))
    d = 1 - c
    d.sort(axis=1)

    if k < 1:
        k = 1

    if k < s:
        k = s + 1

    return np.sum(d[:, s : (k + s)], axis=1) / k
