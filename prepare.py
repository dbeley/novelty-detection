"""
Script de préparation et de nettoyage des données
"""

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import os
import argparse
import logging
import time
import sys
import errno

logger = logging.getLogger()
temps_debut = time.time()
pd.np.set_printoptions(threshold=sys.maxsize)
DATAPAPERS = os.path.expanduser("~/Documents/Données/datapapers.csv")
NYTDATA = os.path.expanduser("~/Documents/Données/data_big_category_long.csv")
SUPPORTED_DATASETS = ["datapapers", "nytdata"]


def petit_nettoyage(ligne, lem_v=True, lem_n=True, len_elt=2, stopw=[]):
    """ On retire la ponctuation et les nombres, puis (lemmatization, stop_words) ou non """
    lemmatizer = WordNetLemmatizer()
    for elt in ligne:
        if elt in (string.punctuation + string.digits):
            ligne = ligne.replace(elt, " ")
    if lem_v and lem_n:
        liste = [
            lemmatizer.lemmatize(elt, pos="v")
            for elt in ligne.split()
            if lemmatizer.lemmatize(elt, pos="v") not in stopw
        ]
        liste = [
            lemmatizer.lemmatize(elt, pos="n")
            for elt in liste
            if len(lemmatizer.lemmatize(elt, pos="n")) > len_elt
        ]
    elif lem_v and lem_n:
        liste = [
            lemmatizer.lemmatize(elt, pos="v")
            for elt in ligne.split()
            if (lemmatizer.lemmatize(elt, pos="v") not in stopw)
            and (len(elt) > len_elt)
        ]
    elif lem_v and lem_n:
        liste = [
            lemmatizer.lemmatize(elt, pos="n")
            for elt in ligne.split()
            if (lemmatizer.lemmatize(elt, pos="n") not in stopw)
            and (len(elt) > len_elt)
        ]
    else:
        liste = [
            elt
            for elt in ligne.split()
            if (elt not in stopw) and (len(elt) > len_elt)
        ]
    ligne = " ".join(liste)
    return ligne


def main():
    args = parse_args()
    file = str(args.file)

    if file == "datapapers":
        logger.info("Chargement du fichier datapapers.csv")
        data = pd.read_csv(DATAPAPERS, sep="\t", encoding="utf-8")
        data = data[["abstract", "theme"]]
        data_text = data.abstract
    elif file == "nytdata":
        logger.info("Chargement du fichier nytdata")
        data = pd.read_csv(NYTDATA, sep="\t", encoding="utf-8")
        data = data[
            [
                "texts",
                "dates",
                "principal_classifier",
                "second_classifier",
                "third_classifier",
            ]
        ]
        data_text = data.texts.astype(str)
    elif file is None:
        logger.error(
            f"Entrez un jeu de données avec l'argument -f/--file parmi {SUPPORTED_DATASETS}."
        )
        exit()
    else:
        logger.error(
            f"Jeu de données {file} non supporté. Jeux de données supportés : {SUPPORTED_DATASETS}"
        )
        exit()

    # Création du dossier Exports si non existant
    if not os.path.exists("Exports"):
        try:
            os.makedirs("Exports")
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    logger.info("Téléchargement fichiers nltk")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

    logger.info("Chargement des stopwords")
    en_stopw = [str(x) for x in stopwords.words("english")]
    logger.info("Nettoyage des données")
    logger.debug("Mise en minuscule")
    data_text = data_text.apply(lambda x: x.lower())
    logger.debug("Nettoyage")
    data_text = data_text.apply(
        lambda x: petit_nettoyage(x, True, False, 3, en_stopw)
    )
    logger.info("Suppression des petites lignes")

    """ Suppression des petites lignes """
    idx = []

    for i, line in enumerate(data_text):
        if len(line.split()) < 10:
            idx.append(i)
    data.drop(data.index[idx])
    data.index = range(len(data.index))

    if file == "datapapers":
        logger.info("Export du fichier datapapers_clean.csv")
        data.abstract = data_text
        vocab = data_text.str.split()
        vocab.to_csv("Exports/vocabulary.csv")
        data.to_csv("Exports/datapapers_clean.csv")
    elif file == "nytdata":
        logger.info("Export du fichier nytdata_clean.csv")
        data.texts = data_text
        vocab = data_text.str.split()
        vocab.to_csv("Exports/vocabulary.csv")
        data.to_csv("Exports/nytdata_clean.csv")

    logger.info("Runtime : %.2f seconds" % (time.time() - temps_debut))


def parse_args():
    parser = argparse.ArgumentParser(description="Preparation")
    parser.add_argument(
        "--debug",
        help="Display debugging information",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "-f",
        "--file",
        help="File to process (datapapers ou nytdata)",
        type=str,
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    return args


if __name__ == "__main__":
    main()
