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


def petit_nettoyage(ligne, lem_v=True, lem_n=True, len_elt=2, stopw=[]):
    """ On retire la ponctuation et les nombres, puis (lemmatization, stop_words) ou non """
    lemmatizer = WordNetLemmatizer()
    for elt in ligne:
        if elt in (string.punctuation + string.digits):
            ligne = ligne.replace(elt, ' ')
    if lem_v and lem_n:
        liste = [lemmatizer.lemmatize(elt, pos='v') for elt in ligne.split()
                 if lemmatizer.lemmatize(elt, pos='v') not in stopw]
        liste = [lemmatizer.lemmatize(elt, pos='n') for elt in liste
                 if len(lemmatizer.lemmatize(elt, pos='n')) > len_elt]
    elif lem_v and lem_n:
        liste = [lemmatizer.lemmatize(elt, pos='v') for elt in ligne.split()
                 if (lemmatizer.lemmatize(elt, pos='v') not in stopw) and (len(elt) > len_elt)]
    elif lem_v and lem_n:
        liste = [lemmatizer.lemmatize(elt, pos='n') for elt in ligne.split()
                 if (lemmatizer.lemmatize(elt, pos='n') not in stopw) and (len(elt) > len_elt)]
    else:
        liste = [elt for elt in ligne.split() if (elt not in stopw) and (len(elt) > len_elt)]
    ligne = ' '.join(liste)
    return ligne


def nettoyage(data, lem_v=True, lem_n=True, len_elt=2, stopw=[]):
    """ Nettoyage des données """
    logger.debug("Mise en minuscule")
    data.abstract = data.abstract.apply(lambda x: x.lower())
    logger.debug("Nettoyage")
    data.abstract = data.abstract.apply(lambda x: petit_nettoyage(x, lem_v, lem_n, len_elt, stopw))
    return data


def drop_little_line(data, seuil):
    """ Suppression des petites lignes """
    idx = []

    for i, line in enumerate(data.abstract):
        if len(line.split()) < seuil:
            idx.append(i)
    data.drop(data.index[idx])
    data.index = range(len(data.index))
    return data


def main():
    args = parse_args()


    # Création du dossier Exports si non existant
    if not os.path.exists("Exports"):
        try:
            os.makedirs("Exports")
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    logger.debug("Téléchargement fichiers nltk")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

    logger.debug("Chargement des données")
    data = pd.read_csv(DATAPAPERS, sep="\t", encoding="utf-8")
    data = data[['abstract', 'theme']]

    logger.debug("Chargement des stopwords")
    en_stopw = [str(x) for x in stopwords.words('english')]
    logger.debug("Nettoyage des données")
    data = nettoyage(data, lem_v=True, lem_n=False, len_elt=3, stopw=en_stopw)
    logger.debug("Suppression des petites lignes")
    data = drop_little_line(data, 10)

    vocab = data.abstract.str.split()
    vocab.to_csv("Exports/vocabulary.csv")

    data.to_csv("Exports/datapapers_clean.csv")

    logger.debug("Runtime : %.2f seconds" % (time.time() - temps_debut))


def parse_args():
    parser = argparse.ArgumentParser(description='Preparation')
    parser.add_argument('--debug', help="Display debugging information", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    return args


if __name__ == '__main__':
    main()
